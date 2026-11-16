"""Evaluate determinant-cone projections and feasible finite Gaussian mixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import qmc, t


OUT = Path("results")
EPSILONS = (.005, .01, .02, .035, .05, .075, .1, .15, .25, .35, .5)
MIXTURE_EPSILONS = (.01, .02, .05, .1, .25)
REPLICATES = 12
QMC_POWER = 16
QMC_SEED = 6210


def density(x):
    return np.exp(-.5 * np.square(x)) / np.sqrt(2 * np.pi)


def bin_moments(edges):
    """Normal bin masses, centroids and unconditional squared distortions."""
    left, right = edges[:-1], edges[1:]
    mass = np.where(left >= 0, ndtr(-left) - ndtr(-right), ndtr(right) - ndtr(left))
    first_raw = density(left) - density(right)
    edge_times_density = np.zeros_like(edges)
    finite = np.isfinite(edges)
    edge_times_density[finite] = edges[finite] * density(edges[finite])
    second_raw = mass + edge_times_density[:-1] - edge_times_density[1:]
    assert np.all(mass > 0)
    means = first_raw / mass
    distortion = second_raw - first_raw * means
    assert np.min(distortion) >= -2e-16
    return mass, means, np.maximum(distortion, 0)


def companding_edges(lo, hi, bins):
    """Equal-width cells in the CDF of N(0,3), including infinite tails."""
    scale = np.sqrt(3)
    cdf_edges = np.linspace(ndtr(lo / scale), ndtr(hi / scale), bins + 1)
    edges = scale * ndtri(cdf_edges)
    edges[0], edges[-1] = lo, hi
    return edges


def mixture_cost(epsilon, positive_a_bins, b_bins):
    # Exploit reflection in A. Retain the independent C=epsilon*G3 and
    # orthogonally project (C,0) onto span((a,b)) on each noncentral cell.
    a_mass, a_mean, a_dist = bin_moments(companding_edges(0., np.inf, positive_a_bins))
    b_mass, b_mean, b_dist = bin_moments(companding_edges(-np.inf, np.inf, b_bins))
    a2 = np.square(a_mean)[:, None]
    b2 = np.square(epsilon * b_mean)[None, :]
    projection = 2 * epsilon**2 * (a_mass @ (b2 / (a2 + b2)) @ b_mass)
    distortion = 2 * (a_dist.sum() + a_mass.sum() * epsilon**2 * b_dist.sum())
    assert abs(2 * a_mass.sum() * b_mass.sum() - 1) < 1e-13
    assert np.min(a_mean) > 0 and np.allclose(b_mean, -b_mean[::-1])
    return distortion + projection, distortion, projection


def smallest_squared_singular(a, b, c, d):
    determinant_sq = np.square(a * d - b * c)
    frobenius_sq = a*a + b*b + c*c + d*d
    root = np.sqrt(np.maximum(frobenius_sq**2 - 4 * determinant_sq, 0))
    return np.divide(2 * determinant_sq, frobenius_sq + root,
                     out=np.zeros_like(root), where=(frobenius_sq + root) > 0)


def projection_sweep():
    regularizations = {
        'singular': lambda e: 0., 'eps6': lambda e: e**6,
        'eps3': lambda e: e**3, '0.3eps2': lambda e: .3*e**2,
        'eps2': lambda e: e**2, '10eps2': lambda e: 10*e**2,
    }
    rng = np.random.default_rng(QMC_SEED - 1)
    check = rng.normal(size=(256, 2, 2))
    values = smallest_squared_singular(check[:, 0, 0], check[:, 0, 1], check[:, 1, 0], check[:, 1, 1])
    max_svd_error = float(np.max(np.abs(values - np.linalg.svd(check, compute_uv=False)[:, 1]**2)))
    assert max_svd_error < 1e-12
    rows = []
    for replicate in range(REPLICATES):
        normals = ndtri(qmc.Sobol(d=7, scramble=True, seed=QMC_SEED + replicate).random_base2(QMC_POWER)).T
        g1, g2, g3, n11, n12, n21, n22 = normals
        for epsilon in EPSILONS:
            for label, spec in regularizations.items():
                delta = spec(epsilon)
                noise = np.sqrt(delta)
                projection = smallest_squared_singular(
                    g1 + noise*n11, epsilon*g2 + noise*n12,
                    epsilon*g3 + noise*n21, noise*n22,
                ).mean()
                gaussian = epsilon**2/2 + delta
                rows.append(dict(epsilon=epsilon, regularization=label, delta=delta,
                                 condition=(1+delta)/delta if delta else np.inf,
                                 replicate=replicate, projection=projection,
                                 gaussian_exact=gaussian, achieved_ratio=gaussian/projection))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'projection_replicates.csv', index=False)
    summary = df.groupby(['epsilon', 'regularization', 'delta', 'condition', 'gaussian_exact'], as_index=False).agg(
        projection=('projection', 'mean'), spread=('projection', 'std'))
    halfwidth = t.ppf(.975, REPLICATES - 1) * summary.spread / np.sqrt(REPLICATES)
    summary['projection_lo'] = summary.projection - halfwidth
    summary['projection_hi'] = summary.projection + halfwidth
    summary['achieved_ratio'] = summary.gaussian_exact / summary.projection
    summary['achieved_ratio_lo'] = summary.gaussian_exact / summary.projection_hi
    summary['achieved_ratio_hi'] = summary.gaussian_exact / summary.projection_lo
    assert np.all(summary.projection_lo > 0)
    summary.to_csv(OUT / 'projection_sweep.csv', index=False)
    return summary, max_svd_error


def mixture_sweep(projections):
    references = projections[projections.regularization == 'singular'].set_index('epsilon')
    rows = []
    budgets = np.unique(np.r_[1, 3, 5, 9, 17, 33, 65, np.round(np.geomspace(100, 10**5, 36)).astype(int)])
    for epsilon in MIXTURE_EPSILONS:
        ref = references.loc[epsilon]
        exact_gaussian = epsilon**2/2
        incumbent = (exact_gaussian, 1, 0, 0, np.nan, np.nan)
        for budget in budgets:
            if budget >= 2:
                ideal_b = np.sqrt(epsilon * budget / 2)
                b_choices = np.clip(np.round(ideal_b * np.array([.5, .75, 1., 1.33, 2.])), 1, budget//2).astype(int)
                b_choices = np.unique(b_choices)
                for b_bins in b_choices:
                    a_bins = budget//(2*b_bins)
                    candidate, distortion, projection = mixture_cost(epsilon, a_bins, b_bins)
                    if candidate < incumbent[0]:
                        incumbent = (candidate, 2*a_bins*b_bins, a_bins, b_bins, distortion, projection)
            cost, used, a_bins, b_bins, distortion, projection = incumbent
            theoretical = epsilon**2 * min(np.pi/48, np.pi/(6*budget**2))
            if budget == 1:
                theoretical = exact_gaussian
            assert cost >= theoretical - 1e-13 and used <= budget
            if used > 1:
                assert cost >= ref.projection_hi * .99
            rows.append(dict(epsilon=epsilon, budget=budget, used_components=used, a_positive_bins=a_bins,
                             b_bins=b_bins, gaussian_exact=exact_gaussian, feasible_cost=cost,
                             theoretical_lower=theoretical, quantization_cost=distortion,
                             row_projection_cost=projection,
                             fixed_direction_projection=ref.projection))
    df = pd.DataFrame(rows)
    for _, group in df.groupby('epsilon'):
        assert np.all(np.diff(group.feasible_cost) <= 1e-14)
    df.to_csv(OUT / 'mixture_scan.csv', index=False)
    return df


def main():
    OUT.mkdir(exist_ok=True)
    projections, _ = projection_sweep()
    mixture_sweep(projections)


if __name__ == "__main__":
    main()
