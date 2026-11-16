"""Measure tangent curvature under transport and excess loss along a rank-one curve."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


OUT = Path("results")
ROTATION = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
X_INDEX = np.array([0, 0, 1])
Z_INDEX = np.array([2, 3, 3])
WEAK = np.array([1, 0, 1]) / np.sqrt(2)
IDENTITY = np.eye(4)


def draw_matrices(rng, n, epsilon, delta):
    normals = rng.normal(size=(n, 3))
    matrices = np.zeros((n, 2, 2))
    matrices[:, 0, 0] = normals[:, 0]
    matrices[:, 0, 1] = epsilon * normals[:, 1]
    matrices[:, 1, 0] = epsilon * normals[:, 2]
    if delta:
        matrices += np.sqrt(delta) * rng.normal(size=matrices.shape)
    return matrices


def encode(matrices):
    rotated = ROTATION @ matrices
    return np.column_stack((rotated[:, 0, 0], rotated[:, 0, 1],
                            rotated[:, 1, 1], -rotated[:, 1, 0]))


def project_rank_one(matrices):
    left, singular, right = np.linalg.svd(matrices)
    projected = (left[:, :, :1] * singular[:, None, :1]) @ right[:, :1, :]
    return projected, singular[:, 1] ** 2


def source_covariance(epsilon, delta):
    basis = np.zeros((3, 2, 2))
    basis[0, 0, 0] = 1
    basis[1, 0, 1] = epsilon
    basis[2, 1, 0] = epsilon
    factors = encode(basis)
    return factors.T @ factors + delta * IDENTITY


def sample_gram(features):
    products = features[:, X_INDEX] * features[:, Z_INDEX]
    return products.T @ products / len(features)


def gaussian_gram(covariance, x=X_INDEX, z=Z_INDEX):
    """Second moments of the bilinear features under a centered Gaussian."""
    return (np.outer(covariance[x, z], covariance[x, z])
            + covariance[np.ix_(x, x)] * covariance[np.ix_(z, z)]
            + covariance[np.ix_(x, z)] * covariance[np.ix_(z, x)])


def summarize(results, group_columns, metrics):
    rows = []
    for keys, group in results.groupby(group_columns):
        row = dict(zip(group_columns, keys))
        count = len(group)
        row["seeds"] = count
        multiplier = student_t.ppf(.975, count - 1) / np.sqrt(count)
        for metric in metrics:
            mean = group[metric].mean()
            halfwidth = multiplier * group[metric].std(ddof=1)
            row.update({f"{metric}_mean": mean, f"{metric}_lo": mean - halfwidth,
                        f"{metric}_hi": mean + halfwidth})
        rows.append(row)
    return pd.DataFrame(rows)


def smoothing_and_interpolation():
    transport_rows = []
    smoothing_rows = []
    same_x = X_INDEX[:, None] == X_INDEX
    same_z = Z_INDEX[:, None] == Z_INDEX
    for epsilon in (.01, .05, .25):
        for source, delta in (("singular", 0), ("spd_e3", epsilon**3)):
            source_cov = source_covariance(epsilon, delta)
            source_gram = gaussian_gram(source_cov)
            source_eigenvalue = np.linalg.eigvalsh(source_gram)[0]
            source_weak = WEAK @ source_gram @ WEAK
            gaussian_radius = epsilon**2 / 2 + delta
            for seed in range(3701, 3713):
                n = 50_000
                matrices = draw_matrices(np.random.default_rng(seed), n, epsilon, delta)
                projected, squared_costs = project_rank_one(matrices)
                y = encode(matrices)
                y_proj = encode(projected)
                projection_cost = squared_costs.mean()
                projected_cov = y_proj.T @ y_proj / n
                projected_gram = sample_gram(y_proj)
                setting = dict(source=source, epsilon=epsilon, delta=delta, seed=seed, n=n)

                for alpha in (0, .25, .5, .75, .9, .99, 1):
                    cost = alpha**2 * projection_cost
                    beta = np.sqrt(cost / gaussian_radius)
                    assert beta <= 1
                    contraction = IDENTITY.copy()
                    contraction[1, 1] = 1 - beta
                    comparisons = {
                        "nonlinear_interpolation": sample_gram(y + alpha * (y_proj - y)),
                        "gaussian_contraction": gaussian_gram(contraction @ source_cov @ contraction),
                        "gaussian_additive_noise": gaussian_gram(source_cov + cost / 4 * IDENTITY),
                    }
                    for method, gram in comparisons.items():
                        eigenvalue = np.linalg.eigvalsh(gram)[0]
                        weak = WEAK @ gram @ WEAK
                        transport_rows.append(dict(
                            **setting, alpha=alpha, method=method, cost=cost,
                            F_exact=gaussian_radius, budget_fraction=cost / gaussian_radius,
                            beta=beta, min_eig=eigenvalue, min_eig_relative=eigenvalue / source_eigenvalue,
                            q=weak, q_relative=weak / source_weak,
                        ))

                # Integrate the smoothing noise conditional on each projected observation.
                for noise in (.001, .01, .1, 1, 10):
                    tau2 = noise * projection_cost / 4
                    smoothed = projected_gram + tau2 * (
                        same_x * projected_cov[np.ix_(Z_INDEX, Z_INDEX)]
                        + same_z * projected_cov[np.ix_(X_INDEX, X_INDEX)]
                    ) + tau2**2 * (same_x & same_z)
                    matched = gaussian_gram(projected_cov + tau2 * IDENTITY)
                    weak = tau2 * np.trace(projected_cov) / 2 + tau2**2
                    matched_weak = WEAK @ matched @ WEAK
                    assert abs(WEAK @ smoothed @ WEAK - weak) < 1e-15
                    eigenvalue = np.linalg.eigvalsh(smoothed)[0]
                    assert eigenvalue > 0
                    cost = projection_cost + 4 * tau2
                    smoothing_rows.append(dict(
                        **setting, lambda_noise=noise, tau2=tau2, C_projection=projection_cost,
                        cost_bound=cost, F_exact=gaussian_radius, budget_fraction=cost / gaussian_radius,
                        q_smoothed=weak, q_gaussian=matched_weak, q_ratio=matched_weak / weak,
                        min_eig_smoothed=eigenvalue, min_eig_gaussian=np.linalg.eigvalsh(matched)[0],
                    ))

    transport = pd.DataFrame(transport_rows)
    smoothing = pd.DataFrame(smoothing_rows)
    transport.to_csv(OUT / "transport_seed_rows.csv", index=False)
    smoothing.to_csv(OUT / "smoothing_seed_rows.csv", index=False)
    summarize(transport, ["source", "epsilon", "alpha", "method"],
              ["cost", "budget_fraction", "min_eig", "min_eig_relative", "q", "q_relative"]
              ).to_csv(OUT / "transport_summary.csv", index=False)
    summarize(smoothing, ["source", "epsilon", "lambda_noise"],
              ["tau2", "cost_bound", "budget_fraction", "q_smoothed", "q_gaussian", "q_ratio",
               "min_eig_smoothed", "min_eig_gaussian"]).to_csv(OUT / "smoothing_summary.csv", index=False)


def loss_curve():
    epsilon = .05
    displacements = [sign * step for sign in (-1, 1)
                     for step in (.1, .03, .01, .003, .001, .0003, .0001)]
    curve_rows = []
    coefficient_rows = []
    curve_x = np.array([0, 1, 1])
    curve_z = np.array([2, 3, 2])
    linear = np.array([1, 1, 0])
    for source, delta in (("singular", 0), ("spd_e3", epsilon**3)):
        for seed in range(2701, 2713):
            n = 100_000
            matrices = draw_matrices(np.random.default_rng(seed), n, epsilon, delta)
            projected, _ = project_rank_one(matrices)
            y_proj = encode(projected)
            laws = {"source": encode(matrices), "svd_target": y_proj, "matched_gaussian": y_proj}
            for method, features in laws.items():
                products = features[:, curve_x] * features[:, curve_z]
                if method == "matched_gaussian":
                    gram = gaussian_gram(features.T @ features / n, curve_x, curve_z)
                    c2 = (linear @ gram @ linear) / 2
                    c3 = linear @ gram[:, 2]
                    c4 = gram[2, 2] / 2
                    assert c2 > 0
                else:
                    a = products[:, 0] + products[:, 1]
                    b = products[:, 2]
                    c2, c3, c4 = np.mean(a*a) / 2, np.mean(a*b), np.mean(b*b) / 2
                    if method == "svd_target":
                        assert c2 < 1e-25
                setting = dict(source=source, method=method, seed=seed, n=n)
                coefficient_rows.append(dict(**setting, c2=c2, c3=c3, c4=c4))
                for step in displacements:
                    risk = c2 * step**2 + c3 * step**3 + c4 * step**4
                    assert risk > 0
                    if method != "matched_gaussian":
                        assert abs(risk - np.mean((step*a + step**2*b)**2) / 2) < 1e-15
                    curve_rows.append(dict(**setting, t=step, risk=risk))
    curve = pd.DataFrame(curve_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    curve.to_csv(OUT / "curve_seed_rows.csv", index=False)
    coefficients.to_csv(OUT / "curve_coefficient_rows.csv", index=False)
    summarize(curve, ["source", "method", "t"], ["risk"]).to_csv(OUT / "curve_summary.csv", index=False)
    summarize(coefficients, ["source", "method"], ["c2", "c3", "c4"]
              ).to_csv(OUT / "curve_coefficient_summary.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    smoothing_and_interpolation()
    loss_curve()
