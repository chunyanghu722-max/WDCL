"""Fit a local rank-one regression under three feature laws."""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from curvature import (OUT, draw_matrices, encode, gaussian_gram, project_rank_one,
                       source_covariance, summarize)


EPSILON = .25
DELTA = EPSILON**3
SAMPLE_SIZES = (128, 512, 2048, 8192)
SIGMAS = (.003, .01, .03)
SEEDS = range(4801, 4833)
STARTS = [(a, a, 1.) for a in (0, .05, -.05, .3, -.3)]
TRUTH = np.array([0, 1, 0, 0])


def bilinear_features(y):
    return (y[:, :2, None] * y[:, None, 2:]).reshape(len(y), 4)


def parameter_vector(p):
    u, v, w = p
    return np.array([v, w, u*v, u*w])


def parameter_jacobian(p):
    u, v, w = p
    return np.array([[0, 1, 0], [0, 0, 1], [v, u, 0], [w, 0, u]])


def feature_laws(rng, n, matched_factor):
    matrices = draw_matrices(rng, n, EPSILON, DELTA)
    projected, _ = project_rank_one(matrices)
    return {
        "source": bilinear_features(encode(matrices)),
        "svd_target": bilinear_features(encode(projected)),
        "matched_gaussian": bilinear_features(rng.normal(size=(n, 4)) @ matched_factor.T),
    }


def fit_local(features, response):
    n = len(response)
    gram = features.T @ features / n
    cross = features.T @ response / n
    eigenvalues, basis = np.linalg.eigh(gram)
    # Drop the numerical nullspace of the projected-law Gram matrix.
    keep = eigenvalues > 1e-12 * eigenvalues.max()
    scale = np.sqrt(eigenvalues[keep])
    design = scale[:, None] * basis[:, keep].T
    target = basis[:, keep].T @ cross / scale

    def residual(params):
        return design @ parameter_vector(params) - target

    def jacobian(params):
        return design @ parameter_jacobian(params)

    fits = []
    for guess in STARTS:
        fit = least_squares(residual, guess, jac=jacobian, bounds=([-1, -1, .5], [1, 1, 1.5]),
                            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000)
        fits.append((np.sum(residual(fit.x)**2), fit))
    return fits


def regression(matched_factor):
    rows, checks = [], []
    for sigma in SIGMAS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            trains = feature_laws(rng, max(SAMPLE_SIZES), matched_factor)
            noise = sigma * rng.normal(size=max(SAMPLE_SIZES))
            tests = feature_laws(rng, 10_000, matched_factor)
            for method, train in trains.items():
                test = tests[method]
                test_gram = test.T @ test / len(test)
                for n in SAMPLE_SIZES:
                    design = train[:n]
                    response = design[:, 1] + noise[:n]
                    fits = fit_local(design, response)
                    setting = dict(sigma=sigma, seed=seed, n=n, method=method)
                    for start, (objective, fit) in enumerate(fits):
                        checks.append(dict(**setting, start=start, success=fit.success,
                                           nfev=fit.nfev, objective=objective))

                    _, best_fit = min(fits, key=lambda result: result[0])
                    u, v, w = best_fit.x
                    error = parameter_vector(best_fit.x) - TRUTH
                    objective_range = np.ptp([objective for objective, _ in fits])
                    boundary = int(abs(u) > .999 or abs(v) > .999 or w < .501 or w > 1.499)
                    rows.append(dict(**setting, param_error=np.linalg.norm(error), param_sq=error @ error,
                                     test_mse=error @ test_gram @ error, boundary=boundary,
                                     start_obj_range=objective_range, success=int(best_fit.success),
                                     fit_u=u, fit_v=v, fit_w=w))

    fits = pd.DataFrame(rows)
    fits.to_csv(OUT / "regression_seed_rows.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "optimization_checks.csv", index=False)
    summarize(fits, ["sigma", "method", "n"],
              ["param_error", "param_sq", "test_mse", "boundary", "start_obj_range"]
              ).to_csv(OUT / "regression_summary.csv", index=False)
    return fits


def bootstrapped_ratios(fits):
    rng = np.random.default_rng(4820)
    rows = []
    for (sigma, n), subset in fits.groupby(["sigma", "n"]):
        for metric in ("param_error", "test_mse"):
            wide = subset.pivot(index="seed", columns="method", values=metric)
            assert len(wide) == len(SEEDS) and not wide.isna().any().any()
            projected = wide.svd_target.to_numpy()
            matched = wide.matched_gaussian.to_numpy()
            indices = rng.integers(0, len(wide), size=(10_000, len(wide)))
            ratios = projected[indices].mean(axis=1) / matched[indices].mean(axis=1)
            lo, hi = np.quantile(ratios, (.025, .975))
            rows.append(dict(sigma=sigma, n=n, metric=metric, ratio=projected.mean() / matched.mean(),
                             lower=lo, upper=hi, projected_mean=projected.mean(), matched_mean=matched.mean()))
    pd.DataFrame(rows).to_csv(OUT / "regression_ratios.csv", index=False)


def population_curvature(matched_covariance):
    tangent = [0, 1, 3]
    weak = np.array([1., 0., 0., 1.])
    x, z = np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])
    projected_grams = []
    for seed in range(6570, 6582):
        matrices = draw_matrices(np.random.default_rng(seed), 80_000, EPSILON, DELTA)
        projected, _ = project_rank_one(matrices)
        features = bilinear_features(encode(projected))
        projected_grams.append(features.T @ features / len(features))
    grams = {
        "source": gaussian_gram(source_covariance(EPSILON, DELTA), x, z),
        "matched_gaussian": gaussian_gram(matched_covariance, x, z),
        "svd_target": np.mean(projected_grams, axis=0),
    }
    rows = []
    for law, gram in grams.items():
        minimum = np.linalg.eigvalsh(gram[np.ix_(tangent, tangent)])[0]
        quadratic = weak @ gram @ weak / 2
        cubic = weak @ gram[:, 2]
        if law == "svd_target":
            assert max(abs(minimum), abs(quadratic), abs(cubic)) < 1e-14
            minimum = quadratic = cubic = 0.
        rows.append(dict(law=law, min_tangent_hessian=minimum, normalized_weak_curvature=quadratic,
                         loss_quadratic=quadratic, loss_cubic=cubic, loss_quartic=gram[2, 2] / 2))
    pd.DataFrame(rows).to_csv(OUT / "population_curvature.csv", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    calibration = draw_matrices(np.random.default_rng(4780), 200_000, EPSILON, DELTA)
    projected, _ = project_rank_one(calibration)
    features = encode(projected)
    covariance = features.T @ features / len(features)
    np.savetxt(OUT / "calibration_covariance.csv", covariance, delimiter=",")
    fits = regression(np.linalg.cholesky(covariance))
    bootstrapped_ratios(fits)
    population_curvature(covariance)


if __name__ == "__main__":
    main()
