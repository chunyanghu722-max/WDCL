# WDCL experiments

The experiments use synthetic data and fixed seeds. Run from this directory with Python 3.13.

```sh
python -m pip install -r requirements.txt
python transport_and_mixtures.py
python curvature.py
python local_regression.py
```

| Script | Main outputs in `results/` |
| --- | --- |
| `transport_and_mixtures.py` | `projection_sweep.csv` for regularization and conditioning, and `mixture_scan.csv` for finite-component budgets |
| `curvature.py` | `smoothing_summary.csv`, `transport_summary.csv`, and `curve_summary.csv` for noise, interpolation, and rank-one loss growth |
| `local_regression.py` | `regression_summary.csv`, `regression_ratios.csv`, and `population_curvature.csv` for estimation, prediction, and tangent curvature |

Transport costs are squared. `projection_sweep.csv` reports the projection cost for the fixed determinant cone. `mixture_scan.csv` reports the costs of feasible couplings under each component budget.
