# OT-CP Regression Experiments

This folder contains a self-contained suite to reproduce and extend multivariate conformal regression experiments using Monge–Kantorovich (MK) ranks and optimal transport quantile maps, with baselines based on hyperrectangles and ellipsoids. A unified entry script orchestrates both synthetic and real-data experiments.

### Directory layout
- `experiments.py`: Unified CLI with `synthetic` and `real` subcommands.
- `utils.py`: Shared utilities for plotting, IO, synthetic shapes, and real-data coverage/evaluation.
- `functions.py`: Core MK ranks/quantiles utilities and (optional) Neural OT CP helpers.
- `ellipsoidal_conformal_utilities.py`: helper routines for ellipsoidal baselines.
- `conformal/predictors.py`: object-oriented conformal predictors with a shared interface.
- `data/`: dataset files (`.arff` and `.csv`) used in real-data experiments.

### How to run
Run from the project root or from within this directory.

Synthetic experiments:
- Shapes comparison (MK vs RECT vs ELL):
```bash
python regression_refactored/experiments.py synthetic --task shapes --alpha 0.9 --seed 62
```
- Coverage and efficiency (repeated):
```bash
python regression_refactored/experiments.py synthetic --task coverage_efficiency --alpha 0.9 --reps 50 --seed 62
```
- 3D homoscedastic visualization:
```bash
python regression_refactored/experiments.py synthetic --task 3d_homo --alpha 0.9
```
- 3D heteroscedastic visualization:
```bash
python regression_refactored/experiments.py synthetic --task 3d_hetero --alpha 0.9
```
- Conditional coverage (OT-CP+):
```bash
python regression_refactored/experiments.py synthetic --task conditional_coverage --alpha 0.9 --k-neighbors 100 --n-test 300
```
- Timing curves:
```bash
python regression_refactored/experiments.py synthetic --task timing --alpha 0.9
```

Real-data experiments:
- Medium ARFF setting:
```bash
python regression_refactored/experiments.py real --setting medium_arff --alpha 0.9 --nrep 5
```
- Medium CSV setting:
```bash
python regression_refactored/experiments.py real --setting medium_csv --alpha 0.9 --nrep 5
```
- Large setting:
```bash
python regression_refactored/experiments.py real --setting large --alpha 0.9 --nrep 5
```

Outputs are written under `regression_refactored/outputs/` (PDFs) and `regression_refactored/csv/` (metrics) by default. Use `--out-dir` and `--csv-dir` to override.

### Mathematical details

See the sections above in this README for the quadratic OT potentials, MK rank/quantile region, OT-CP+, and baseline definitions (hyperrectangles, ellipsoids).

### Object-oriented conformal predictors

We expose a simple, consistent API to add and compare conformal regression methods. All predictors implement:

- `fit(X_cal, y_cal, y_pred_cal=None)`: Calibrate from calibration covariates and scores/residuals.
- `contains(X_test, y_test, y_pred_test=None) -> Optional[mask]`: Optional per-sample inclusion mask.
- `metrics(X_test, y_test, y_pred_test) -> Dict[str, float]`: At minimum returns coverage, may include size/volume metrics.

Implemented predictors (`conformal/predictors.py`):

- `OTCPGlobal(alpha=0.9)`
  - Global MK rank threshold calibrated on scores (independent of X).
  - Uses `functions.MultivQuantileTreshold` and `functions.RankFunc`.
  - Metrics: marginal coverage and global average volume via `functions.get_volume_QR`.

- `OTCPAdaptiveKNN(alpha=0.9, n_neighbors=100)`
  - Adaptive OT-CP+ using KNN neighborhoods in X.
  - Uses `functions.MultivQuantileTreshold_Adaptive` and `functions.ConditionalRank_Adaptive`.
  - Metrics: marginal coverage and average volume via `utils.set_coverage`.

- `EllipsoidalLocal(alpha=0.9, n_neighbors=100, lam=0.95)`
  - Baseline using local ellipsoidal thresholds with KNN mixing.
  - Uses `utils.get_params_local_ellipsoids` and `utils.set_coverage_ell`.
  - Metrics: marginal coverage (validity) and efficiency.

Example usage in Python:
```python
import numpy as np
from regression_refactored.conformal.predictors import OTCPGlobal, OTCPAdaptiveKNN, EllipsoidalLocal

# Calibration
otcp = OTCPGlobal(alpha=0.9).fit(X_cal, y_cal, y_pred_cal)
otcp_knn = OTCPAdaptiveKNN(alpha=0.9, n_neighbors=100).fit(X_cal, y_cal, y_pred_cal)
ell = EllipsoidalLocal(alpha=0.9, n_neighbors=100, lam=0.95).fit(X_cal, y_cal, y_pred_cal)

# Evaluation
res_otcp = otcp.metrics(X_test, y_test, y_pred_test)
res_knn = otcp_knn.metrics(X_test, y_test, y_pred_test)
res_ell = ell.metrics(X_test, y_test, y_pred_test)
print(res_otcp, res_knn, res_ell)
```

### Function breakdown

- `experiments.py`: CLI orchestration for synthetic and real experiments.
- `utils.py`: IO/plotting, synthetic shapes, and real-data coverage helpers.
- `functions.py`: MK ranks/quantiles core, including adaptive OT-CP+.
- `conformal/predictors.py`: OO predictors wrapping the above with a consistent API.

### Dependencies
- numpy, scipy, scikit-learn, matplotlib, seaborn, pandas
- POT (`ot`), and for some real-data utilities: `scipy.io.arff`

Optional (Neural OT CP): PyTorch.

### Reproducibility and outputs
- Seeds are set per run; results (PDFs/CSVs) saved under `outputs/` and `csv/` unless overridden. 