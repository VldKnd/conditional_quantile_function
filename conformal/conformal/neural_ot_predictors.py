# regression_refactored/conformal/neural_ot_predictors.py

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os as _os
import sys as _sys

from sklearn.model_selection import train_test_split
from .base import ConformalRegressor

# --- Neural OT Dependencies and Helpers ---
import torch as _torch
from torch.utils.data import DataLoader as _DataLoader, TensorDataset as _TensorDataset

# Updated import to use the amortized solver (correct class name)
from pushforward_operators.quantile_regression.unconstrained_amortized_optimal_transport_quantile_regression import (
    UnconstrainedAmortizedOTQuantileRegression as _AmortizedUOTQR,
)


def _to_torch(x, device=None, dtype=None):
    if isinstance(x, _torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return _torch.from_numpy(x).to(device=device, dtype=dtype)


def _estimate_U(model, X_t: _torch.Tensor, Y_t: _torch.Tensor) -> _torch.Tensor:
    # Use the model's push_y_given_x to estimate U given (X, Y)
    if hasattr(model, "push_y_given_x") and callable(getattr(model, "push_y_given_x")):
        return model.push_y_given_x(x=X_t, y=Y_t)
    raise RuntimeError(
        "Neural OT model does not expose 'push_y_given_x' to estimate U from (X,Y)."
    )


def train_neural_ot_regression(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    number_of_hidden_layers: int = 2,
    z_dimension: int = 32,  # map to hidden_dimension
    batch_size: int = 256,
    num_epochs: int = 500,
    lr: float = 1e-3,
    device: str | None = None,
    dtype: str = "float32",
):
    """Trains a Neural OT regression model using the amortized solver."""

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(_torch, dtype)

    X_t = _to_torch(X_train, device=device, dtype=torch_dtype)
    Y_t = _to_torch(Y_train, device=device, dtype=torch_dtype)

    ds = _TensorDataset(X_t, Y_t)
    dl = _DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    x_dim = X_t.shape[1]
    y_dim = Y_t.shape[1]

    print(
        f"[NeuralOT] Initializing amortized UOTQR: device={device}, dtype={dtype}, "
        f"x_dim={x_dim}, y_dim={y_dim}, epochs={num_epochs}, batch={batch_size}, hidden_dim={z_dimension}, layers={number_of_hidden_layers}"
    )

    # Use the amortized solver with appropriate constructor parameters
    model = _AmortizedUOTQR(
        feature_dimension=x_dim,
        response_dimension=y_dim,
        hidden_dimension=z_dimension,
        number_of_hidden_layers=number_of_hidden_layers,
        activation_function_name="Softplus",
        network_type="PISCNN",#"SCFFNN",
        potential_to_estimate_with_neural_network="y",
    )

    model = model.to(device=_torch.device(device), dtype=torch_dtype)
    model.train()

    # Build a TrainParameters-like object (duck-typed)
    train_parameters = type("TP", (), {})()
    setattr(train_parameters, "number_of_epochs_to_train", num_epochs)
    setattr(train_parameters, "optimizer_parameters", {"lr": lr})
    setattr(train_parameters, "scheduler_parameters", {})
    setattr(train_parameters, "verbose", True)

    print("[NeuralOT] Starting training ...")
    model.fit(dl, train_parameters=train_parameters)
    model.eval()
    print("[NeuralOT] Training complete.")
    return model


def calibrate_neural_ot_cp(
    model,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    *,
    alpha: float = 0.9,
    device: str | None = None,
    dtype: str = "float32",
):
    """Calibrates the conformal threshold using latent U norms with robust U-estimation."""

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(_torch, dtype)

    X_t = _to_torch(X_cal, device=device, dtype=torch_dtype)
    Y_t = _to_torch(Y_cal, device=device, dtype=torch_dtype)

    print(
        f"[NeuralOT] Calibrating threshold on {len(X_cal)} samples (alpha={alpha}) ..."
    )
    U_hat = _estimate_U(model, X_t=X_t, Y_t=Y_t)
    U_norm = _torch.linalg.norm(U_hat, dim=-1).detach().cpu().numpy()

    n = len(U_norm)
    q = np.min([np.ceil((n + 1) * alpha) / n, 1.0])
    threshold = np.quantile(U_norm, q)
    print(
        f"[NeuralOT] Calibration complete. Quantile level={q:.4f}, threshold={threshold:.6f}."
    )
    return float(threshold)


def neural_ot_inclusion_mask(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    threshold: float,
    *,
    device: str | None = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Computes the inclusion mask for new data points with robust U-estimation."""

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(_torch, dtype)

    X_t = _to_torch(X, device=device, dtype=torch_dtype)
    Y_t = _to_torch(Y, device=device, dtype=torch_dtype)

    U_hat = _estimate_U(model, X_t=X_t, Y_t=Y_t)
    mask = (_torch.linalg.norm(U_hat, dim=-1) <= threshold).detach().cpu().numpy()
    return mask




@dataclass
class NeuralOTCPScore(ConformalRegressor):
    """
    Neural OT-CP that learns the multivariate quantiles of a score function (e.g., residuals)
    using an amortized solver.


    """

    alpha: float = 0.9
    nn_kwargs: Dict = None

    model_: Optional[object] = None
    threshold_: Optional[float] = None

    # Volume settings and residual box bounds (learned during fit)
    volume_samples: int = int(1e6)
    S_box_min_: Optional[np.ndarray] = None
    S_box_max_: Optional[np.ndarray] = None

    def __init__(self, alpha: float = 0.9, nn_kwargs: Optional[Dict] = None):
        """     nn_kwargs keys (with defaults if None provided):
      -  number_of_hidden_layers: int = 2
      - z_dimension: int = 32
      - batch_size: int = 256
      - num_epochs: int = 500
      - lr: float = 1e-3
      - device: Optional[str] = None  # auto-selects CUDA if available
      - dtype: str = "float32"
      """
        self.alpha = alpha
    
        # Set defaults, then update with any user-provided overrides
        self.nn_kwargs = {
            "number_of_hidden_layers": 2,
            "z_dimension": 32,
            "batch_size": 256,
            "num_epochs": 500,
            "lr": 1e-3,
            "device": None,
            "dtype": "float32",
        }
        if nn_kwargs is not None:
            self.nn_kwargs.update(nn_kwargs)
        # Initialize learned attributes
        self.model_ = None
        self.threshold_ = None
        self.S_box_min_ = None
        self.S_box_max_ = None
        self.volume_samples = int(1e4)

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None,
    ):
        if y_pred_cal is None:
            raise ValueError(
                "NeuralOTCPScore requires y_pred_cal for training on residuals."
            )

        X_train, X_calib, y_train, y_calib, y_pred_train, y_pred_calib = (
            train_test_split(X_cal, y_cal, y_pred_cal, test_size=0.5, random_state=42)
        )
        S_train = y_train - y_pred_train
        S_calib = y_calib - y_pred_calib

        print(
            f"[NeuralOT] Fitting on residuals: train={len(X_train)}, calib={len(X_calib)}, "
            f"y_dim={S_train.shape[1]}"
        )
        train_kwargs = self.nn_kwargs if self.nn_kwargs is not None else {}
        self.model_ = train_neural_ot_regression(X_train, S_train, **train_kwargs)

        self.threshold_ = calibrate_neural_ot_cp(
            self.model_, X_calib, S_calib, alpha=self.alpha
        )

        # Store residual box bounds over the combined residuals for Monte Carlo volume estimation
        S_all = np.vstack([S_train, S_calib])
        self.S_box_min_ = np.min(S_all, axis=0)
        self.S_box_max_ = np.max(S_all, axis=0)
        return self

    def contains(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if self.model_ is None or self.threshold_ is None:
            raise RuntimeError("The model must be fit before calling contains.")
        if y_pred_test is None:
            raise ValueError(
                "NeuralOTCPScore.contains requires y_pred_test to form residuals."
            )
        S_test = y_test - y_pred_test
        return neural_ot_inclusion_mask(self.model_, X_test, S_test, self.threshold_)

    def _estimate_volume_for_x(self, x_tick: np.ndarray, N: Optional[int] = None) -> float:
        """Monte Carlo estimate of volume of {s: ||U_hat(x_tick, s)|| <= threshold}."""
        if self.S_box_min_ is None or self.S_box_max_ is None:
            raise RuntimeError("Residual box bounds are not initialized. Fit the model first.")
        N = int(self.volume_samples if N is None else N)
        m = self.S_box_min_
        M = self.S_box_max_
        v = m + np.random.random((N, m.shape[0])) * (M - m)
        scale = float(np.prod(M - m))

        # Vectorized evaluation of U_hat(x, v) for all samples v
        device = self.nn_kwargs.get("device") or ("cuda" if _torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(_torch, self.nn_kwargs.get("dtype", "float32"))
        X_rep = np.repeat(x_tick.reshape(1, -1), N, axis=0)
        X_t = _to_torch(X_rep, device=device, dtype=torch_dtype)
        V_t = _to_torch(v, device=device, dtype=torch_dtype)
        with _torch.no_grad():
            U_hat = _estimate_U(self.model_, X_t=X_t, Y_t=V_t)
            norms = _torch.linalg.norm(U_hat, dim=-1).detach().cpu().numpy()
        prob = float(np.mean(norms <= self.threshold_))
        return float(prob * scale)

    def metrics(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray
    ) -> Dict[str, Any]:
        mask = self.contains(X_test, y_test, y_pred_test)
        avg_coverage = float(np.mean(mask))

        # Estimate per-point volumes via Monte Carlo using stored residual box
        volumes = [self._estimate_volume_for_x(X_test[i]) for i in range(len(X_test))]
        avg_volume = float(np.mean(volumes)) if len(volumes) > 0 else float("nan")

        return {
            "avg_coverage": avg_coverage,
            "avg_volume": avg_volume,
            "coverage": mask,
            "volume": np.array(volumes, dtype=float),
        }
