import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Tuple

# Import Neural OT CP (same import style as other scripts here)
from conformal.neural_ot_predictors import NeuralOTCPScore
from torch.utils.data import DataLoader as _DataLoader, TensorDataset as _TensorDataset
from infrastructure.classes import TrainParameters as _TrainParameters
from pushforward_operators.quantile_regression.unconstrained_amortized_optimal_transport_quantile_regression import (
    UnconstrainedAmortizedOTQuantileRegression as _AmortizedUOTQR,
)
from conformal.neural_ot_predictors import calibrate_neural_ot_cp as _calibrate_neural_ot_cp


# -----------------------------
# Config (Jupyter-friendly)
# -----------------------------
CONFIG: Dict = {
    "dataset_name": "enb",  # 2D output dataset
    "alpha": 0.9,
    "seed": 567,
    "test_size": 0.5,  # split for test; then we'll split test into cal/test equally
    "neural_ot": {
        "number_of_hidden_layers": 2,
        "z_dimension": 32,
        "batch_size": 256,
        "num_epochs": 200,
        "lr": 1e-3,
        "device": None,     # auto-selects CUDA if available when None
        "dtype": "float32",
    },
    "viz": {
        "figsize": (10, 4),
        "bins": 60,
        "alpha_points": 0.35,
        "alpha_hist": 0.35,
        "save_dir": "outputs/ot_neural_inspect",
        "prefix": "enb_residual_uniformity",
        "save": False,
        "show": True,
    },
}

CONFIG["incremental"] = {
    "enabled": True,
    "total_epochs": 200,
    "step_epochs": 50,
    "batch_size": CONFIG["neural_ot"]["batch_size"],
    "lr": CONFIG["neural_ot"]["lr"],
    "device": CONFIG["neural_ot"]["device"],
    "dtype": CONFIG["neural_ot"]["dtype"],
    "save_dir": CONFIG["viz"]["save_dir"],
    "prefix": "enb_uniformity_snap",
}


# -----------------------------
# Data utilities
# -----------------------------

def load_real_dataframe(name: str) -> pd.DataFrame:
    """Load a real dataset by name.

    Tries, in order:
    - regression_refactored/data/{name}.arff
    - OTCP/data/{name}.arff
    - regression_refactored/csv/{name}.csv
    """
    repo_root = "/home/labcmap/mahmoud.hegazy/conditional_quantile_function"

    candidates = [
        os.path.join(repo_root, "regression_refactored", "data", f"{name}.arff"),
        os.path.join(repo_root, "OTCP", "data", f"{name}.arff"),
    ]

    # Try ARFF first
    for arff_path in candidates:
        if os.path.exists(arff_path):
            from scipy.io import arff  # type: ignore
            data, _ = arff.loadarff(arff_path)
            df = pd.DataFrame(data)
            return df.dropna()

    # Fallback to CSV
    csv_path = os.path.join(repo_root, "regression_refactored", "csv", f"{name}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).dropna()

    raise FileNotFoundError(f"Could not find dataset {name} in known locations.")


def get_xy_from_df(df: pd.DataFrame, y_dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataframe into features X and responses Y, taking the last y_dim columns as Y."""
    y = df[df.columns[-y_dim:]].values
    x = df[df.columns[:-y_dim]].values
    return x.astype(np.float64), y.astype(np.float64)


# -----------------------------
# Math helpers
# -----------------------------

def chi2_radius_2d(alphas: np.ndarray) -> np.ndarray:
    """For 2D standard normal U, radius r such that P(||U|| <= r) = alpha.
    Since R^2 ~ ChiSquare(df=2), CDF(R) = 1 - exp(-r^2/2) => r = sqrt(-2 log(1 - alpha)).
    """
    alphas = np.clip(alphas, 1e-6, 1 - 1e-6)
    return np.sqrt(-2.0 * np.log(1.0 - alphas))


def make_circle_points(radius: float, num_angles: int = 256) -> np.ndarray:
    thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    return np.stack([radius * np.cos(thetas), radius * np.sin(thetas)], axis=-1)


def standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def rayleigh_pdf(r: np.ndarray) -> np.ndarray:
    # Rayleigh(sigma=1) is the norm distribution of 2D Normal(0, I)
    return r * np.exp(-0.5 * r**2)


# -----------------------------
# Training utilities
# -----------------------------

def train_models(x: np.ndarray, y: np.ndarray, config: Dict):
    seed = config.get("seed", 567)
    test_size = config.get("test_size", 0.5)
    alpha = config.get("alpha", 0.9)

    # Split: train vs test, then test split into cal/test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    X_test, X_cal, y_test, y_cal = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

    # Base regressor for residuals
    base = RandomForestRegressor(random_state=seed)
    base.fit(X_train, y_train)

    y_pred_train = base.predict(X_train)
    y_pred_cal = base.predict(X_cal)
    y_pred_test = base.predict(X_test)

    # Neural OT model on residual scores
    neural_cfg = dict(config.get("neural_ot", {}))
    neural = NeuralOTCPScore(alpha=alpha, nn_kwargs=neural_cfg)
    neural.fit(X_cal, y_cal, y_pred_cal)

    return {
        "base": base,
        "neural": neural,
        "splits": {
            "X_train": X_train,
            "y_train": y_train,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred_train": y_pred_train,
            "y_pred_cal": y_pred_cal,
            "y_pred_test": y_pred_test,
        },
    }


def compute_residuals(models: Dict) -> Dict[str, np.ndarray]:
    X_train = models["splits"]["X_train"]
    y_train = models["splits"]["y_train"]
    y_pred_train = models["splits"]["y_pred_train"]

    X_test = models["splits"]["X_test"]
    y_test = models["splits"]["y_test"]
    y_pred_test = models["splits"]["y_pred_test"]

    S_train = y_train - y_pred_train
    S_test = y_test - y_pred_test

    return {"S_train": S_train, "S_test": S_test, "X_train": X_train, "X_test": X_test}


def push_residuals_to_U(models: Dict, S: np.ndarray, X: np.ndarray) -> np.ndarray:
    import torch
    neural = models["neural"].model_

    # Device/dtype from model
    p = next(neural.parameters())
    device = p.device
    dtype = p.dtype

    X_t = torch.from_numpy(X).to(device=device, dtype=dtype)
    S_t = torch.from_numpy(S).to(device=device, dtype=dtype)
    with torch.no_grad():
        U_hat_t = neural.push_y_given_x(y=S_t, x=X_t)
    return U_hat_t.detach().cpu().numpy()


# -----------------------------
# Plotting utilities
# -----------------------------

def plot_scatter_before_after(S_train: np.ndarray, S_test: np.ndarray, U_train: np.ndarray, U_test: np.ndarray, viz: Dict):
    figsize = tuple(viz.get("figsize", (10, 4)))
    a_pts = float(viz.get("alpha_points", 0.35))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].set_title("Residuals S (train/test)")
    axes[0].scatter(S_train[:, 0], S_train[:, 1], s=6, alpha=a_pts, label="train")
    axes[0].scatter(S_test[:, 0], S_test[:, 1], s=6, alpha=a_pts, label="test")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(markerscale=3)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Pushed residuals U_hat = T_x(S) (train/test)")
    axes[1].scatter(U_train[:, 0], U_train[:, 1], s=6, alpha=a_pts, label="train")
    axes[1].scatter(U_test[:, 0], U_test[:, 1], s=6, alpha=a_pts, label="test")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(markerscale=3)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_marginals(U_train: np.ndarray, U_test: np.ndarray, viz: Dict):
    # Overlay standard normal pdf on histograms of each coordinate
    figsize = tuple(viz.get("figsize", (10, 4)))
    bins = int(viz.get("bins", 60))
    a_hist = float(viz.get("alpha_hist", 0.35))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for d in range(2):
        ax = axes[d]
        ax.set_title(f"U coord {d} ~ N(0,1)?")
        data_train = U_train[:, d]
        data_test = U_test[:, d]

        rng = np.linspace(min(data_train.min(), data_test.min()), max(data_train.max(), data_test.max()), 400)
        pdf = standard_normal_pdf(rng)

        ax.hist(data_train, bins=bins, density=True, alpha=a_hist, label="train")
        ax.hist(data_test, bins=bins, density=True, alpha=a_hist, label="test")
        ax.plot(rng, pdf, "k--", label="N(0,1) PDF")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_radius(U_train: np.ndarray, U_test: np.ndarray, viz: Dict):
    # Norm radius distribution vs Rayleigh(sigma=1)
    figsize = tuple(viz.get("figsize", (10, 4)))
    bins = int(viz.get("bins", 60))
    a_hist = float(viz.get("alpha_hist", 0.35))

    r_train = np.linalg.norm(U_train, axis=1)
    r_test = np.linalg.norm(U_test, axis=1)

    r_max = max(r_train.max(), r_test.max())
    rng = np.linspace(0.0, r_max, 400) 
    pdf = rayleigh_pdf(rng)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title("||U|| distribution vs Rayleigh(1)")
    ax.hist(r_train, bins=bins, density=True, alpha=a_hist, label="train")
    ax.hist(r_test, bins=bins, density=True, alpha=a_hist, label="test")
    ax.plot(rng, pdf, "k--", label="Rayleigh PDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# %%
# Reproducibility
np.random.seed(CONFIG.get("seed", 567))

# %%
# Load data and split X, Y (last 2 columns as Y)
df = load_real_dataframe(CONFIG.get("dataset_name", "enb"))
x, y = get_xy_from_df(df, y_dim=2)
print("Shapes:", x.shape, y.shape)

# %%
# Train RF + NeuralOT on residuals
models = train_models(x, y, CONFIG)

# %%
# Compute residuals for train and test, then push through OT to latent U
parts = compute_residuals(models)
S_train, S_test = parts["S_train"], parts["S_test"]
U_train = push_residuals_to_U(models, S_train, parts["X_train"])
U_test = push_residuals_to_U(models, S_test, parts["X_test"])

# %%
# Scatter before/after
fig1 = plot_scatter_before_after(S_train, S_test, U_train, U_test, CONFIG.get("viz", {}))

# %%
# Marginal histograms for pushed residuals vs N(0,1)
fig2 = plot_marginals(U_train, U_test, CONFIG.get("viz", {}))

# %%
# Radial histogram vs Rayleigh
fig3 = plot_radius(U_train, U_test, CONFIG.get("viz", {}))

# %%
# Optional save/show control
viz = CONFIG.get("viz", {})
if viz.get("save", False):
    os.makedirs(viz.get("save_dir", "outputs/ot_neural_inspect"), exist_ok=True)
    prefix = viz.get("prefix", "enb_residual_uniformity")
    fig1.savefig(os.path.join(viz.get("save_dir", "outputs/ot_neural_inspect"), f"{prefix}_scatter.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(viz.get("save_dir", "outputs/ot_neural_inspect"), f"{prefix}_marginals.pdf"), bbox_inches="tight")
    fig3.savefig(os.path.join(viz.get("save_dir", "outputs/ot_neural_inspect"), f"{prefix}_radius.pdf"), bbox_inches="tight")
if viz.get("show", True):
    plt.show()
else:
    plt.close(fig1); plt.close(fig2); plt.close(fig3) 


def _prepare_splits_and_base(x: np.ndarray, y: np.ndarray, config: Dict):
    seed = config.get("seed", 567)
    test_size = config.get("test_size", 0.5)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    X_test, X_cal, y_test, y_cal = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

    base = RandomForestRegressor(random_state=seed)
    base.fit(X_train, y_train)

    y_pred_train = base.predict(X_train)
    y_pred_cal = base.predict(X_cal)
    y_pred_test = base.predict(X_test)

    # Residuals for full train/test (for visualization)
    S_train_full = y_train - y_pred_train
    S_test_full = y_test - y_pred_test

    # Split calibration into train/calib for neural OT
    X_tr_in, X_ca_in, y_tr_in, y_ca_in, y_pred_tr_in, y_pred_ca_in = train_test_split(
        X_cal, y_cal, y_pred_cal, test_size=0.5, random_state=42
    )
    S_tr_in = y_tr_in - y_pred_tr_in
    S_ca_in = y_ca_in - y_pred_ca_in

    return {
        "base": base,
        "X_train": X_train,
        "y_train": y_train,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_cal": y_pred_cal,
        "y_pred_test": y_pred_test,
        "S_train_full": S_train_full,
        "S_test_full": S_test_full,
        "X_tr_in": X_tr_in,
        "S_tr_in": S_tr_in,
        "X_ca_in": X_ca_in,
        "S_ca_in": S_ca_in,
    }


def _init_amortized_model(x_dim: int, y_dim: int, cfg: Dict):
    import torch
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.get("dtype", "float32"))

    model = _AmortizedUOTQR(
        feature_dimension=x_dim,
        response_dimension=y_dim,
        hidden_dimension=cfg.get("z_dimension", 32),
        number_of_hidden_layers=cfg.get("number_of_hidden_layers", 2),
        activation_function_name="Softplus",
        network_type="PISCNN",
        potential_to_estimate_with_neural_network="y",
    )
    model = model.to(device=torch.device(device), dtype=torch_dtype)
    return model


def _make_loader(X: np.ndarray, S: np.ndarray, batch_size: int, cfg: Dict):
    import torch
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, cfg.get("dtype", "float32"))
    X_t = torch.from_numpy(X).to(device=device, dtype=torch_dtype)
    S_t = torch.from_numpy(S).to(device=device, dtype=torch_dtype)
    ds = _TensorDataset(X_t, S_t)
    return _DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)


def _push_to_U(model, X: np.ndarray, S: np.ndarray) -> np.ndarray:
    import torch
    p = next(model.parameters())
    device = p.device
    dtype = p.dtype
    X_t = torch.from_numpy(X).to(device=device, dtype=dtype)
    S_t = torch.from_numpy(S).to(device=device, dtype=dtype)
    with torch.no_grad():
        U_t = model.push_y_given_x(y=S_t, x=X_t)
    return U_t.detach().cpu().numpy()


def run_incremental_training_and_snapshots(x: np.ndarray, y: np.ndarray, config: Dict):
    alpha = config.get("alpha", 0.9)
    net_cfg = dict(config.get("neural_ot", {}))
    inc = dict(config.get("incremental", {}))
    viz = dict(config.get("viz", {}))

    splits = _prepare_splits_and_base(x, y, config)

    x_dim = splits["X_train"].shape[1]
    y_dim = splits["S_train_full"].shape[1]

    model = _init_amortized_model(x_dim, y_dim, net_cfg | {"device": inc.get("device"), "dtype": inc.get("dtype")})
    loader = _make_loader(splits["X_tr_in"], splits["S_tr_in"], inc.get("batch_size", 256), net_cfg | {"device": inc.get("device"), "dtype": inc.get("dtype")})

    total = int(inc.get("total_epochs", 200))
    step = int(inc.get("step_epochs", 50))

    os.makedirs(inc.get("save_dir", "outputs/ot_neural_inspect"), exist_ok=True)

    epochs_done = 0
    while epochs_done < total:
        this_step = min(step, total - epochs_done)
        tp = _TrainParameters(
            number_of_epochs_to_train=this_step,
            optimizer_parameters={"lr": float(inc.get("lr", 1e-3))},
            scheduler_parameters={},
            verbose=True,
        )
        model.fit(loader, train_parameters=tp)
        epochs_done += this_step

        # Calibrate threshold on held-out residuals (from cal split)
        threshold = _calibrate_neural_ot_cp(model, splits["X_ca_in"], splits["S_ca_in"], alpha=alpha)
        # Push full train/test residuals to U for visualization
        U_train = _push_to_U(model, splits["X_train"], splits["S_train_full"])
        U_test = _push_to_U(model, splits["X_test"], splits["S_test_full"])

        # Make figures
        fig1 = plot_scatter_before_after(splits["S_train_full"], splits["S_test_full"], U_train, U_test, viz)
        fig2 = plot_marginals(U_train, U_test, viz)
        fig3 = plot_radius(U_train, U_test, viz)

        # Save snapshots
        prefix = inc.get("prefix", "uniformity_snap")
        save_dir = inc.get("save_dir", "outputs/ot_neural_inspect")
        fig1.savefig(os.path.join(save_dir, f"{prefix}_scatter_e{epochs_done}.pdf"), bbox_inches="tight")
        fig2.savefig(os.path.join(save_dir, f"{prefix}_marginals_e{epochs_done}.pdf"), bbox_inches="tight")
        fig3.savefig(os.path.join(save_dir, f"{prefix}_radius_e{epochs_done}.pdf"), bbox_inches="tight")

        plt.close(fig1); plt.close(fig2); plt.close(fig3)

    return model

# %%
# Incremental training with snapshots every step_epochs
if CONFIG.get("incremental", {}).get("enabled", False):
    _ = run_incremental_training_and_snapshots(x, y, CONFIG) 