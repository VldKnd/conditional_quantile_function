# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Tuple

# Import Neural OT CP (same import style as other scripts here)
from conformal.neural_ot_predictors import NeuralOTCPScore


# -----------------------------
# Config (Jupyter-friendly)
# -----------------------------
CONFIG: Dict = {
    "dataset_name": "enb",  # 2D output dataset
    "alpha": 0.9,
    "seed": 567,
    "test_size": 0.5,  # split for test; then we'll split test into cal/test equally
    "visualization": {
        "x_choice": "first",  # "first" or integer index
        "quantile_levels": [0.5, 0.8, 0.9, 0.95],
        "num_angles": 256,
        "figsize": (9, 4),
        "out_dir": "outputs/ot_neural_inspect",
        "filename": "enb_neural_ot_contours.pdf",
        "show": True,
    },
    "neural_ot": {
        "number_of_hidden_layers": 2,
        "z_dimension": 32,
        "batch_size": 256,
        "num_epochs": 200,
        "lr": 1e-3,
        "device": None,     # auto-selects CUDA if available when None
        "dtype": "float32",
    },
}


# -----------------------------
# Data utilities
# -----------------------------

def load_real_dataframe(name: str) -> pd.DataFrame:
    """Load a real dataset by name. Falls back to CSV if ARFF is unavailable.

    This mirrors the logic used in the real-data experiments.
    """


    repo_root  = os.path.dirname("/home/labcmap/mahmoud.hegazy/conditional_quantile_function/")
    arff_path = os.path.join(repo_root, "regression_refactored", "data", f"{name}.arff")

    from scipy.io import arff  # type: ignore
    data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    return df.dropna()


def get_xy_from_df(df: pd.DataFrame, y_dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataframe into features X and responses Y, taking the last y_dim columns as Y."""
    y = df[df.columns[-y_dim:]].values
    x = df[df.columns[:-y_dim]].values
    return x.astype(np.float64), y.astype(np.float64)



def chi2_radius_2d(alphas: np.ndarray) -> np.ndarray:
    """For 2D standard normal U, radius r such that P(||U|| <= r) = alpha.
    Since R^2 ~ ChiSquare(df=2), CDF(R) = 1 - exp(-r^2/2) => r = sqrt(-2 log(1 - alpha)).
    """
    alphas = np.clip(alphas, 1e-6, 1 - 1e-6)
    return np.sqrt(-2.0 * np.log(1.0 - alphas))


def make_circle_points(radius: float, num_angles: int = 256) -> np.ndarray:
    thetas = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    return np.stack([radius * np.cos(thetas), radius * np.sin(thetas)], axis=-1)


# -----------------------------
# Training and visualization
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
            "y_pred_cal": y_pred_cal,
            "y_pred_test": y_pred_test,
        },
    }


# %%
config = CONFIG
np.random.seed(config.get("seed", 567))

# Data
df = load_real_dataframe(config.get("dataset_name", "enb"))
x, y = get_xy_from_df(df, y_dim=2)
print(x.shape, y.shape)

# %%
plt.scatter(y[:, 0], y[:, 1])

# %%

# Train
models = train_models(x, y, config)

# %%
base = models["base"]
X_test= models["splits"]["X_test"]
y_test = models["splits"]["y_test"]
y_pred_test = base.predict(X_test)
residuals = y_test - y_pred_test
plt.scatter(residuals[:,0], residuals[:,1])




# %%
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
neural = models["neural"]
model_torch = neural.model_
model_torch.eval()
residuals = torch.from_numpy(residuals).to(device=device, dtype=dtype)
u_test_pred = model_torch.push_y_given_x(x=X_test, y=residuals)
u_test_pred = u_test_pred.cpu().numpy()
plt.scatter(u_test_pred[:,0], u_test_pred[:,1])







