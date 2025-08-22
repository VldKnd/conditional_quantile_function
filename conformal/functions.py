import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

_THIS_DIR = os.path.dirname(__file__)


def get_out_dirs(out_dir: str | None, csv_dir: str | None):
    """Return output and csv directories, creating them if missing."""
    out = out_dir or os.path.join(_THIS_DIR, "outputs")
    csv = csv_dir or os.path.join(_THIS_DIR, "csv")
    os.makedirs(out, exist_ok=True)
    os.makedirs(csv, exist_ok=True)
    return out, csv


def set_seed(seed: int | None = None):
    if seed is None:
        return
    np.random.seed(seed)


def setup_plotting():
    plt.rcParams.update({"font.size": 14})
    sns.set_style("whitegrid")


def get_palette():
    cmap = cm.get_cmap("tab20c")
    return {
        "y1": cmap(1 / 20),
        "y2": cmap(6 / 20),
        "y3": cmap(9 / 20),
        "y4": cmap(14 / 20),
        "mk": "cornflowerblue",
        "ell": "lightpink",
    }
