import argparse
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from time import time
from tqdm import tqdm
from itertools import product
from conformal.ot_predictors import OTCPAdaptiveKNN
from conformal.ellipsoidal_predictors import EllipsoidalLocal
from functions import (
    get_out_dirs,
    set_seed,
    setup_plotting,
    get_palette,
)

from conformal.neural_ot_predictors import NeuralOTCPScore


def load_real_dataframe(name: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arff_path = os.path.join(base_dir, "OTCP", "data", f"{name}.arff")
    csv_path = os.path.join(base_dir, "regression_refactored", "csv", f"{name}.csv")

    try:
        from scipy.io import arff

        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
    except (FileNotFoundError, ImportError):
        df = pd.read_csv(csv_path)
    return df.dropna()


def get_datasets(setting):
    """Get dataset names and dimensions based on the setting."""
    if setting == "medium_arff":
        names = ["enb", "atp1d", "jura", "rf1", "wq", "scm20d"]
        dims = [2, 6, 7, 8, 14, 16]
    elif setting == "medium_csv":
        names = ["ansur2", "bio", "air", "births1", "taxi", "households"]
        dims = [2, 2, 2, 2, 6, 2]
    elif setting == "large":
        names = ["rf1", "scm20d"]
        dims = [8, 16]
    else:
        raise ValueError(f"Unknown setting: {setting}")
    return names, dims


def _parse_int_grid(arg_val: str | None):
    if not arg_val:
        return []
    if isinstance(arg_val, list):
        return [int(v) for v in arg_val]
    return [int(v.strip()) for v in str(arg_val).split(",") if v.strip()]


def get_args():
    parser = argparse.ArgumentParser(description="Real-data experiments for OT-CP.")
    parser.add_argument(
        "--setting",
        type=str,
        default="medium_arff",
        choices=["medium_arff", "medium_csv", "large"],
    )
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--nrep", type=int, default=5)
    parser.add_argument("--seed", type=int, default=567)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--csv-dir", type=str, default="csv")
    # Neural OT hyperparameters (single values)
    parser.add_argument(
        "--neural-ot-mode",
        type=str,
        default="score",
        choices=["score", "direct"],
        help="Neural OT mode: learn residual score or direct Y.",
    )
    parser.add_argument("--neural-epochs", type=int, default=200)
    parser.add_argument("--neural-batch", type=int, default=256)
    parser.add_argument("--neural-hidden", type=int, default=2)
    parser.add_argument("--neural-z", type=int, default=32)
    # Neural OT hyperparameter grids (comma-separated lists)
    parser.add_argument(
        "--neural-epochs-grid",
        type=str,
        default="",
        help="Comma-separated list of epoch counts for sweep (e.g., '100,200,400').",
    )
    parser.add_argument(
        "--neural-hidden-grid",
        type=str,
        default="",
        help="Comma-separated list of hidden layer counts for sweep (e.g., '1,2,3').",
    )
    parser.add_argument(
        "--neural-z-grid",
        type=str,
        default="",
        help="Comma-separated list of latent dims for sweep (e.g., '16,32,64').",
    )
    parser.add_argument(
        "--neural-batch-grid",
        type=str,
        default="",
        help="Comma-separated list of batch sizes for sweep (e.g., '128,256').",
    )
    return parser.parse_args()


def run_one_dataset(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    n_repeats: int = 5,
    n_neighbors: int = 50,
    dataset_idx: int = 0,
    t0: float = 0.0,
    alpha: float = 0.9,
    *,
    neural_ot_mode: str = "score",
    neural_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Run experiments for one dataset."""
    results = []

    for i in range(n_repeats):
        print(f"{dataset_idx}, {i}: {time() - t0:.3f}")
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        X_test, X_cal, y_test, y_cal = train_test_split(X_test, y_test, test_size=0.5)

        current_n_neighbors = min(n_neighbors, X_cal.shape[0])

        predictors = {
            "OT-CP+": OTCPAdaptiveKNN(alpha=alpha, n_neighbors=current_n_neighbors),
            "ELL": EllipsoidalLocal(alpha=alpha, n_neighbors=current_n_neighbors),
        }

        # Always include Neural OT predictors by default
        if neural_ot_mode == "score":
            predictors["NeuralOT-Score"] = NeuralOTCPScore(
                alpha=alpha,
                nn_kwargs=(neural_kwargs),
            )
        else:
            print("NeuralOT-Direct is not implemented yet")
            raise NotImplementedError("NeuralOT-Direct is not implemented yet")

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        y_pred_cal = model.predict(X_cal)
        y_pred_test = model.predict(X_test)

        for method, predictor in predictors.items():
            t_start = time()
            if method == "ELL":
                predictor.fit(X_cal, y_cal, y_pred_cal)
            elif method.startswith("NeuralOT"):
                if neural_ot_mode == "score":
                    predictor.fit(X_cal, y_cal, y_pred_cal)
                else:
                    predictor.fit(X_cal, y_cal)
            else:
                predictor.fit(X_cal, y_cal - y_pred_cal)
            t_elapsed = time() - t_start

            # Build a label that encodes neural hyperparameters for plotting across configs
            if method.startswith("NeuralOT") and neural_kwargs is not None:
                method_label = (
                    f"{method}(e{neural_kwargs.get('num_epochs')},"
                    f"h{neural_kwargs.get('number_of_hidden_layers')},"
                    f"z{neural_kwargs.get('z_dimension')},"
                    f"b{neural_kwargs.get('batch_size')})"
                )
            else:
                method_label = method

            metrics = predictor.metrics(X_test, y_test, y_pred_test)
            coverage = metrics["avg_coverage"]
            volume = metrics.get("avg_volume")

            worst_set_coverage, _ = predictor.worst_set_coverage(
                X_test, y_test, y_pred_test, n_clusters=int(len(X_cal) * 0.2)
            )

            results.append(
                {
                    "method": method,
                    "method_label": method_label,
                    "coverage": coverage,
                    "volume": volume,
                    "worst_set_coverage": worst_set_coverage,
                    "time": t_elapsed,
                    "Data": name,
                }
            )
    return pd.DataFrame(results)


def run_real_data_experiments(args):
    """Run all real data experiments, with optional hyperparameter sweep for Neural OT."""

    set_seed(0)
    datasets, dims = get_datasets(args.setting)

    t0 = time()
    out_dir, csv_dir = get_out_dirs(args.out_dir, args.csv_dir)

    # Parse grids; if empty, fall back to single values
    epochs_grid = _parse_int_grid(args.neural_epochs_grid) or [args.neural_epochs]
    hidden_grid = _parse_int_grid(args.neural_hidden_grid) or [args.neural_hidden]
    z_grid = _parse_int_grid(args.neural_z_grid) or [args.neural_z]
    batch_grid = _parse_int_grid(args.neural_batch_grid) or [args.neural_batch]

    df_all = pd.DataFrame()
    for (nepochs, nhid, nz, nbatch) in product(
        epochs_grid, hidden_grid, z_grid, batch_grid
    ):
        print(f"Hyperparams: epochs={nepochs}, hidden_layers={nhid}, z_dim={nz}, batch={nbatch}")
        neural_kwargs = {
            "num_epochs": nepochs,
            "batch_size": nbatch,
            "number_of_hidden_layers": nhid,
            "z_dimension": nz,
        }

        # Collect per-sweep results to also plot per-sweep figures
        df_sweep = pd.DataFrame()

        for i, (name, dim) in enumerate(tqdm(zip(datasets, dims), desc="Datasets")):
            print(f"Running dataset {i+1}/{len(datasets)}: {name}")
            df = load_real_dataframe(name)

            y = df[df.columns[-dim:]].values
            x = df[df.columns[:-dim]].values

            n_neighbors = int(
                x.shape[0] * 0.25 * 0.1
            )  # 25% for test, 10% of that for neighbors

            df_res = run_one_dataset(
                x,
                y,
                name,
                n_repeats=args.nrep,
                dataset_idx=i,
                t0=t0,
                n_neighbors=n_neighbors,
                alpha=args.alpha,
                neural_ot_mode=args.neural_ot_mode,
                neural_kwargs=neural_kwargs,
            )

            # Tag results with hyperparameters
            df_res["neural_num_epochs"] = nepochs
            df_res["neural_hidden_layers"] = nhid
            df_res["neural_z_dimension"] = nz
            df_res["neural_batch_size"] = nbatch

            df_all = pd.concat([df_all, df_res])
            df_sweep = pd.concat([df_sweep, df_res])

            # Save intermediate results and plots after each dataset
            print(f"Saving intermediate results to {csv_dir}")

            metrics_to_save = {
                "coverage": "Coverage",
                "worst_set_coverage": "Worst Set Coverage",
                "volume": "Volume",
                "time": "Time",
            }

            for metric_col, metric_name in metrics_to_save.items():
                df_metric = df_all[[
                    metric_col,
                    "method",
                    "method_label",
                    "Data",
                    "neural_num_epochs",
                    "neural_hidden_layers",
                    "neural_z_dimension",
                    "neural_batch_size",
                ]].copy()
                df_metric.rename(
                    columns={
                        metric_col: metric_name,
                        "method": "Method",
                        "method_label": "MethodLabel",
                    },
                    inplace=True,
                )
                filename = os.path.join(
                    csv_dir, f"reg_{args.setting}_realdata_{metric_name}.csv"
                )
                df_metric.to_csv(filename, index=False)

            print(f"Saving intermediate plots to {out_dir}")
            plot_metrics(df_all, args.setting, out_dir, args.alpha)

        # Also save per-sweep figures in a dedicated subdirectory
        sweep_dir = os.path.join(
            out_dir,
            f"sweep_e{nepochs}_h{nhid}_z{nz}_b{nbatch}",
        )
        os.makedirs(sweep_dir, exist_ok=True)
        print(f"Saving sweep-specific plots to {sweep_dir}")
        plot_metrics(df_sweep, args.setting, sweep_dir, args.alpha)


def plot_metrics(df, setting, out_dir, alpha):
    pal = get_palette()
    plot_config = {
        "coverage": {"ylabel": "Coverage", "filename": "Coverage.pdf"},
        "worst_set_coverage": {
            "ylabel": "Worst Set Coverage",
            "filename": "ConditionalCoverage.pdf",
        },
        "volume": {"ylabel": "Volume", "filename": "Volume.pdf"},
        "time": {"ylabel": "Time", "filename": "Time.pdf"},
    }

    for metric, config in plot_config.items():
        plt.figure()

        data_to_plot = df
        if metric == "volume":
            data_to_plot = df.copy()
            for dataname in data_to_plot["Data"].unique():
                norm = (
                    np.var(data_to_plot[data_to_plot["Data"] == dataname]["volume"]) 
                    ** 0.5
                )
                if norm > 0:
                    data_to_plot.loc[data_to_plot["Data"] == dataname, "volume"] /= norm

        hue_key = "method_label" if "method_label" in data_to_plot.columns else "method"
        g = sns.catplot(
            data=data_to_plot,
            kind="bar",
            x="Data",
            y=metric,
            hue=hue_key,
            palette=[
                pal.get("mk", "C0"),
                pal.get("ell", "C1"),
                pal.get("mk", "C2"),
                pal.get("ell", "C3"),
            ],
            height=5,
            aspect=1.2 if setting.startswith("medium") else 0.6,
            capsize=0.3,
        )
        if "Coverage" in config["ylabel"]:
            plt.gca().axhline(alpha, color="black", linestyle="dashed", linewidth=2)

        plt.ylabel(config["ylabel"])
        plt.savefig(
            os.path.join(out_dir, f"{setting}_{config['filename']}"),
            bbox_inches="tight",
        )
        plt.close("all")



def main():
    args = get_args()
    run_real_data_experiments(args)


if __name__ == "__main__":
    main()
