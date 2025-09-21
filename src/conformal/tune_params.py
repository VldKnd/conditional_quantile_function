import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from conformal.real_datasets.reproducible_split import get_dataset_split
from conformal.wrappers.cvq_regressor import CVQRegressor
from utils.network import get_total_number_of_parameters
from utils.quantile import get_quantile_level_analytically

from conformal.experiment import RESULTS_DIR

grid_cvqr_small = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [256, 512, 1024, 2048],
    "n_epochs": [50, 100, 150],
    "warmup_iterations": [50],
    "kwargs": [
        {
            "hidden_dimension": 10,
            "number_of_hidden_layers": 1
        },
        {
            "hidden_dimension": 8,
            "number_of_hidden_layers": 2
        },
        {
            "hidden_dimension": 6,
            "number_of_hidden_layers": 3
        },
    ],
}

alpha_grid = np.linspace(0.1, 0.9, 9)
alpha_grid_torch = torch.tensor(alpha_grid)


def run_tuning(args):
    tuning_path = Path(RESULTS_DIR) / args.dataset / str(args.seed)
    os.makedirs(tuning_path, exist_ok=True)
    fn_feather = tuning_path / "tuning.feather"
    fn_csv = tuning_path / "tuning.csv"

    ds = get_dataset_split(name=args.dataset, seed=args.seed)
    grid = grid_cvqr_small
    if ds.n_train > 10_000:
        grid["batch_size"] = [1024, 2048]
        for d in grid["kwargs"]:
            d["hidden_dimension"] += 2
    if ds.n_train > 55_000:
        grid["batch_size"] = [4096, 8192]
        for d in grid["kwargs"]:
            d["hidden_dimension"] += 4

    records = []

    for learning_rate in grid["learning_rate"]:
        for batch_size in grid["batch_size"]:
            for n_epochs in grid["n_epochs"]:
                for warmup_iterations in grid["warmup_iterations"]:
                    for kwargs in grid["kwargs"]:
                        params = dict(
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            warmup_iterations=warmup_iterations,
                            feature_dimension=ds.n_features,
                            response_dimension=ds.n_outputs,
                            **kwargs
                        )
                        print(f"Trying {params=}")
                        reg_cvqr = CVQRegressor(**params)
                        print(
                            f"Number of parameters: {get_total_number_of_parameters(reg_cvqr.model.potential_network)}, "
                            f"number of training samples: {ds.n_train}."
                        )
                        reg_cvqr.fit(ds.X_train, ds.Y_train)
                        Q_pred = reg_cvqr.predict_quantile(ds.X_test, ds.Y_test)
                        U_ranks = np.linalg.norm(Q_pred, axis=-1, keepdims=True)
                        levels = get_quantile_level_analytically(
                            alpha=alpha_grid_torch,
                            distribution="gaussian",
                            dimension=ds.n_outputs,
                        ).reshape(1, -1).numpy(force=True)
                        errors = np.abs((U_ranks <= levels).mean(axis=0) - alpha_grid)
                        err = errors.mean()
                        records.append(dict(error=err, errors=errors, **params))
                        print(records[-1])
                        df = pd.DataFrame(records)
                        df.to_feather(fn_feather)
                        df.to_csv(fn_csv, index=False)

    df = pd.DataFrame(records)

    df.to_feather(fn_feather)
    df.to_csv(fn_csv)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="rf1")
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"{args=}")
    results = run_tuning(args)
    print(results)
    print("Done!")
