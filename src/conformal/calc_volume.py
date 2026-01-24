import os
import argparse
from pathlib import Path
import pandas as pd
from tqdm import trange

import numpy as np
import torch

from conformal.classes.conformalizers import SplitConformalPredictor
from conformal.real_datasets.reproducible_split import get_dataset_split
from conformal.score_calculators.cvq_regressor import CVQRegressor

RESULTS_DIR = "./conformal_results_u/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="scm20d")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--n_cpus", type=int, default=8)
    parser.add_argument("-p", "--path", type=str, default=RESULTS_DIR)
    args = parser.parse_args()
    print(f"{args=}")

    os.makedirs(Path(args.path) / args.dataset / str(args.seed), exist_ok=True)
    results_dir = Path(args.path)

    dataframe_as_dict = {
        "dataset_name": [],
        "seed": [],
        "method_name": [],
        "alpha": [],
        "volume": [],
        "log_vol_d": []
    }

    for alpha in [0.1, 0.2, 0.3]:

        test_log_volumes = []

        ds = get_dataset_split(name=args.dataset, seed=args.seed)
        print(ds.n_outputs)
        model_u = CVQRegressor.create_or_load(
            path=Path(results_dir / args.dataset / str(args.seed)),
            args=args,
            dataset_split=ds
        )
        model_u.model.eval()

        scores_u_cal = model_u.calculate_scores(ds.X_cal, ds.Y_cal)
        scores_u_test = model_u.calculate_scores(ds.X_test, ds.Y_test)
        
        pb_method = SplitConformalPredictor(
            d_y=ds.n_outputs, seed=0, alpha=0.1, lower_is_better=True
        )
        pb_method.fit(ds.X_cal, scores_u_cal["MK Rank"], alpha=alpha)
        print(f"PB threshold: ", pb_method.threshold)

        test_progress_bar = trange(min(ds.X_test.shape[0], 100))
        for x_index in test_progress_bar:
            x = ds.X_test[x_index]
            test_log_volumes.append(
                model_u.model.get_log_volume(
                    torch.tensor(x, dtype=torch.float32),
                    pb_method.threshold,
                    number_of_points_to_estimate_bounding_box=1000,
                    number_of_points_to_estimate_volume=1000
                )
            )
            mean, std = torch.tensor(test_log_volumes).div(ds.n_outputs).mean().item(
            ), torch.tensor(test_log_volumes).div(ds.n_outputs).std().item()
            test_progress_bar.set_postfix(
                {
                    "index": x_index,
                    "mean": mean,
                    "std": std,
                }
            )

        dataframe_as_dict["dataset_name"].append(args.dataset)
        dataframe_as_dict["seed"].append(args.seed)
        dataframe_as_dict["method_name"].append("PB")
        dataframe_as_dict["alpha"].append(alpha)
        dataframe_as_dict["volume"].append(float(np.mean(np.exp(test_log_volumes))))
        dataframe_as_dict["log_vol_d"].append(float(np.mean(test_log_volumes) / ds.n_outputs))

    df_vol = pd.DataFrame.from_dict(dataframe_as_dict)
    df_vol.to_csv(
        results_dir / args.dataset / str(args.seed) / "test_volumes.csv", index=False
    )
    df_vol.to_feather(
        results_dir / args.dataset / str(args.seed) / "test_volumes.feather"
    )
    print(df_vol)
    print("Done!")
