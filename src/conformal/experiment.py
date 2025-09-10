import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

from conformal.classes.method_desc import ConformalMethodDescription
from conformal.real_datasets.reproducible_split import get_dataset_split
from conformal.wrappers.cvq_regressor import CVQRegressor, calculate_scores_cvqr
from conformal.classes.conformalizers import SplitConformalPredictor, OTCPGlobalPredictor
from metrics.wsc import wsc_unbiased
from utils.network import get_total_number_of_parameters

RESULTS_DIR = "./conformal_results"

# CVQR configuration for ~ 1000 parameters
_model_config_small = {
    "hidden_dimension": 8,
    "number_of_hidden_layers": 2,
    "batch_size": 64,
    "n_epochs": 50,
    "learning_rate": 0.01
}


methods = [
    ConformalMethodDescription(
        name="PB",
        name_mathtext=r"$\mathcal{C}^{\mathrm{pb}}$",
        base_model_name="CVQRegressor",
        score_name="MK Rank",
        class_name="SplitConformalPredictor",
        instance=SplitConformalPredictor()
    ),
    ConformalMethodDescription(
        name="RPB",
        name_mathtext=r"$\mathcal{C}^{\mathrm{rpb}}$",
        base_model_name="CVQRegressor",
        score_name="MK Quantile",
        class_name="OTCPGlobalPredictor",
        instance=OTCPGlobalPredictor()
    ),
    ConformalMethodDescription(
        name="HPD",
        name_mathtext=r"$\mathcal{C}^{\mathrm{HPD}}$",
        base_model_name="CVQRegressor",
        score_name="Log Density",
        class_name="SplitConformalPredictor",
        instance=SplitConformalPredictor(lower_is_better=False)
    ),
    ConformalMethodDescription(
        name="OT-CP-Global",
        name_mathtext=r"$\mathrm{OT}-\mathrm{CP}$",
        base_model_name="RandomForest",
        score_name="Signed Error",
        class_name="OTCPGlobalPredictor",
        instance=OTCPGlobalPredictor()
    ),

]


def calculate_scores_rf(rf: RandomForestRegressor, X:np.ndarray, Y:np.ndarray):
    Y_pred = rf.predict(X)
    signed_error = Y - Y_pred
    return signed_error


def run_experiment(args):
    current_seed_dir = Path(RESULTS_DIR) / args.dataset / str(args.seed)
    os.makedirs(current_seed_dir, exist_ok=True)

    trained_model_path = current_seed_dir / f"model.pth"

    alpha = 0.3
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Number of samples for volume estimation
    n_samples = 10_000

    ds = get_dataset_split(name=args.dataset, seed=args.seed)

    # Base multidimensional quantile model
    reg = CVQRegressor(
        feature_dimension=ds.n_features,
        response_dimension=ds.n_outputs,
        **_model_config_small
    )
    print(
        f"Number of parameters: {get_total_number_of_parameters(reg.model.potential_network)}, "
        f"number of training samples: {ds.n_train}."
    )

    # Base model for OT-CP: Random Forest
    rf = RandomForestRegressor(random_state=args.seed)

    # Fit base models
    if Path.is_file(trained_model_path):
        reg.model.load(trained_model_path)
    else:
        reg.fit(ds.X_train, ds.Y_train)
        reg.model.save(trained_model_path)

    rf.fit(ds.X_train, ds.Y_train)

    def _calculate_scores(X, Y):
        quantiles, ranks, log_p = calculate_scores_cvqr(reg, X, Y)

        signed_error = calculate_scores_rf(rf, X, Y)
        return {
            "MK Quantile": quantiles,
            "MK Rank": ranks,
            "Log Density": log_p,
            "Signed Error": signed_error,
        }

    # Calculate scores for Neural Vector Quantile regression
    scores_calibration = _calculate_scores(ds.X_cal, ds.Y_cal)
    scores_test = _calculate_scores(ds.X_test, ds.Y_test)

    # Compute metrics
    records = []
    records_volumes = []

    rng = np.random.default_rng(args.seed)
    ymin = ds.Y_train.min(axis=0)
    ymax = ds.Y_train.max(axis=0)

    scale = np.prod(ymax - ymin)

    for alpha in alphas:
        records_alpha = []
        for method in methods:
            if hasattr(method, "seed"):
                setattr(method, "seed", args.seed)
            method.instance.fit(scores_calibration[method.score_name], alpha=alpha)
            is_covered = method.instance.is_covered(scores_test[method.score_name])
            coverage = is_covered.mean()
            wsc = wsc_unbiased(
                ds.X_test,
                is_covered,
                delta=0.1,
                M=5000,
                random_state=args.seed,
                n_cpus=8,
                verbose=True
            )
            records_alpha.append(dict(dataset_name=args.dataset, seed=args.seed,
                                method_name=method.name, method_name_mathtext=method.name_mathtext,
                                score_name=method.score_name, base_model_name=method.base_model_name, 
                                alpha=alpha, marginal_coverage=coverage, worst_slab_coverage=wsc))
        
        # For each test point Xi, sample Y values randomly in the range of all observed Ys,
        # then calculate the ratio of covered points and multiply by the bounding box's volume
        coverage_ratios = np.zeros((len(methods), ds.n_test))
        print(f"{alpha=:.2f}, estimating areas:")
        for i in tqdm(range(ds.n_test)):
            X_samples = np.repeat(ds.X_test[i:i + 1], repeats=n_samples, axis=0)
            Y_smaples = ymin + rng.random((n_samples, ds.n_outputs)) * (ymax - ymin)

            scores_samples = _calculate_scores(X_samples, Y_smaples)
            for j, method in enumerate(methods):
                is_covered = method.instance.is_covered(scores_samples[method.score_name])
                coverage_ratios[j, i] = is_covered.mean()
        mean_volumes = coverage_ratios.mean(axis=-1) * scale
        for j, _ in enumerate(methods):
            records_alpha[j]["volume"] = mean_volumes[j]
        records += records_alpha

    df_metrics = pd.DataFrame(records)

    df_metrics.to_feather(current_seed_dir / f"metrics.feather")

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    results = run_experiment(args)
    print(results.head())
    print("Done!")
