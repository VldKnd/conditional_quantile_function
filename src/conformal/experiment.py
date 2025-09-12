import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

from conformal.classes.method_desc import ConformalMethodDescription
from conformal.real_datasets.reproducible_split import get_dataset_split
from conformal.wrappers.cvq_regressor import CVQRegressor, CPFlowRegressor, ScoreCalculator
from conformal.wrappers.rf_score import RandomForestWithScore
from metrics.wsc import wsc_unbiased
from utils.network import get_total_number_of_parameters
from conformal.method_zoo import section5, baselines, cpflow_based

RESULTS_DIR = "./conformal_results"

# CVQR configuration for ~ 1000 parameters
_model_config_small = {
    "hidden_dimension": 8,
    "number_of_hidden_layers": 2,
    "batch_size": 256,
    "n_epochs": 50,
    "learning_rate": 0.01,
    "dtype": torch.float32,
}


def run_experiment(args):
    # Decide what methods to test\
    methods: list[ConformalMethodDescription] = []
    if args.baselines or args.all:
        methods += baselines.copy()
    if args.ours or args.all:
        methods += section5.copy()
    if args.cpflow or args.all:
        methods += cpflow_based.copy()

    print(f"Testing methods: {methods}")
    if len(methods) < 1:
        print("Nothing to do!")
        return pd.DataFrame()

    current_seed_dir = Path(RESULTS_DIR) / args.dataset / str(args.seed)
    os.makedirs(current_seed_dir, exist_ok=True)

    trained_model_path_cvqr = current_seed_dir / f"model_cvqr.pth"
    trained_model_path_cpflow = current_seed_dir / f"model_cpflow.pth"

    #alpha = 0.3
    #alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    alphas = [0.1, 0.2, 0.3]
    # Number of samples for volume estimation
    n_samples = 10_000

    ds = get_dataset_split(name=args.dataset, seed=args.seed)
    if ds.n_train > 10_000:
        _model_config_small["batch_size"] = 1024
    if ds.n_train > 55_000:
        _model_config_small["batch_size"] = 8192

    #_model_config_small["n_epochs"] = 1

    # Instantiate conformal methods
    required_model_names = set()
    for method in methods:
        required_model_names.add(method.base_model_name)
        method.instance = method.cls(**method.kwargs, seed=args.seed)

    # Base multidimensional quantile model
    reg_cvqr = CVQRegressor(
        feature_dimension=ds.n_features,
        response_dimension=ds.n_outputs,
        **_model_config_small
    )
    print(
        f"Number of parameters: {get_total_number_of_parameters(reg_cvqr.model.potential_network)}, "
        f"number of training samples: {ds.n_train}."
    )

    # Base model for OT-CP: Random Forest
    rf = RandomForestWithScore(random_state=args.seed, n_jobs=-1)

    reg_cpflow = CPFlowRegressor(        
        feature_dimension=ds.n_features,
        response_dimension=ds.n_outputs,
        **_model_config_small
    )

    score_calculators: dict[str, ScoreCalculator] = {}

    if "CVQRegressor" in required_model_names:
        # Fit base models
        if Path.is_file(trained_model_path_cvqr):
            reg_cvqr.model.load(trained_model_path_cvqr)
        else:
            reg_cvqr.fit(ds.X_train, ds.Y_train)
            reg_cvqr.model.save(trained_model_path_cvqr)
        score_calculators["CVQRegressor"] = reg_cvqr

    if "RandomForest" in required_model_names:
        rf.fit(ds.X_train, ds.Y_train)
        score_calculators["RandomForest"] = rf
    
    if "CPFlowRegressor" in required_model_names:
        if Path.is_file(trained_model_path_cpflow):
            reg_cpflow.model.load(trained_model_path_cpflow)
        else:
            reg_cpflow.fit(ds.X_train, ds.Y_train)
            reg_cpflow.model.save(trained_model_path_cpflow)
        score_calculators["CPFlowRegressor"] = reg_cvqr


    def _calculate_scores(X, Y):
        return {base_model_name: score_calculators[base_model_name].calculate_scores(X, Y) for base_model_name in required_model_names}

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
            method.instance.fit(
                X_cal=ds.X_cal, scores_cal=scores_calibration[method.base_model_name][method.score_name], alpha=alpha
            )
            is_covered = method.instance.is_covered(
                X_test=ds.X_test, scores_test=scores_test[method.base_model_name][method.score_name]
            )
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
            records_alpha.append(
                dict(
                    dataset_name=args.dataset,
                    seed=args.seed,
                    method_name=method.name,
                    method_name_mathtext=method.name_mathtext,
                    score_name=method.score_name,
                    conformalizer=method.class_name,
                    base_model_name=method.base_model_name,
                    alpha=alpha,
                    marginal_coverage=coverage,
                    worst_slab_coverage=wsc
                )
            )

        # For each test point Xi, sample Y values randomly in the range of all observed Ys,
        # then calculate the ratio of covered points and multiply by the bounding box's volume
        coverage_ratios = np.zeros((len(methods), ds.n_test))
        print(f"{alpha=:.2f}, estimating areas:")
        for i in tqdm(range(ds.n_test)):
            X_samples = np.repeat(ds.X_test[i:i + 1], repeats=n_samples, axis=0)
            Y_smaples = ymin + rng.random((n_samples, ds.n_outputs)) * (ymax - ymin)

            scores_samples = _calculate_scores(X_samples, Y_smaples)
            for j, method in enumerate(methods):
                if hasattr(method.instance, "get_volume"):
                    # Can estimate volume without sampling
                    is_covered = np.array(
                        [
                            method.instance.get_volume(
                                ds.X_test[i], scores_test[method.base_model_name][method.score_name][i]
                            )
                        ]
                    )
                else:
                    is_covered = method.instance.is_covered(
                        X_samples, scores_samples[method.base_model_name][method.score_name]
                    )
                coverage_ratios[j, i] = is_covered.mean()
        mean_volumes = coverage_ratios.mean(axis=-1) * scale
        for j, _ in enumerate(methods):
            records_alpha[j]["volume"] = mean_volumes[j]
        records += records_alpha

    df_metrics = pd.DataFrame(records)

    df_metrics.to_feather(current_seed_dir / f"metrics_all.feather")

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="rf1")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--baselines", action='store_true')
    parser.add_argument("--ours", action='store_true')
    parser.add_argument("--cpflow", action='store_true')
    parser.add_argument("--all", action='store_true')
    args = parser.parse_args()
    print(f"{args=}")
    results = run_experiment(args)
    print(results.head())
    print("Done!")
