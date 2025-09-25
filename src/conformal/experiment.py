from dataclasses import asdict
import warnings
import copy
import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import scipy
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

from conformal.classes.method_desc import ConformalMethodDescription
from conformal.plots.diagnostic import draw_density_scores_pair, draw_qq_scores_pair
from conformal.real_datasets.reproducible_split import get_dataset_split
import conformal.score_calculators as all_score_calculators
from metrics.wsc import wsc_unbiased
from pushforward_operators.neural_quantile_regression.amortized_neural_quantile_regression import AmortizedNeuralQuantileRegression
from utils.network import get_total_number_of_parameters
from conformal.method_zoo import section5, section5_y, baselines, cpflow_based, section5_rf, section5_y_rf

RESULTS_DIR = "./conformal_results_u"

# Batch size for Hessians in score calculation
_scores_batch_size = 4096


def run_experiment(args):
    # Decide what methods to test
    methods: list[ConformalMethodDescription] = []
    if args.baselines or args.all:
        methods += baselines.copy()
    if args.ours or args.all:
        methods += section5.copy() + section5_y.copy()
    if args.cpflow or args.all:
        methods += cpflow_based.copy()
    if args.rf or args.all:
        methods += section5_rf.copy() + section5_y_rf.copy()

    #df_methods_desc = pd.DataFrame([asdict(method) for method in methods])
    print(f"Testing methods: \n {methods}")
    if len(methods) < 1:
        print("Nothing to do!")
        return pd.DataFrame()

    skip_area_computation = args.skip_area_computation

    current_seed_dir = Path(args.path if args.path is not None else RESULTS_DIR
                            ) / args.dataset / str(args.seed)
    os.makedirs(current_seed_dir, exist_ok=True)

    # Metrics paths
    fn_feather = current_seed_dir / f"metrics_all.feather"
    fn_csv = current_seed_dir / f"metrics_all.csv"

    alphas = [0.1, 0.2, 0.3]

    # Number of samples for volume estimation
    n_samples = 10_000

    ds = get_dataset_split(name=args.dataset, seed=args.seed)

    # Instantiate conformal methods
    required_model_names = set()
    for method in methods:
        required_model_names.add(method.base_model_name)
        method.instance = method.cls(**method.kwargs, seed=args.seed, d_y=ds.n_outputs)

    score_calculators = {
        base_model_name:
        getattr(all_score_calculators, base_model_name
                ).create_or_load(path=current_seed_dir, args=args, dataset_split=ds)
        for base_model_name in required_model_names
    }

    def _calculate_scores(X, Y):
        return {
            base_model_name:
            score_calculators[base_model_name].calculate_scores(
                X, Y, batch_size=_scores_batch_size
            )
            for base_model_name in required_model_names
        }

    # Calculate scores for all base models
    scores_calibration = _calculate_scores(ds.X_cal, ds.Y_cal)
    if args.area_only:
        #
        scores_test = scores_calibration
    else:
        scores_test = _calculate_scores(ds.X_test, ds.Y_test)

    # Diagnostic plotting
    for model_name in scores_calibration.keys():
        print(f"{list(scores_calibration.keys())=}")
        if "MK Quantile" in scores_calibration[model_name]:
            draw_qq_scores_pair(
                scores_calibration[model_name]["MK Quantile"],
                scores_test[model_name]["MK Quantile"],
                title_2="Calibration",
                sup_title=f"{model_name}, {args.dataset}, {args.seed}",
                save_path=current_seed_dir / f"{model_name}_QQ.png"
            )
            draw_density_scores_pair(
                scores_calibration[model_name]["MK Quantile"],
                scores_test[model_name]["MK Quantile"],
                title_2="Calibration",
                sup_title=f"{model_name}, {args.dataset}, {args.seed}",
                save_path=current_seed_dir / f"{model_name}_U_kde.png"
            )

    # Compute metrics
    records = []

    rng = np.random.default_rng(args.seed)
    ymin = ds.Y_train.min(axis=0)
    ymax = ds.Y_train.max(axis=0)

    scale = np.prod(ymax - ymin)

    print("Computing metrics")
    for alpha in alphas:
        records_alpha = []
        for method in methods:
            print(f"{alpha=:.2f}, {method.name=}")
            record = dict(
                dataset_name=args.dataset,
                seed=args.seed,
                method_name=method.name,
                method_name_mathtext=method.name_mathtext,
                score_name=method.score_name,
                conformalizer=method.class_name,
                base_model_name=method.base_model_name,
                alpha=alpha,
            )
            method.instance.fit(
                X_cal=ds.X_cal,
                scores_cal=scores_calibration[method.base_model_name][method.score_name
                                                                      ],
                alpha=alpha
            )
            if not args.area_only:
                is_covered = method.instance.is_covered(
                    X_test=ds.X_test,
                    scores_test=scores_test[method.base_model_name][method.score_name],
                    verbose=True
                )
                coverage = is_covered.mean()
                wsc_list = []
                for k in range(10):
                    wsc_list.append(
                        wsc_unbiased(
                            ds.X_test,
                            is_covered,
                            delta=0.1,
                            M=10000,
                            random_state=args.seed + k,
                            n_cpus=args.n_cpus,
                            verbose=True
                        )
                    )
                wsc = np.mean(wsc_list)
                record.update(
                    marginal_coverage=coverage,
                    worst_slab_coverage=wsc,
                    worst_slab_coverage_se=scipy.stats.sem(wsc_list)
                )
                print(f"{method.name}, {coverage=:.4f}, {wsc=:.4f}")
            records_alpha.append(record)
        # Print the incomplete results (without volume) for this alpha
        print(pd.DataFrame(records_alpha))
        # Save all results obtained so far (without volume)
        pd.DataFrame(records + records_alpha).to_csv(fn_csv, index=False)

        # For each test point Xi, sample Y values randomly in the range of all observed Ys,
        # then calculate the ratio of covered points and multiply by the bounding box's volume
        volumes = np.zeros((len(methods), ds.n_test))
        if not skip_area_computation:
            print(f"{alpha=:.2f}, estimating areas:")
            for i in tqdm(range(ds.n_test)):
                X_samples = np.repeat(ds.X_test[i:i + 1], repeats=n_samples, axis=0)
                Y_smaples = ymin + rng.random((n_samples, ds.n_outputs)) * (ymax - ymin)

                scores_samples = _calculate_scores(X_samples, Y_smaples)
                for j, method in enumerate(methods):
                    if hasattr(method.instance, "get_volume"):
                        # Can estimate volume without sampling
                        volume = method.instance.get_volume(
                            ds.X_test[i],
                            scores_test[method.base_model_name][method.score_name][i]
                        )
                    else:
                        # Approximate volume using samples
                        volume = method.instance.is_covered(
                            X_samples,
                            scores_samples[method.base_model_name][method.score_name]
                        ).mean() * scale
                    volumes[j, i] = volume
            mean_volumes = volumes.mean(axis=-1)
            for j, _ in enumerate(methods):
                records_alpha[j]["volume"] = mean_volumes[j]
        else:
            mean_volumes = volumes.mean(axis=-1)
        # Print results for this alpha
        print(pd.DataFrame(records_alpha))
        records += records_alpha

        # Save all results obtained so far
        pd.DataFrame(records).to_csv(fn_csv, index=False)

    df_metrics = pd.DataFrame(records)

    df_metrics.to_feather(fn_feather)

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="rf1")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-j", "--jobs", type=int, default=-1)
    parser.add_argument("-c", "--n_cpus", type=int, default=8)
    parser.add_argument("--baselines", action='store_true')
    parser.add_argument("--ours", action='store_true')
    parser.add_argument("--cpflow", action='store_true')
    parser.add_argument("--rf", action='store_true')
    parser.add_argument("--all", action='store_true')
    parser.add_argument(
        "-f", "--skip-area-computation", action='store_true', default=False
    )
    parser.add_argument("-p", "--path", type=str, default=RESULTS_DIR)
    parser.add_argument("--area-only", action='store_true', default=False)
    args = parser.parse_args()
    print(f"{args=}")
    results = run_experiment(args)
    print(results)
    print("Done!")
