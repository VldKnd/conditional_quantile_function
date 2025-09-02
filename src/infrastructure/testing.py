import torch
from tqdm import tqdm
from datasets import (
    BananaDataset,
    QuadraticPotentialConvexBananaDataset,
    TicTacDataset,
    Dataset,
    NotConditionalBananaDataset,
    FNLVQR_MVN,
    PICNN_FNLVQR_Glasses,
    PICNN_FNLVQR_Star,
    PICNN_FNLVQR_Banana,
)
from infrastructure.classes import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map
from metrics import (
    wassertein2, percentage_of_unexplained_variance, sliced_wasserstein2,
    kernel_density_estimate_kl_divergence, kernel_density_estimate_l1_divergence
)
from pushforward_operators import PushForwardOperator


def test_from_json_file(
    path_to_experiment_file: str,
    verbose: bool = False,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
) -> dict:
    experiment = Experiment.load_from_path_to_experiment_file(
        path_to_experiment_file=path_to_experiment_file
    )
    metrics = test(
        experiment,
        verbose=verbose,
        exclude_wasserstein2=exclude_wasserstein2,
        exclude_unexplained_variance_percentage=exclude_unexplained_variance_percentage,
        exclude_sliced_wasserstein2=exclude_sliced_wasserstein2,
        exclude_kde_kl_divergence=exclude_kde_kl_divergence,
        exclude_kde_l1_divergence=exclude_kde_l1_divergence
    )

    if experiment.path_to_metrics is not None:
        torch.save(metrics, experiment.path_to_metrics)

    return metrics


def load_pushforward_operator_from_experiment(
    experiment: Experiment
) -> PushForwardOperator:
    """
    Load a pushforward operator from an experiment.

    Args:
        experiment (Experiment): The experiment to load the pushforward operator from.

    Returns:
        PushForwardOperator: The loaded pushforward operator.
    """
    pushforward_operator = name_to_pushforward_operator_map[
        experiment.pushforward_operator_name](
            **experiment.pushforward_operator_parameters
        )
    pushforward_operator.to(**experiment.tensor_parameters)

    if experiment.path_to_weights is not None:
        pushforward_operator.load(
            experiment.path_to_weights,
            map_location=torch.device(experiment.tensor_parameters["device"])
        )
    else:
        raise ValueError("Path to the model is not specified. Model can not be loaded.")

    pushforward_operator.eval()
    return pushforward_operator


def test_on_synthetic_dataset(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
    verbose: bool = False
) -> dict:
    """
    Test a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to test.
        exclude_wasserstein2 (bool): Whether to exclude the Wasserstein-2 distance.
        exclude_unexplained_variance_percentage (bool): Whether to exclude the unexplained variance percentage.
        exclude_sliced_wasserstein2 (bool): Whether to exclude the sliced Wasserstein-2 distance.
        exclude_kde_kl_divergence (bool): Whether to exclude the KDE KL divergence.
        exclude_kde_l1_divergence (bool): Whether to exclude the KDE L1 divergence.
        verbose (bool): Whether to print verbose output.
    """
    dataset: Dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )
    pushforward_operator = load_pushforward_operator_from_experiment(experiment)
    pushforward_operator.to(**experiment.tensor_parameters)

    metrics = {"quantile": {}, "inverse_quantile": {}}

    quantile_metrics = {
        "wasserstein2": [],
        "unexplained_variance_percentage": [],
        "monotonicity_violations": [],
        "sliced_wasserstein2": [],
        "kde_kl_divergence": [],
        "kde_l1_divergence": []
    }
    inverse_quantile_metrics = {
        "wasserstein2": [],
        "unexplained_variance_percentage": [],
        "monotonicity_violations": [],
        "sliced_wasserstein2": [],
        "kde_kl_divergence": [],
        "kde_l1_divergence": []
    }

    random_number_generator = torch.Generator(
        device=experiment.tensor_parameters["device"]
    )
    random_number_generator.manual_seed(42)

    for i in tqdm(range(100), desc="Running tests", disable=not verbose):
        X_tensor, Y_tensor, U_tensor = dataset.sample_x_y_u(n_points=1000)
        Y_approximation = pushforward_operator.push_u_given_x(U_tensor, X_tensor)
        U_approximation = pushforward_operator.push_y_given_x(Y_tensor, X_tensor)

        if not exclude_wasserstein2:
            quantile_metrics["wasserstein2"].append(
                wassertein2(Y_tensor, Y_approximation)
            )
            inverse_quantile_metrics["wasserstein2"].append(
                wassertein2(U_tensor, U_approximation)
            )
        if not exclude_sliced_wasserstein2:
            quantile_metrics["sliced_wasserstein2"].append(
                sliced_wasserstein2(Y_tensor, Y_approximation)
            )
            inverse_quantile_metrics["sliced_wasserstein2"].append(
                sliced_wasserstein2(U_tensor, U_approximation)
            )
        if not exclude_unexplained_variance_percentage:
            quantile_metrics["unexplained_variance_percentage"].append(
                percentage_of_unexplained_variance(Y_tensor, Y_approximation)
            )
            inverse_quantile_metrics["unexplained_variance_percentage"].append(
                percentage_of_unexplained_variance(U_tensor, U_approximation)
            )

        if not exclude_kde_kl_divergence or not exclude_kde_l1_divergence:
            _, Y_sample, U_sample = dataset.sample_x_y_u(n_points=1000)

            if not exclude_kde_kl_divergence:
                quantile_metrics["kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                inverse_quantile_metrics["kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )
            if not exclude_kde_l1_divergence:
                quantile_metrics["kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                inverse_quantile_metrics["kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )

    metrics["quantile"] = quantile_metrics
    metrics["inverse_quantile"] = inverse_quantile_metrics

    return metrics


def test(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    exclude_monotonicity_violations: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
    verbose: bool = False
) -> dict:
    dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )

    if type(dataset) in {
        BananaDataset, TicTacDataset, QuadraticPotentialConvexBananaDataset,
        NotConditionalBananaDataset, FNLVQR_MVN, PICNN_FNLVQR_Glasses,
        PICNN_FNLVQR_Star, PICNN_FNLVQR_Banana
    }:
        return test_on_synthetic_dataset(
            experiment, exclude_wasserstein2, exclude_unexplained_variance_percentage,
            exclude_monotonicity_violations, exclude_sliced_wasserstein2,
            exclude_kde_kl_divergence, exclude_kde_l1_divergence, verbose
        )
    else:
        raise NotImplementedError(
            f"Testing on the dataset {dataset.__class__.__name__} is not implemented."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_experiment_file", type=str, required=True)
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument(
        "--exclude-monotonicity-violations",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--exclude-sliced-wasserstein2",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--exclude-wasserstein2", action="store_true", required=False, default=False
    )
    parser.add_argument(
        "--exclude-unexplained-variance",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--exclude-kde-kl-divergence",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--exclude-kde-l1-divergence",
        action="store_true",
        required=False,
        default=False
    )

    args = parser.parse_args()

    test_from_json_file(
        args.path_to_experiment_file, args.verbose, args.exclude_wasserstein2,
        args.exclude_unexplained_variance_percentage,
        args.exclude_monotonicity_violations, args.exclude_sliced_wasserstein2,
        args.exclude_kde_kl_divergence, args.exclude_kde_l1_divergence
    )
