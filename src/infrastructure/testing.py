import torch
from tqdm import tqdm
from datasets import Dataset, BananaDataset, TicTacDataset, StarDataset, ConvexBananaDataset
from infrastructure.classes import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map
from metrics import wassertein2, unexplained_variance_percentage
from pushforward_operators import PushForwardOperator


def test_from_json_file(
    path_to_experiment_file: str,
    verbose: bool = False,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
) -> dict:
    """
    Test a model on a synthetic dataset from an experiment set in a JSON file.

    Args:
        path_to_experiment_file (str): The path to the JSON file containing the experiment description.

    Returns:
        dict: The metrics.
    """
    experiment = Experiment.load_from_path_to_experiment_file(
        path_to_experiment_file=path_to_experiment_file
    )
    metrics = test(
        experiment,
        verbose=verbose,
        exclude_wasserstein2=exclude_wasserstein2,
        exclude_unexplained_variance_percentage=exclude_unexplained_variance_percentage
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


def sample_inverse_quantile_wasserstein2_distance(
    pushforward_operator: PushForwardOperator,
    dataset: Dataset,
    number_of_samples: int,
    number_of_points_per_estimate: int,
    verbose: bool = False
) -> torch.Tensor:

    wasserstein2_progress_bar = tqdm(
        range(number_of_samples),
        desc="Computing Inverse Quantile Wasserstein-2 metrics",
        disable=not verbose
    )
    wasserstein2_metrics = []

    for _ in wasserstein2_progress_bar:
        X_dataset, Y_dataset, U_dataset = dataset.sample_x_y_u(
            n_points=number_of_points_per_estimate
        )
        U_approximation = pushforward_operator.push_y_given_x(y=Y_dataset, x=X_dataset)
        wasserstein2_metrics.append(wassertein2(U_dataset, U_approximation))

    return torch.stack(wasserstein2_metrics)


def sample_quantile_wasserstein2_distance(
    pushforward_operator: PushForwardOperator,
    dataset: Dataset,
    number_of_samples: int,
    number_of_points_per_estimate: int,
    verbose: bool = False
) -> torch.Tensor:

    wasserstein2_progress_bar = tqdm(
        range(number_of_samples),
        desc="Computing Quantile Wasserstein-2 metrics",
        disable=not verbose
    )
    wasserstein2_metrics = []

    for _ in wasserstein2_progress_bar:
        X_dataset, Y_dataset, U_dataset = dataset.sample_x_y_u(
            n_points=number_of_points_per_estimate
        )
        Y_approximation = pushforward_operator.push_u_given_x(u=U_dataset, x=X_dataset)
        wasserstein2_metrics.append(wassertein2(Y_dataset, Y_approximation))

    return torch.stack(wasserstein2_metrics)


def sample_quantile_unexplained_variance_percentage(
    pushforward_operator: PushForwardOperator,
    dataset: Dataset,
    number_of_samples: int,
    number_of_points_per_estimate: int,
    verbose: bool = False
) -> torch.Tensor:

    unexplained_variance_percentage_progress_bar = tqdm(
        range(number_of_samples),
        desc="Computing Quantile unexplained variance percentage",
        disable=not verbose
    )
    unexplained_variance_percentage_metrics = []

    for _ in unexplained_variance_percentage_progress_bar:
        X_dataset, Y_dataset, U_dataset = dataset.sample_x_y_u(
            n_points=number_of_points_per_estimate
        )
        Y_approximation = pushforward_operator.push_u_given_x(u=U_dataset, x=X_dataset)
        unexplained_variance_percentage_metric = unexplained_variance_percentage(
            Y_dataset, Y_approximation
        )
        unexplained_variance_percentage_metrics.append(
            unexplained_variance_percentage_metric
        )

    return torch.stack(unexplained_variance_percentage_metrics)


def sample_inverse_quantile_unexplained_variance_percentage(
    pushforward_operator: PushForwardOperator,
    dataset: Dataset,
    number_of_samples: int,
    number_of_points_per_estimate: int,
    verbose: bool = False
) -> torch.Tensor:

    unexplained_variance_percentage_progress_bar = tqdm(
        range(number_of_samples),
        desc="Computing Inverse Quantile unexplained variance percentage",
        disable=not verbose
    )
    unexplained_variance_percentage_metrics = []

    for _ in unexplained_variance_percentage_progress_bar:
        X_dataset, Y_dataset, U_dataset = dataset.sample_x_y_u(
            n_points=number_of_points_per_estimate
        )
        U_approximation = pushforward_operator.push_y_given_x(y=Y_dataset, x=X_dataset)
        unexplained_variance_percentage_metric = unexplained_variance_percentage(
            U_dataset, U_approximation
        )
        unexplained_variance_percentage_metrics.append(
            unexplained_variance_percentage_metric
        )

    return torch.stack(unexplained_variance_percentage_metrics)


def test_on_synthetic_dataset(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    verbose: bool = False
) -> dict:
    """
    Test a model on a synthetic dataset.
    """
    dataset: Dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )
    pushforward_operator = load_pushforward_operator_from_experiment(experiment)
    pushforward_operator.to(**experiment.tensor_parameters)

    metrics = {"quantile": {}, "inverse_quantile": {}}

    quantile_metrics = {}
    inverse_quantile_metrics = {}

    random_number_generator = torch.Generator(
        device=experiment.tensor_parameters["device"]
    )
    random_number_generator.manual_seed(42)

    try:
        if not exclude_unexplained_variance_percentage:
            inverse_quantile_metrics["unexplained_variance_percentage"] \
            = sample_inverse_quantile_unexplained_variance_percentage(
                pushforward_operator=pushforward_operator,
                dataset=dataset,
                number_of_samples=1,
                verbose=verbose
            )
    except NotImplementedError:
        print("Skipping inverse quantile unexplained variance percentage metrics.")

    try:
        if not exclude_wasserstein2:
            inverse_quantile_metrics["wasserstein2"] \
            = sample_inverse_quantile_wasserstein2_distance(
                pushforward_operator=pushforward_operator,
                dataset=dataset,
                number_of_samples=1,
                verbose=verbose
            )
    except NotImplementedError:
        print("Skipping inverse quantile Wasserstein-2 distance metrics.")

    try:
        if not exclude_unexplained_variance_percentage:
            quantile_metrics["unexplained_variance_percentage"
                             ] = sample_quantile_unexplained_variance_percentage(
                                 pushforward_operator=pushforward_operator,
                                 dataset=dataset,
                                 number_of_samples=1,
                                 verbose=verbose
                             )
    except NotImplementedError:
        print("Skipping quantile unexplained variance percentage metrics.")

    try:
        if not exclude_wasserstein2:
            quantile_metrics["wasserstein2"] = sample_quantile_wasserstein2_distance(
                pushforward_operator=pushforward_operator,
                dataset=dataset,
                number_of_samples=1,
                verbose=verbose
            )
    except NotImplementedError:
        print("Skipping quantile Wasserstein-2 distance metrics.")

    metrics["quantile"] = quantile_metrics
    metrics["inverse_quantile"] = inverse_quantile_metrics

    return metrics


def test(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    verbose: bool = False
) -> dict:
    """
    Test a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to train.

    Returns:
        dict: The metrics.
    """
    dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )

    if type(dataset) in {
        BananaDataset, TicTacDataset, StarDataset, ConvexBananaDataset
    }:
        return test_on_synthetic_dataset(
            experiment, exclude_wasserstein2, exclude_unexplained_variance_percentage,
            verbose
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
        "--exclude-wasserstein2", action="store_true", required=False, default=False
    )
    parser.add_argument(
        "--exclude-unexplained-variance-percentage",
        action="store_true",
        required=False,
        default=False
    )
    args = parser.parse_args()

    test_from_json_file(
        args.path_to_experiment_file, args.verbose, args.exclude_wasserstein2,
        args.exclude_unexplained_variance_percentage
    )
