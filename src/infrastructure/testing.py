import torch
from tqdm import tqdm
from datasets import Dataset, BananaDataset, TicTacDataset, StarDataset, ConvexBananaDataset
from infrastructure.classes import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map
from metrics import wassertein2, compute_hausdorff_distance
from pushforward_operators import PushForwardOperator
from torch.distributions import multivariate_normal
from scipy import stats


def test_from_json_file(
    path_to_experiment_file: str,
    verbose: bool = False,
    exclude_wasserstein2: bool = False,
    exclude_l2_distance: bool = False,
    exclude_quantile_similarity: bool = False
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
        exclude_l2_distance=exclude_l2_distance,
        exclude_quantile_similarity=exclude_quantile_similarity
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
    X_dataset: torch.Tensor,
    Y_dataset: torch.Tensor,
    number_of_samples: int,
    verbose: bool = False
) -> torch.Tensor:

    wasserstein2_progress_bar = tqdm(
        range(X_dataset.shape[0]),
        desc="Computing Inverse Quantile Wasserstein-2 metrics",
        disable=not verbose
    )
    wasserstein2_metrics = []

    for i in wasserstein2_progress_bar:
        metrics_per_x = []

        for _ in range(number_of_samples):
            Y_batch = Y_dataset[i, :, :]
            X_batch = X_dataset[i, :, :]
            U_batch = torch.randn_like(Y_batch)
            U_approximation = pushforward_operator.push_y_given_x(y=Y_batch, x=X_batch)
            metrics_per_x.append(wassertein2(U_batch, U_approximation))

        wasserstein2_metrics.append(torch.tensor(metrics_per_x))

    return torch.stack(wasserstein2_metrics)


def sample_quantile_wasserstein2_distance(
    pushforward_operator: PushForwardOperator,
    X_dataset: torch.Tensor,
    Y_dataset: torch.Tensor,
    number_of_samples: int,
    verbose: bool = False
) -> torch.Tensor:

    wasserstein2_progress_bar = tqdm(
        range(X_dataset.shape[0]),
        desc="Computing Quantile Wasserstein-2 metrics",
        disable=not verbose
    )
    wasserstein2_metrics = []

    for i in wasserstein2_progress_bar:
        metrics_per_x = []

        for _ in range(number_of_samples):
            Y_batch = Y_dataset[i, :, :]
            X_batch = X_dataset[i, :, :]
            U_batch = torch.randn_like(Y_batch)
            Y_approximation = pushforward_operator.push_u_given_x(u=U_batch, x=X_batch)
            metrics_per_x.append(wassertein2(Y_batch, Y_approximation))

        wasserstein2_metrics.append(torch.tensor(metrics_per_x))

    return torch.stack(wasserstein2_metrics)


def sample_inverse_quantile_l2_distance(
    pushforward_operator: PushForwardOperator,
    X_dataset: torch.Tensor,
    Y_dataset: torch.Tensor,
    U_dataset: torch.Tensor,
    number_of_samples: int,
    verbose: bool = False
) -> torch.Tensor:

    l2_progress_bar = tqdm(
        range(X_dataset.shape[0]),
        desc="Computing Inverse Quantile l2 distances",
        disable=not verbose
    )
    l2_metrics = []

    for i in l2_progress_bar:
        metrics_per_x = []
        for _ in range(number_of_samples):
            Y_batch = Y_dataset[i, :, :]
            X_batch = X_dataset[i, :, :]
            U_batch = U_dataset[i, :, :]

            U_approximation = pushforward_operator.push_y_given_x(y=Y_batch, x=X_batch)
            l2_distance = torch.norm((U_approximation - U_batch), dim=-1)**2
            metrics_per_x.append(l2_distance.mean())

        l2_metrics.append(torch.stack(metrics_per_x))

    return torch.stack(l2_metrics)


def sample_quantile_l2_distance(
    pushforward_operator: PushForwardOperator,
    X_dataset: torch.Tensor,
    Y_dataset: torch.Tensor,
    U_dataset: torch.Tensor,
    number_of_samples: int,
    verbose: bool = False
) -> torch.Tensor:

    l2_progress_bar = tqdm(
        range(X_dataset.shape[0]),
        desc="Computing Quantile l2 distances",
        disable=not verbose
    )
    l2_metrics = []

    for i in l2_progress_bar:
        metrics_per_x = []
        for _ in range(number_of_samples):
            Y_batch = Y_dataset[i, :, :]
            X_batch = X_dataset[i, :, :]
            U_batch = U_dataset[i, :, :]

            Y_approximation = pushforward_operator.push_u_given_x(u=U_batch, x=X_batch)
            l2_distance = torch.norm((Y_approximation - Y_batch), dim=-1)**2
            metrics_per_x.append(l2_distance.mean())

        l2_metrics.append(torch.stack(metrics_per_x))

    return torch.stack(l2_metrics)


def test_on_synthetic_dataset(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_l2_distance: bool = False,
    exclude_quantile_similarity: bool = False,
    verbose: bool = False
) -> dict:
    """
    Test a model on a synthetic dataset.
    """
    dataset: Dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )
    number_of_covariates_per_dimension = 10
    pushforward_operator = load_pushforward_operator_from_experiment(experiment)
    pushforward_operator.to(**experiment.tensor_parameters)

    metrics = {"quantile": {}, "inverse_quantile": {}}

    random_number_generator = torch.Generator(
        device=experiment.tensor_parameters["device"]
    )
    random_number_generator.manual_seed(42)

    _, _Y = dataset.sample_joint(1)
    X = dataset.meshgrid_of_covariates(number_of_covariates_per_dimension)

    U_dataset = torch.randn(
        number_of_covariates_per_dimension,
        2000,
        _Y.shape[-1],
        generator=random_number_generator,
        **experiment.tensor_parameters
    )
    X_dataset = X.unsqueeze(1).repeat(1, 2000, 1).to(**experiment.tensor_parameters)
    Y_dataset = dataset.push_u_given_x(u=U_dataset, x=X_dataset)

    try:
        if not exclude_l2_distance:
            metrics["inverse_quantile"]["l2_distance"
                                        ] = sample_inverse_quantile_l2_distance(
                                            pushforward_operator=pushforward_operator,
                                            X_dataset=X_dataset,
                                            Y_dataset=Y_dataset,
                                            U_dataset=U_dataset,
                                            number_of_samples=1,
                                            verbose=verbose
                                        )
    except NotImplementedError:
        print(
            "Pushbackward of u is not implemented for this dataset. Skipping Gaussian Likelihood metrics."
        )

    try:
        if not exclude_wasserstein2:
            metrics["inverse_quantile"][
                "wasserstein2"] = sample_inverse_quantile_wasserstein2_distance(
                    pushforward_operator=pushforward_operator,
                    X_dataset=X_dataset,
                    Y_dataset=Y_dataset,
                    number_of_samples=1,
                    verbose=verbose
                )
    except NotImplementedError:
        print(
            "Pushbackward of u is not implemented for this dataset. Skipping Gaussian Likelihood metrics."
        )

    try:
        if not exclude_l2_distance:
            metrics["quantile"]["l2_distance"] = sample_quantile_l2_distance(
                pushforward_operator=pushforward_operator,
                X_dataset=X_dataset,
                Y_dataset=Y_dataset,
                U_dataset=U_dataset,
                number_of_samples=1,
                verbose=verbose
            )
    except NotImplementedError:
        print(
            "Pushbackward of u is not implemented for this dataset. Skipping Gaussian Likelihood metrics."
        )

    try:
        if not exclude_wasserstein2:
            metrics["quantile"]["wasserstein2"] = sample_quantile_wasserstein2_distance(
                pushforward_operator=pushforward_operator,
                X_dataset=X_dataset,
                Y_dataset=Y_dataset,
                number_of_samples=1,
                verbose=verbose
            )
    except NotImplementedError:
        print(
            "Pushbackward of u is not implemented for this dataset. Skipping Gaussian Likelihood metrics."
        )

    return metrics


def test(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_l2_distance: bool = False,
    exclude_quantile_similarity: bool = False,
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
            experiment, exclude_wasserstein2, exclude_l2_distance,
            exclude_quantile_similarity, verbose
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
        "--exclude-l2-distance", action="store_true", required=False, default=False
    )
    parser.add_argument(
        "--exclude-quantile-similarity",
        action="store_true",
        required=False,
        default=False
    )
    args = parser.parse_args()

    test_from_json_file(
        args.path_to_experiment_file, args.verbose, args.exclude_wasserstein2,
        args.exclude_l2_distance, args.exclude_quantile_similarity
    )
