import torch
from tqdm import tqdm
from datasets import Dataset, BananaDataset, TicTacDataset
from infrastructure.classes import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map
from metrics import wassertein2, compare_quantile_in_latent_space, compute_gaussian_negative_log_likelihood
from pushforward_operators import PushForwardOperator

def test_from_json_file(path_to_experiment_file: str, verbose: bool = False, exclude_wasserstein2: bool = False, exclude_gaussian_likelihood: bool = False, exclude_quantile_similarity: bool = False) -> dict:
    """
    Test a model on a synthetic dataset from an experiment set in a JSON file.

    Args:
        path_to_experiment_file (str): The path to the JSON file containing the experiment description.

    Returns:
        dict: The metrics.
    """
    try:
        with open(path_to_experiment_file, "r") as f:
            experiment_as_json = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path_to_experiment_file} not found. Make sure the file exists and path is correct.")

    try:
        experiment = Experiment.model_validate_json(experiment_as_json)
    except Exception as e:
        raise ValueError(f"Error loading experiment from {path_to_experiment_file}: {e}. Make sure the file is a valid JSON file and is consistent with the Experiment class.")

    metrics = test(experiment, verbose=verbose, exclude_wasserstein2=exclude_wasserstein2, exclude_gaussian_likelihood=exclude_gaussian_likelihood, exclude_quantile_similarity=exclude_quantile_similarity)

    if experiment.path_to_metrics is not None:
        torch.save(metrics, experiment.path_to_metrics)

    return metrics

def load_pushforward_operator_from_experiment(experiment: Experiment) -> PushForwardOperator:
    """
    Load a pushforward operator from an experiment.

    Args:
        experiment (Experiment): The experiment to load the pushforward operator from.

    Returns:
        PushForwardOperator: The loaded pushforward operator.
    """
    pushforward_operator = name_to_pushforward_operator_map[experiment.pushforward_operator_name](**experiment.pushforward_operator_parameters)
    pushforward_operator.to(**experiment.tensor_parameteres)

    if experiment.path_to_weights is not None:
        pushforward_operator.load(experiment.path_to_weights, map_location=torch.device(experiment.tensor_parameteres["device"]))
    else:
        raise ValueError("Path to the model is not specified. Model can not be loaded.")

    pushforward_operator.eval()
    return pushforward_operator

def sample_wasserstein2_metrics(
        pushforward_operator: PushForwardOperator,
        X_dataset: torch.Tensor,
        Y_dataset: torch.Tensor,
        number_of_samples: int,
        verbose: bool = False
    ) -> torch.Tensor:
    number_of_covariates = X_dataset.shape[0]
    wasserstein2_progress_bar = tqdm(
        range(number_of_covariates),
        desc="Computing Wasserstein-2 metrics",
        disable=not verbose
    )
    wasserstein2_metrics = []

    for i in wasserstein2_progress_bar:
        metrics_per_x = []

        for _ in range(number_of_samples):
            Y_batch = Y_dataset[i, :, :]
            X_batch = X_dataset[i:i+1, :].repeat(Y_batch.shape[0], 1)
            U_batch = torch.randn_like(Y_batch)

            Y_approximation = pushforward_operator.push_forward_u_given_x(U=U_batch, X=X_batch)
            metrics_per_x.append(wassertein2(Y_batch, Y_approximation))
        wasserstein2_metrics.append(torch.tensor(metrics_per_x))

    return torch.stack(wasserstein2_metrics)

def sample_gaussian_likelihood_metrics(
        pushforward_operator: PushForwardOperator,
        dataset: Dataset,
        X_dataset: torch.Tensor,
        Y_dataset: torch.Tensor,
        number_of_samples: int,
        verbose: bool = False
    ) -> torch.Tensor:
    number_of_covariates = X_dataset.shape[0]
    likelihood_metrics = []
    gaussian_likelihood_progress_bar = tqdm(range(number_of_covariates), desc="Computing Gaussian Likelihood metrics", disable=not verbose)
    for i in gaussian_likelihood_progress_bar:
        metrics_per_x = []
        for _ in range(number_of_samples):
            Y_batch = Y_dataset[i, :, :]
            X_batch = X_dataset[i:i+1, :].repeat(Y_batch.shape[0], 1)
            U_batch = torch.randn_like(Y_batch)

            Y_approximation = pushforward_operator.push_forward_u_given_x(U=U_batch, X=X_batch)
            U_approximation = dataset.pushbackward_Y_given_X(Y=Y_approximation, X=X_batch)
            metrics_per_x.append(compute_gaussian_negative_log_likelihood(U_approximation))

        likelihood_metrics.append(torch.stack(metrics_per_x))
    return torch.stack(likelihood_metrics)


def sample_quantile_similarity_metrics(
        pushforward_operator: PushForwardOperator,
        dataset: Dataset,
        X_dataset: torch.Tensor,
        Y_dataset: torch.Tensor,
        number_of_samples: int,
        number_of_alphas: int = 10,
        verbose: bool = False
    ) -> torch.Tensor:
    number_of_covariates = X_dataset.shape[0]
    alpha_metrics = []
    quantile_similarity_progress_bar = tqdm(
        range(number_of_covariates),
        desc="Computing Quantile Similarity metrics",
        disable=not verbose
    )
    alphas = torch.linspace(0.05, 0.95, number_of_alphas)
    for i in quantile_similarity_progress_bar:
        metrics_per_x = []
        for alpha in alphas:
            metrics_per_alpha = []
            for _ in range(number_of_samples):
                Y_batch = Y_dataset[i, :, :]
                X_batch = X_dataset[i:i+1, :].repeat(Y_batch.shape[0], 1)
                U_batch = torch.randn_like(Y_batch)

                Y_approximation = pushforward_operator.push_forward_u_given_x(U=U_batch, X=X_batch)
                U_approximation = dataset.pushbackward_Y_given_X(Y=Y_approximation, X=X_batch)
                metrics_per_alpha.append(compare_quantile_in_latent_space(U_approximation, alpha, "gaussian"))

            metrics_per_x.append(torch.stack(metrics_per_alpha))
        alpha_metrics.append(torch.stack(metrics_per_x))
    return torch.stack(alpha_metrics)

def test_on_synthetic_dataset(experiment: Experiment, exclude_wasserstein2: bool = False, exclude_gaussian_likelihood: bool = False, exclude_quantile_similarity: bool = False, verbose: bool = False) -> dict:
    """
    Test a model on a synthetic dataset.
    """
    dataset = name_to_dataset_map[experiment.dataset_name](experiment.tensor_parameteres, **experiment.dataset_parameters)
    number_of_covariates_per_dimension = 10
    pushforward_operator = load_pushforward_operator_from_experiment(experiment)
    pushforward_operator.to(**experiment.tensor_parameteres)

    metrics = {}

    X_dataset = dataset.meshgrid_of_covariates(n_points_per_dimension=number_of_covariates_per_dimension)
    X_dataset = X_dataset.to(**experiment.tensor_parameteres)
    Y_dataset = dataset.sample_conditional(n_points=2048, X=X_dataset)
    Y_dataset = Y_dataset.to(**experiment.tensor_parameteres)

    try:
        if not exclude_gaussian_likelihood:
            metrics["gaussian_likelihood"] = sample_gaussian_likelihood_metrics(
                pushforward_operator=pushforward_operator,
                dataset=dataset,
                X_dataset=X_dataset,
                Y_dataset=Y_dataset,
                number_of_samples=10,
                verbose=verbose
            )
    except NotImplementedError:
        print("Pushbackward of u is not implemented for this dataset. Skipping Gaussian Likelihood metrics.")


    if not exclude_wasserstein2:
        metrics["wasserstein2"] = sample_wasserstein2_metrics(
            pushforward_operator=pushforward_operator,
            X_dataset=X_dataset,
            Y_dataset=Y_dataset,
            number_of_samples=10,
            verbose=verbose
        )

    try:
        if not exclude_quantile_similarity:
            metrics["quantile_similarity"] = sample_quantile_similarity_metrics(
                pushforward_operator=pushforward_operator,
                dataset=dataset,
            X_dataset=X_dataset,
            Y_dataset=Y_dataset,
            number_of_samples=10,
            number_of_alphas=10,
            verbose=verbose
        )
    except NotImplementedError:
        print("Pushbackward of u is not implemented for this dataset. Skipping Quantile Similarity metrics.")

    return metrics

def test(experiment: Experiment, exclude_wasserstein2: bool = False, exclude_gaussian_likelihood: bool = False, exclude_quantile_similarity: bool = False, verbose: bool = False) -> dict:
    """
    Test a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to train.

    Returns:
        dict: The metrics.
    """
    dataset = name_to_dataset_map[experiment.dataset_name](**experiment.dataset_parameters)

    if type(dataset) in {BananaDataset, TicTacDataset}:
        return test_on_synthetic_dataset(experiment, exclude_wasserstein2, exclude_gaussian_likelihood, exclude_quantile_similarity, verbose)
    else:
        raise NotImplementedError(f"Testing on the dataset {dataset.__class__.__name__} is not implemented.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_experiment_file", type=str, required=True)
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument("--exclude-wasserstein2", action="store_true", required=False, default=False)
    parser.add_argument("--exclude-gaussian-likelihood", type=bool, required=False, default=False)
    parser.add_argument("--exclude-quantile-similarity", type=bool, required=False, default=False)
    args = parser.parse_args()

    test_from_json_file(args.path_to_experiment_file, args.verbose, args.exclude_wasserstein2, args.exclude_gaussian_likelihood, args.exclude_quantile_similarity)
