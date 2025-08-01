from pushforward_operators.protocol import PushForwardOperator
from datasets.protocol import Dataset
import torch
from typing import Dict, Optional, Tuple, List
from metrics.wasserstein2 import wassertein2

from infrastructure.dataclasses import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map

def test_from_json_file(path_to_experiment_file: str) -> float:
    """
    Test a model on a synthetic dataset from an experiment set in a JSON file.

    Args:
        path_to_experiment_file (str): The path to the JSON file containing the experiment description.

    Returns:
        PushForwardOperator: The trained model.
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

    return test(experiment)

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

    if experiment.path_to_result is not None:
        pushforward_operator.load(experiment.path_to_result)
    else:
        raise ValueError("Path to the model is not specified. Model can not be loaded.")

    pushforward_operator.eval()
    return pushforward_operator

def test(experiment: Experiment) -> PushForwardOperator:
    """
    Test a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to train.

    Returns:
        PushForwardOperator: The trained model.
    """
    dataset = name_to_dataset_map[experiment.dataset_name](**experiment.dataset_parameters)
    X_dataset, Y_dataset = dataset.sample_joint(n_points=experiment.dataset_number_of_points)
    X_dataset = X_dataset.to(**experiment.tensor_parameteres)
    Y_dataset = Y_dataset.to(**experiment.tensor_parameteres)
    pushforward_operator = load_pushforward_operator_from_experiment(experiment)

    return 0

def benchmark_wasserstein2_on_synthetic_data(model: PushForwardOperator, dataset: Dataset, device_and_dtype_specifications: Dict, path_to_load_model: Optional[str] = None, covariates: Optional[torch.Tensor] = None) -> Tuple[List[float], torch.Tensor]:
    """
    Benchmark the Wasserstein-2 distance between the ground truth and the model's pushforward of a random sample of the dataset.

    Args:
        model (PushForwardOperator): The model to benchmark.
        dataset (Dataset): The dataset to use for the benchmark.
        device_and_dtype_specifications (Dict): The device and dtype specifications to use for the benchmark.
        path_to_load_model (Optional[str], optional): The path to the model to load. Defaults to None.
        covariates (Optional[torch.Tensor], optional): The tensor of covariates to use for the benchmark. Defaults to None.
    Returns:
        Tuple[List[float], torch.Tensor]: The Wasserstein-2 distance between the ground truth and the model's pushforward of a random sample of the dataset.
    """
    if covariates is None:
        covariates, _ = dataset.sample_joint(n_points=1000)

    Y_dataset = dataset.sample_conditional(n_points=covariates.shape[0], X=covariates)
    metric_values = []

    covariates = covariates.to(**device_and_dtype_specifications)
    Y_dataset = Y_dataset.to(**device_and_dtype_specifications)
    model.to(**device_and_dtype_specifications)

    if path_to_load_model is not None:
        model.load(path_to_load_model)

    for i in range(covariates.shape[0]):
        U_dataset = torch.randn_like(Y_dataset[i, :, :])
        metric_value = wassertein2(
            ground_truth=Y_dataset[i, :, :],
            approximation=model.push_forward_u_given_x(
                U=U_dataset,
                X=covariates[i:i+1, :].repeat(U_dataset.shape[0], 1)
            )
        )
        metric_values.append(metric_value)

    return metric_values, covariates