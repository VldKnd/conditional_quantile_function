from protocols.pushforward_operator import PushForwardOperator
from protocols.dataset import Dataset
import torch
from typing import Dict, Optional, Tuple, List
from metrics.wasserstein2 import wassertein2

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