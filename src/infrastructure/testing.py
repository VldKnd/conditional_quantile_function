from protocols.pushforward_operator import PushForwardOperator
from protocols.dataset import Dataset
import torch
from typing import Dict, Optional
from metrics.wasserstein2 import wassertein2

def benchmark_wasserstein2_on_synthetic_data(model: PushForwardOperator, dataset: Dataset, device_and_dtype_specifications: Dict, path_to_load_model: Optional[str] = None) -> float:
    """
    Benchmark the Wasserstein-2 distance between the ground truth and the model's pushforward of a random sample of the dataset.

    Args:
        model (PushForwardOperator): The model to benchmark.
        dataset (Dataset): The dataset to use for the benchmark.
        device_and_dtype_specifications (Dict): The device and dtype specifications to use for the benchmark.
        path_to_load_model (Optional[str], optional): The path to the model to load. Defaults to None.

    Returns:
        float: The Wasserstein-2 distance between the ground truth and the model's pushforward of a random sample of the dataset.
    """
    X_dataset, Y_dataset = dataset.sample_joint(n_points=1000)
    U_dataset = torch.randn_like(Y_dataset)

    X_dataset = X_dataset.to(**device_and_dtype_specifications)
    Y_dataset = Y_dataset.to(**device_and_dtype_specifications)
    U_dataset = U_dataset.to(**device_and_dtype_specifications)
    model.to(**device_and_dtype_specifications)

    if path_to_load_model is not None:
        model.load(path_to_load_model)

    metric_value = wassertein2(
        ground_truth=Y_dataset,
        approximation=model.push_forward_u_given_x(
            U=U_dataset,
            X=X_dataset
        )
    )

    return metric_value