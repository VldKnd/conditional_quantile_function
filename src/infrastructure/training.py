from protocols.pushforward_operator import PushForwardOperator
from utils import TrainParams
from protocols.dataset import Dataset
import torch
from typing import Dict, Optional

def train_model_on_synthetic_data(model: PushForwardOperator, dataset: Dataset, train_params: TrainParams, device_and_dtype_specifications: Dict, path_to_save_model: Optional[str] = None) -> PushForwardOperator:
    """
    Train a model on a synthetic dataset. If path_to_save_model is provided, the trained model is saved to the path.

    Args:
        model (PushForwardOperator): The model to train.
        dataset (Dataset): The dataset to train on.
        train_params (TrainParams): The training parameters.
        device_and_dtype_specifications (Dict): The device and dtype specifications to use for the training.
        path_to_save_model (Optional[str], optional): The path to save the model. Defaults to None.

    Returns:
        PushForwardOperator: The trained model.
    """
    X_dataset, Y_dataset = dataset.sample_joint(n_points=1000)

    X_dataset = X_dataset.to(**device_and_dtype_specifications)
    Y_dataset = Y_dataset.to(**device_and_dtype_specifications)

    dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_dataset, Y_dataset),
        batch_size=256,
        shuffle=True
    )

    model.to(**device_and_dtype_specifications)
    _ = model.fit(dataloader, train_params=train_params)

    if path_to_save_model is not None:
        model.save(path_to_save_model)

    return model