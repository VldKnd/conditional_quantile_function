from protocols.pushforward_operator import PushForwardOperator
from infrastructure.dataclasses import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map
import torch

def train(experiment: Experiment) -> PushForwardOperator:
    """
    Train a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to train.
        path_to_save_model (str | None): The path to save the model.

    Returns:
        PushForwardOperator: The trained model.
    """
    dataset = name_to_dataset_map[experiment.dataset_name](**experiment.dataset_parameters)
    pushforward_operator = name_to_pushforward_operator_map[experiment.pushforward_operator_name](**experiment.pushforward_operator_parameters)
    X_dataset, Y_dataset = dataset.sample_joint(n_points=experiment.dataset_number_of_points)
    X_dataset = X_dataset.to(**experiment.tensor_parameteres)
    Y_dataset = Y_dataset.to(**experiment.tensor_parameteres)
    dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_dataset, Y_dataset),
        **experiment.dataloader_parameters
    )
    pushforward_operator.to(**experiment.tensor_parameteres)

    pushforward_operator.train()
    _ = pushforward_operator.fit(dataloader, train_parameters=experiment.train_parameters)
    if experiment.path_to_result is not None:
        pushforward_operator.save(experiment.path_to_result)
    return pushforward_operator