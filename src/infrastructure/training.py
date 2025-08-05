import torch
from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map

def train_from_json_file(path_to_experiment_file: str) -> PushForwardOperator:
    """
    Train a model on a synthetic dataset from an experiment set in a JSON file.

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

    return train(experiment=experiment)

def train(experiment: Experiment) -> PushForwardOperator:
    """
    Train a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to train.
        path_to_save_model (str | None): The path to save the model.

    Returns:
        PushForwardOperator: The trained model.
    """
    dataset = name_to_dataset_map[experiment.dataset_name](experiment.tensor_parameters, **experiment.dataset_parameters)
    pushforward_operator = name_to_pushforward_operator_map[experiment.pushforward_operator_name](**experiment.pushforward_operator_parameters)
    X_dataset, Y_dataset = dataset.sample_joint(n_points=experiment.dataset_number_of_points)
    X_dataset = X_dataset.to(**experiment.tensor_parameters)
    Y_dataset = Y_dataset.to(**experiment.tensor_parameters)
    dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_dataset, Y_dataset),
        **experiment.dataloader_parameters
    )

    try:
        pushforward_operator.to(**experiment.tensor_parameters)
        pushforward_operator.train()
    except AttributeError:
        pass

    _ = pushforward_operator.fit(dataloader, train_parameters=experiment.train_parameters)

    if experiment.path_to_weights is not None:
        pushforward_operator.save(experiment.path_to_weights)

    return pushforward_operator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_experiment_file", type=str, required=True)
    args = parser.parse_args()
    train_from_json_file(args.path_to_experiment_file)
