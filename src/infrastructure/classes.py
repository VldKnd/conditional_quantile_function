import pydantic
import torch
import os


class TrainParameters(pydantic.BaseModel):
    number_of_epochs_to_train: int = pydantic.Field(default=500)
    optimizer_parameters: dict = pydantic.Field(default={})
    scheduler_parameters: dict = pydantic.Field(default={})
    verbose: bool = pydantic.Field(default=False)


class TensorParameters(pydantic.BaseModel):
    dtype: torch.dtype | str = pydantic.Field(default=torch.float64)
    device: torch.device | str = pydantic.Field(default=torch.device("cpu"))
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, *args, **kwargs):
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    @pydantic.field_serializer('device')
    def serialize_timestamp(self, device: torch.device | str) -> str:
        return str(device)


class Experiment(pydantic.BaseModel):
    dataset_name: str
    dataset_number_of_points: int = pydantic.Field(default=1000)
    dataloader_parameters: dict = pydantic.Field(default={})
    dataset_parameters: dict = pydantic.Field(default={})
    pushforward_operator_name: str
    pushforward_operator_parameters: dict = pydantic.Field(default={})
    train_parameters: TrainParameters = pydantic.Field(default=TrainParameters())
    path_to_experiment_file: str | None = pydantic.Field(default=None)

    tensor_parameters_raw: TensorParameters = pydantic.Field(
        default=TensorParameters(),
        validation_alias="tensor_parameters",
        serialization_alias="tensor_parameters"
    )
    path_to_weights_raw: str | None = pydantic.Field(
        default=None,
        validation_alias="path_to_weights",
        serialization_alias="path_to_weights"
    )
    path_to_metrics_raw: str | None = pydantic.Field(
        default=None,
        validation_alias="path_to_metrics",
        serialization_alias="path_to_metrics"
    )

    @property
    def path_to_weights(self) -> str | None:
        if self.path_to_experiment_file is not None and self.path_to_weights_raw is not None:
            return os.path.join(self.path_to_experiment_file, self.path_to_weights_raw)
        else:
            return self.path_to_weights_raw

    @property
    def path_to_metrics(self) -> str | None:
        if self.path_to_experiment_file is not None and self.path_to_metrics_raw is not None:
            return os.path.join(self.path_to_experiment_file, self.path_to_metrics_raw)
        else:
            return self.path_to_metrics_raw

    @property
    def tensor_parameters(self) -> dict:
        return self.tensor_parameters_raw.model_dump()

    @classmethod
    def load_from_path_to_experiment_file(cls, path_to_experiment_file: str):
        try:
            with open(path_to_experiment_file, "r") as f:
                experiment_as_json = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {path_to_experiment_file} not found. Make sure the file exists and path is correct."
            )

        try:
            experiment = cls.model_validate_json(experiment_as_json)
            experiment.path_to_experiment_file = os.path.dirname(
                path_to_experiment_file
            )
        except Exception as e:
            raise ValueError(
                f"Error loading experiment from {path_to_experiment_file}: {e}. Make sure the file is a valid JSON file and is consistent with the Experiment class."
            )

        return experiment
