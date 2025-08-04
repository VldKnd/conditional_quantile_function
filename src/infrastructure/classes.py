import pydantic
import torch

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

class Experiment(pydantic.BaseModel):
    dataset_name: str
    dataset_number_of_points: int = pydantic.Field(default=1000)
    dataloader_parameters: dict = pydantic.Field(default={})
    dataset_parameters: dict = pydantic.Field(default={})
    pushforward_operator_name: str
    pushforward_operator_parameters: dict = pydantic.Field(default={})
    train_parameters: TrainParameters = pydantic.Field(default=TrainParameters())
    tensor_parameteres: TensorParameters = pydantic.Field(default=TensorParameters())
    path_to_result: str | None = pydantic.Field(default=None)

    def __getattribute__(self, name):
        if name == "tensor_parameteres":
            return object.__getattribute__(self, 'tensor_parameteres').model_dump()
        else:
            return object.__getattribute__(self, name)
