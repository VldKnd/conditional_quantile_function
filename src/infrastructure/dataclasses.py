import pydantic

class TrainParameters(pydantic.BaseModel):
    number_of_epochs_to_train: int = pydantic.Field(default=500)
    optimizer_parameters: dict = pydantic.Field(default={})
    scheduler_parameters: dict = pydantic.Field(default={})
    verbose: bool = pydantic.Field(default=False)

class Experiment(pydantic.BaseModel):
    dataset_name: str
    dataset_number_of_points: int = pydantic.Field(default=1000)
    dataloader_parameters: dict = pydantic.Field(default={})
    dataset_parameters: dict = pydantic.Field(default={})
    pushforward_operator_name: str
    pushforward_operator_parameters: dict = pydantic.Field(default={})
    train_parameters: TrainParameters = pydantic.Field(default=TrainParameters())
    tensor_parameteres: dict = pydantic.Field(default={})
    path_to_result: str | None = pydantic.Field(default=None)