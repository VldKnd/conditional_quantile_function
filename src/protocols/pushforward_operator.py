from abc import ABC
import torch
from typing_extensions import TypedDict

class TrainParams(TypedDict):
    num_epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    verbose: bool = False

class PushForwardOperator(ABC):
    def push_forward_u_given_x(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Generates Y|X by applying a push forward operator to U.

        Args:
            U (torch.Tensor): Random variable to be pushed forward.
            X (torch.Tensor): Condition.

        Returns:
            torch.Tensor: Y|X.
        """
        ...

    def sample_y_given_x(self, n_samples: int, X: torch.Tensor) -> torch.Tensor:
        """Samples Y|X.

        Args:
            n_samples (int): Number of samples to draw.
            X (torch.Tensor): Condition.

        Returns:
            torch.Tensor: Y|X.
        """
        ...

    def fit(self, dataloader: torch.utils.data.DataLoader, *args, train_params: TrainParams = TrainParams(), **kwargs) -> "PushForwardOperator":
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_params (TrainParams): Training parameters.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            self: The fitted pushforward operator.
        """
        ...

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        ...

    def load(self, path: str) -> "PushForwardOperator":
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        ...