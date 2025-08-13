from abc import ABC
import torch

class PushForwardOperator(ABC):
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Generates U by applying a push forward operator to Y|X.

        Args:
            Y (torch.Tensor): Data distribution.
            X (torch.Tensor): Condition.

        Returns:
            torch.Tensor: U.
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