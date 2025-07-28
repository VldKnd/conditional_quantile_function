from abc import ABC
import torch

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