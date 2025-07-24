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