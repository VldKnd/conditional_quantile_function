import torch

from typing import Tuple
from datasets.protocol import Dataset


class BananaDataset(Dataset):
    """
    Creating data in the form of a banana with x values distributed between 1 and 5.

    X: 1D, distributed between 1 and 5.
    Y: 2D, derived from x and random noise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
        Sample the covariates from the uniform distribution between 1 and 5.
        """
        x = torch.rand(size=(n_points, 1)) * 2 + 0.5
        return x

    def sample_conditional(self, n_points: int, X: torch.Tensor) -> torch.Tensor:
        """
        Sample the conditional distribution of the response given the covariates.
        """
        U = torch.randn(size=(X.shape[0], n_points, 2))
        X_unsqueezed = X.unsqueeze(1)
        y = torch.concatenate(
            [
                U[:, :, 0:1] * X_unsqueezed,
                U[:, :, 1:2] / X_unsqueezed + (U[:, :, 0:1] ** 2 + X_unsqueezed**3),
            ],
            dim=-1,
        )
        return y

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the joint distribution of the covariates and the response.
        """
        X = self.sample_covariates(n_points=n_points)
        U = torch.randn(size=(X.shape[0], 2))
        Y = torch.concatenate(
            [
                U[:, 0:1] * X,
                U[:, 1:2] / X + (U[:, 0:1] ** 2 + X**3),
            ],
            dim=-1,
        )

        return X, Y

    def pushbackwards_Y_given_X(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Push backwards the conditional distribution of the response given the covariates.
        """
        assert Y.shape[0] == X.shape[0], "The number of rows in Y and X must be the same."
        U0 = Y[:, 0:1] / X
        U1 = (Y[:, 1:2] - (U0 ** 2 + X**3))* X
        U = torch.concatenate([U0, U1], dim=-1)
        return U