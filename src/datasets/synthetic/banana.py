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

        U = torch.randn(size=(X.shape[0], n_points, 2), device=X.device, dtype=X.dtype)
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
        U = torch.randn(size=(X.shape[0], 2), device=X.device, dtype=X.dtype)
        Y = torch.concatenate(
            [
                U[:, 0:1] * X,
                U[:, 1:2] / X + (U[:, 0:1] ** 2 + X**3),
            ],
            dim=-1,
        )

        return X, Y

    def pushbackward_Y_given_X(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Push backwards the conditional distribution of the response given the covariates.
        """
        assert Y.shape[0] == X.shape[0], "The number of rows in Y and X must be the same."

        U_shape = Y.shape[:-1] + (2,)
        Y_flat = Y.reshape(-1, 2)
        X_flat = X.reshape(-1, 1)

        U = torch.concatenate([
            Y_flat[:, 0:1] / X_flat,
            (Y_flat[:, 1:2] - ((Y_flat[:, 0:1] / X_flat) ** 2 + X_flat**3))* X_flat
        ], dim=-1)

        return U.reshape(U_shape)

    def pushforward_U_given_X(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        assert U.shape[:-1] == X.shape[:-1], "The number of rows in U and X must be the same."
        Y_shape = U.shape[:-1] + (2,)

        U_flat = U.reshape(-1, 2)
        X_flat = X.reshape(-1, 1)
        Y_flat = torch.concatenate(
            [
                U_flat[:, 0:1] * X_flat,
                U_flat[:, 1:2] / X_flat + (U_flat[:, 0:1] ** 2 + X_flat**3),
            ],
            dim=1,
        )
        return Y_flat.reshape(Y_shape)

    def meshgrid_of_covariates(self, n_points_per_dimension: int) -> torch.Tensor:
        """
        Create a meshgrid of covariates.
        """
        x = torch.linspace(0.5, 2.5, n_points_per_dimension)
        return x.unsqueeze(1)
