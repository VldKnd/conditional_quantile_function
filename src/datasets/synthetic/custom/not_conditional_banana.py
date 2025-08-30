import torch

from typing import Tuple
from datasets.protocol import Dataset


class NotConditionalBananaDataset(Dataset):
    """
    Creating data in the form of a banana with x values distributed between 1 and 5.

    X: 1D, distributed between 1 and 5.
    Y: 2D, derived from x and random noise.
    """

    def __init__(self, tensor_parameters: dict, seed: int = 31337, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_parameters = tensor_parameters
        self.seed = seed

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
        Sample the covariates from the uniform distribution between 1 and 5.
        """
        x = torch.ones(size=(n_points, 1)) * 1.5
        return x.to(**self.tensor_parameters)

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the joint distribution of the covariates and the response.
        """
        X = self.sample_covariates(n_points=n_points)
        U = torch.randn(size=(X.shape[0], 2)).to(**self.tensor_parameters)
        Y = torch.concatenate(
            [
                U[:, 0:1] * X,
                U[:, 1:2] / X + (U[:, 0:1]**2 + X**3),
            ],
            dim=-1,
        )

        return X, Y

    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        assert y.shape[0] == x.shape[0], (
            "The number of rows in Y and X must be the same."
        )

        U_shape = y.shape[:-1] + (2, )
        Y_flat = x.reshape(-1, 2)
        X_flat = x.reshape(-1, 1)

        U = torch.concatenate(
            [
                Y_flat[:, 0:1] / X_flat,
                (Y_flat[:, 1:2] - ((Y_flat[:, 0:1] / X_flat)**2 + X_flat**3)) * X_flat,
            ],
            dim=-1,
        )

        return U.reshape(U_shape)

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        assert u.shape[:-1] == x.shape[:-1], (
            "The number of rows in U and X must be the same."
        )
        Y_shape = u.shape[:-1] + (2, )

        U_flat = u.reshape(-1, 2)
        X_flat = x.reshape(-1, 1)
        Y_flat = torch.concatenate(
            [
                U_flat[:, 0:1] * X_flat,
                U_flat[:, 1:2] / X_flat + (U_flat[:, 0:1]**2 + X_flat**3),
            ],
            dim=1,
        )
        return Y_flat.reshape(Y_shape)