import math
import numpy as np
import torch

from datasets.protocol import Dataset


class EightGaussians(Dataset):
    """
    2D mixture of eight Gaussians, conditional on one-dimensional covariate
    Adapted from: https://github.com/CW-Huang/CP-Flow/blob/main/data/toy_data.py#L53
    """

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        x = torch.rand(size=(n_points, 1)) * 2 + 0.5
        return x.to(**self.tensor_prameters)

    def meshgrid_of_covariates(self, n_points_per_dimension: int) -> torch.Tensor:
        return torch.linspace(0.5, 2.5, steps=n_points_per_dimension, **self.tensor_prameters)

    def sample_conditional(self, n_points: int, x: torch.Tensor) -> torch.Tensor:
        n_x, d = x.shape
        #print(f"{x.shape=}")
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / math.sqrt(2), 1. / math.sqrt(2)),
                   (1. / math.sqrt(2), -1. / math.sqrt(2)),
                   (-1. / math.sqrt(2), 1. / math.sqrt(2)),
                   (-1. / math.sqrt(2), -1. / math.sqrt(2))]
        centers = torch.tensor(data=[(scale * x, scale * y)
                                     for x, y in centers],
                               **self.tensor_prameters)

        idx = np.random.randint(8, size=(n_x, n_points))
        #print(idx[:5])

        selected_centers = centers[idx]

        #print(f"{selected_centers.shape=}")

        points = torch.randn(size=(n_x, n_points, 2)) * 0.5

        y = (points + selected_centers * torch.abs(x).reshape(-1, 1,
                                                               1)) / 1.414

        return y

    def sample_joint(self, n_points: int) -> torch.Tensor:
        x = self.sample_covariates(n_points=n_points)
        y = self.sample_conditional(n_points=1, x=x).squeeze(dim=1)
        return x, y
