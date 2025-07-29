import torch
from typing import Union

class Dataset:
    def __init__(self, seed=31337, *args, **kwargs):
        self.seed = seed

    def sample_joint(self, n_points: int) -> Union[torch.Tensor, torch.Tensor]:
        """
            Returns:
            (x, y) - Union[torch.Tensor[n, k], torch.Tensor[n, p]]
        """
        pass

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        pass

    def sample_conditional(self, n_points: int, x: torch.Tensor) -> torch.Tensor:
        """Sample conditional distribution from y|x.

        Args:
            n_points (int): number of points
            x (torch.Tensor[1, k]): covariate

        Returns:
            torch.Tensor[n, p]: Conditional sample
        """
        pass
