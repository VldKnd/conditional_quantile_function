import torch
from typing import Union


class Dataset:
    def __init__(self, tensor_parameters: dict = {}, seed: int = 31337, *args, **kwargs):
        self.tensor_prameters = tensor_parameters
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

    def meshgrid_of_covariates(self, n_points_per_dimension: int) -> torch.Tensor:
        """
            Creates uniform grid of covariates.

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

    def pushbackward_Y_given_X(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Push backwards the conditional distribution of the response given the covariates.
        """
        raise NotImplementedError("Pushbackward of y is not implemented for this dataset.")

    def pushforward_U_given_X(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        raise NotImplementedError("Pushforward of u is not implemented for this dataset.")
