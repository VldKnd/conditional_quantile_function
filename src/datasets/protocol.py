import torch
from typing import Tuple


class Dataset:

    def __init__(
        self, tensor_parameters: dict = {}, seed: int = 31337, *args, **kwargs
    ):
        ...

    def sample_conditional(self,
                           X_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Returns:
            (x, y) - Union[torch.Tensor[n, k], torch.Tensor[n, p]]
        """
        raise NotImplementedError(
            "Sampling of covariates is not implemented for this dataset."
        )

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Returns:
            (x, y) - Union[torch.Tensor[n, k], torch.Tensor[n, p]]
        """
        raise NotImplementedError(
            "Sampling of joint distribution is not implemented for this dataset."
        )

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        raise NotImplementedError(
            "Sampling of covariates is not implemented for this dataset."
        )

    def sample_x_y_u(self,
                     n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples triple (x, y, u), where y = f(x, u).

        Returns:
            (x, y, u) - Union[torch.Tensor[n, k], torch.Tensor[n, p], torch.Tensor[n, q]]
        """
        raise NotImplementedError(
            "Sampling of x, y, u is not implemented for this dataset."
        )

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        raise NotImplementedError(
            "Pushforward of u is not implemented for this dataset."
        )
