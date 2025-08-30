from datasets.protocol import Dataset
import torch
from typing import Tuple


class FNLVQR_Glasses(Dataset):

    def __init__(
        self,
        *args,
        tensor_parameters: dict = dict(dtype=torch.float64, device=torch.device("cpu")),
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.beta_distribution = torch.distributions.beta.Beta(
            torch.tensor(0.5), torch.tensor(1.)
        )
        self.tensor_parameters = tensor_parameters
        self.args = args
        self.kwargs = kwargs

    def sample_latent_variables(self, n_points: int) -> torch.Tensor:
        """
        Samples latent variables.
        """
        epsilon = self.beta_distribution.sample((n_points, 1))
        gamma = (torch.rand(n_points, 1) > 0.5).float()
        return torch.cat([epsilon, gamma], dim=1).to(**self.tensor_parameters)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        x = torch.rand(n_points, 1)
        return x.to(**self.tensor_parameters)

    def sample_x_y_u(self,
                     n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples triple (x, y, u), where y = f(x, u).

        Returns:
            (x, y, u) - Union[torch.Tensor[n, k], torch.Tensor[n, p], torch.Tensor[n, q]]
        """
        u = self.sample_latent_variables(n_points)
        x = self.sample_covariates(n_points)

        z1 = 3 * torch.pi * x
        z2 = torch.pi * (1 + 3 * x)
        y1 = 5 * torch.sin(z1) + 2.5 + u[:, 0:1]
        y2 = 5 * torch.sin(z2) + 2.5 - u[:, 0:1]
        y = u[:, 1:2] * y1 + (1 - u[:, 1:2]) * y2

        return x, y, u

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Returns:
            (x, y) - Union[torch.Tensor[n, k], torch.Tensor[n, p]]
        """
        x, y, _ = self.sample_x_y_u(n_points)
        return x, y

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        raise NotImplementedError("Not implemented")
