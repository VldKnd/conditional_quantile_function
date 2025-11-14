from re import X
from datasets.protocol import Dataset
import torch
from typing import Tuple


class FNLVQR_Banana(Dataset):

    def __init__(
        self,
        *args,
        tensor_parameters: dict = dict(dtype=torch.float64, device=torch.device("cpu")),
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.tensor_parameters = tensor_parameters
        self.args = args
        self.kwargs = kwargs

    def sample_latent_variables(self, n_points: int) -> torch.Tensor:
        """
        Samples latent variables.
        """

        z = torch.rand(n_points, 1) * 2 * torch.pi - torch.pi
        phi = torch.rand(n_points, 1) * 2 * torch.pi
        r = torch.rand(n_points, 1) * 0.2 - 0.1
        beta = torch.rand(n_points, 1)
        beta = beta / torch.norm(beta, dim=1, keepdim=True, p=1)

        return torch.cat([z, phi, r, beta], dim=1).to(**self.tensor_parameters)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        x = torch.rand(n_points, 1) * 2.4 + 0.8
        return x.to(**self.tensor_parameters)

    def sample_conditional(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = list(x.shape)
        x_flat = x.flatten(0, -2)
        n_points = x_flat.shape[0]

        u = self.sample_latent_variables(n_points)
        z, phi, r, beta = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        y0 = 0.5 * (-torch.cos(z) + 1) + r * torch.sin(phi) + torch.sin(x_flat)
        y1 = z / (beta * x_flat) + r * torch.cos(phi)

        y = torch.cat([y0, y1], dim=-1)

        return x_flat.reshape(input_shape[:-1] +
                              [-1]), y.reshape(input_shape[:-1] + [-1])

    def sample_x_y_u(self,
                     n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples triple (x, y, u), where y = f(x, u).

        Returns:
            (x, y, u) - Union[torch.Tensor[n, k], torch.Tensor[n, p], torch.Tensor[n, q]]
        """
        u = self.sample_latent_variables(n_points)
        x = self.sample_covariates(n_points)

        z, phi, r, beta = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        y0 = 0.5 * (-torch.cos(z) + 1) + r * torch.sin(phi) + torch.sin(x)
        y1 = z / (beta * x) + r * torch.cos(phi)

        y = torch.cat([y0, y1], dim=-1)

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


class Not_Conditional_FNLVQR_Banana(FNLVQR_Banana):

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        x = torch.rand(n_points, 1) * 0 + 1.
        return x.to(**self.tensor_parameters)