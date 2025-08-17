import torch
from datasets.synthetic.quadratic_potential import ConvexQuadraticPotential
from typing import Tuple
from datasets.protocol import Dataset
import os

class ConvexBananaDataset(Dataset):
    """
    Creating data in the form of a banana with x values distributed between 1 and 5.

    X: 1D, distributed between 1 and 5.
    Y: 2D, derived from x and random noise.
    """

    def __init__(self,
            tensor_parameters: dict,
            seed: int = 31337,
            path_to_parameters: str = f"{os.path.dirname(__file__)}/parameters/convex_banana.pth",
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.tensor_parameters = tensor_parameters
        self.path_to_parameters = path_to_parameters
        self.seed = seed
        self.quadratic_potential = ConvexQuadraticPotential.load(path_to_parameters).to(**tensor_parameters)
        
    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
        Sample the covariates from the uniform distribution between 1 and 5.
        """
        x = torch.rand(size=(n_points, 1)) * 2 + 0.5
        return x.to(**self.tensor_parameters)

    def sample_conditional(self, n_points: int, x: torch.Tensor) -> torch.Tensor:
        """
        Sample the conditional distribution of the response given the covariates.
        """

        u = torch.randn(size=(x.shape[0], n_points, 2)).to(**self.tensor_parameters).requires_grad_(True)
        x_unsqueezed = x.unsqueeze(1).repeat(1, n_points, *([1]*len(x.shape[1:])))
        u_potential = self.quadratic_potential(x_unsqueezed, u)
        y = torch.autograd.grad(u_potential.sum(), u)[0]
        return y

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the joint distribution of the covariates and the response.
        """
        x = self.sample_covariates(n_points=n_points)
        u = torch.randn(size=(x.shape[0], 2)).to(**self.tensor_parameters).requires_grad_(True)
        u_potential = self.quadratic_potential(x, u)
        y = torch.autograd.grad(u_potential.sum(), u)[0]

        return x, y
    
    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        assert u.shape[:-1] == x.shape[:-1], (
            "The number of rows in U and X must be the same."
        )
        u_requieres_grad = u.requires_grad
        u = u.requires_grad_(True)
        u_potential = self.quadratic_potential(x, u)
        y = torch.autograd.grad(u_potential.sum(), u)[0]
        u.requires_grad = u_requieres_grad
        return y

    def meshgrid_of_covariates(self, n_points_per_dimension: int) -> torch.Tensor:
        """
        Create a meshgrid of covariates.
        """
        x = torch.linspace(0.5, 2.5, n_points_per_dimension)
        return x.unsqueeze(1)
