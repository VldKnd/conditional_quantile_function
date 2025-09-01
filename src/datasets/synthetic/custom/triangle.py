import torch
from datasets.protocol import Dataset

from typing import Tuple


class TriangleDataset(Dataset):

    def __init__(self, tensor_parameters: dict = {}):
        self.tensor_parameters = tensor_parameters
        self.triangle_side = 1.
        self.v1 = torch.tensor([[0.0, 0.0]], **self.tensor_parameters)
        self.v2 = torch.tensor([[1.0, 0.0]], **self.tensor_parameters)
        self.v3 = torch.tensor([[0.5, 1.0]], **self.tensor_parameters)
        self.rotation_center = torch.tensor([[0.5, 0.33]], **self.tensor_parameters)

    def sample_latent_variables(self, n_points: int) -> torch.Tensor:
        """
        Samples latent variables.
        """
        return torch.rand(n_points, 2).to(**self.tensor_parameters)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        return torch.rand(n_points, 1).to(**self.tensor_parameters) * 2 * torch.pi / 3.

    def sample_x_y_u(self,
                     n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples triple (x, y, u), where y = f(x, u).

        Returns:
            (x, y, u) - Union[torch.Tensor[n, k], torch.Tensor[n, p], torch.Tensor[n, q]]
        """
        raw_latent_variables = self.sample_latent_variables(n_points)
        u, v = raw_latent_variables[:, 0:1], raw_latent_variables[:, 1:2]
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]

        points = self.v1 + u.unsqueeze(1) * (self.v2 - self.v1
                                             ) + v.unsqueeze(1) * (self.v3 - self.v1)
        x = self.sample_covariates(n_points)
        y = self.rotate_points(points, x, self.rotation_center)
        return x, y, u

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
      Generates a triangle by sampling points from a uniform distribution
      within a defined triangular region using PyTorch.
      """
        x, y, _ = self.sample_x_y_u(n_points)
        return x, y

    def rotate_points(
        self,
        points: torch.Tensor,
        angle: float,
        center: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Rotates a set of 2D points around a specified center.

        Args:
        points (torch.Tensor): A tensor of shape (N, 2) containing the 2D points.
        angle_deg (float): The rotation angle in degrees.
        center (torch.Tensor, optional): The center of rotation. If None,
                                it rotates around the origin (0,0).

        Returns:
        A PyTorch tensor of the same shape as 'points' with the rotated coordinates.
        """
        angle_rad = torch.deg2rad(angle)
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)

        rotation_matrix = torch.stack(
            [
                torch.cat([cos_theta, -sin_theta], dim=1),
                torch.cat([sin_theta, cos_theta], dim=1)
            ],
            dim=1
        ).to(**self.tensor_parameters)

        translated_points = points - self.rotation_center
        rotated_points = torch.bmm(translated_points, rotation_matrix.transpose(1, 2))
        rotated_points += self.rotation_center.unsqueeze(0)

        return rotated_points
