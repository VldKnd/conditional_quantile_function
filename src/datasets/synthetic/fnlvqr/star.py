from datasets.protocol import Dataset
import torch
from typing import Tuple


class FNLVQR_Star(Dataset):

    def __init__(
        self,
        *args,
        tensor_parameters: dict = dict(dtype=torch.float64, device=torch.device("cpu")),
        amplitude: float = 1.,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tensor_parameters = tensor_parameters
        self.amplitude = amplitude
        self.number_of_star_angles = 3
        self.kwargs = kwargs
        self.args = args

    def sample_latent_variables(self, n_points: int) -> torch.Tensor:
        """
        Samples latent variables.
        """
        return torch.randn(n_points, 2).to(**self.tensor_parameters)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        return torch.rand(n_points, 1).to(
            **self.tensor_parameters
        ) * 2 * torch.pi / self.number_of_star_angles

    def transform_to_star(self, points: torch.Tensor) -> torch.Tensor:
        """
        Transforms a 2D point cloud into a star shape.

        Args:
            points (torch.Tensor): A torch.Tensor of shape (n, 2) representing the (x, y) coordinates.

        Returns:
            torch.Tensor: A new torch.Tensor with the transformed points.
        """
        points_shape = points.shape
        points_flat = points.reshape(-1, 2)
        x, y = points_flat[:, 0], points_flat[:, 1]
        theta = torch.arctan2(y, x)

        scaling_factor = 1 + self.amplitude * torch.cos(
            self.number_of_star_angles * theta
        )

        x_transformed = x * scaling_factor
        y_transformed = y * scaling_factor
        transformed_points = torch.stack((x_transformed, y_transformed),
                                         axis=1).reshape(points_shape)

        return transformed_points

    def rotate_points(self, points: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotates 2D points around the origin.
        """
        points_shape = points.shape
        points_flat = points.reshape(-1, 2)
        angle_flat = angle.reshape(-1, 1)
        x, y = points_flat[:, 0:1], points_flat[:, 1:2]
        cos_angle = torch.cos(angle_flat)
        sin_angle = torch.sin(angle_flat)
        x_rotated = x * cos_angle - y * sin_angle
        y_rotated = x * sin_angle + y * cos_angle
        return torch.stack((x_rotated, y_rotated), axis=1).reshape(points_shape)

    def sample_x_y_u(self,
                     n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples triple (x, y, u), where y = f(x, u).

        Returns:
            (x, y, u) - Union[torch.Tensor[n, k], torch.Tensor[n, p], torch.Tensor[n, q]]
        """
        u = self.sample_latent_variables(n_points)
        x = self.sample_covariates(n_points)
        star_points = self.transform_to_star(u)
        y = self.rotate_points(star_points, x)

        return x, y, u

    def sample_conditional(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = list(x.shape)
        x_flat = x.flatten(0, -2)
        n_points = x_flat.shape[0]

        u = self.sample_latent_variables(n_points)
        star_points = self.transform_to_star(u)
        y = self.rotate_points(star_points, x_flat)
        u = self.sample_latent_variables(n_points)

        z1 = 3 * torch.pi * x_flat
        z2 = torch.pi * (1 + 3 * x_flat)
        y1 = 5 * torch.sin(z1) + 2.5 + u[:, 0:1]
        y2 = 5 * torch.sin(z2) + 2.5 - u[:, 0:1]
        y = u[:, 1:2] * y1 + (1 - u[:, 1:2]) * y2

        return x_flat.reshape(input_shape[:-1] +
                              [-1]), y.reshape(input_shape[:-1] + [-1])

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
        return self.rotate_points(self.transform_to_star(u), x)
