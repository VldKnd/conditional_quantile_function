import torch

class StarDataset:
    """
    A class to transform a 2D Gaussian distribution into a swirled star shape
    and back using PyTorch.

    The process is as follows:
    1.  Sample points from a standard N(0, I) Gaussian distribution.
    2.  Apply a radial distortion in polar coordinates to create the star shape.
    3.  Apply a swirl effect by rotating points based on their new radius.
    4.  Apply a final linear transformation to match a target mean and covariance.

    The entire transformation is invertible.
    """

    def __init__(self, n_lobes: int, tensor_parameters: dict, seed: int):
        """
        Initializes the StarShaper.

        Args:
            n_lobes (int): The number of lobes (points) of the star.
            tensor_parameters (dict): The parameters for the tensor.
            seed (int): The seed for the random number generator.
        """
        self.tensor_parameters = tensor_parameters
        self.seed = seed
        self.random_number_generator = torch.Generator(device=self.tensor_parameters["device"])
        self.random_number_generator.manual_seed(self.seed)

        self.n_lobes = n_lobes

    def _to_polar(self, XY: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts Cartesian coordinates to polar coordinates."""
        XY_shape_without_last_dimension = XY.shape[:-1]
        XY_flattened = XY.reshape(-1, 2)
        X = XY_flattened[:, 0]
        Y = XY_flattened[:, 1]
        radius = torch.sqrt(X**2 + Y**2)
        angle = torch.atan2(Y, X)
        radius = radius.reshape(*XY_shape_without_last_dimension, 1)
        angle = angle.reshape(*XY_shape_without_last_dimension, 1)
        return radius, angle

    def _to_cartesian(self, radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Converts polar coordinates to Cartesian coordinates."""
        X = radius * torch.cos(angle)
        Y = radius * torch.sin(angle)
        return torch.cat([X, Y], dim=-1)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """Samples covariates from a standard Gaussian distribution."""
        return torch.rand(n_points, 2, **self.tensor_parameters, generator=self.random_number_generator)* torch.tensor([2, 1], **self.tensor_parameters)

    def sample_conditional(self, n_points: int, X: torch.Tensor) -> torch.Tensor:
        """Samples the conditional distribution of the response given the covariates."""
        U = torch.randn(X.shape[0], n_points, 2, **self.tensor_parameters, generator=self.random_number_generator)
        X_extended = X.unsqueeze(1).repeat(1, n_points, 1)
        radius, angle = self._to_polar(U)
        lobe_strength = X_extended[:, :, 0:1]
        swirl_strength = X_extended[:, :, 1:2]

        radius_star = radius * (1 + lobe_strength * torch.sin(self.n_lobes * angle))
        angle_star = angle + swirl_strength * radius_star

        Y = self._to_cartesian(radius_star, angle_star)

        return Y

    def sample_joint(self, n_points: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples the joint distribution of the covariates and the response."""
        X = self.sample_covariates(n_points)
        Y = self.sample_conditional(1, X).squeeze(1)
        return X, Y

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes forward the noise given the covariates."""
        radius, angle = self._to_polar(u)
        lobe_strength = x[:, 0:1]
        swirl_strength = x[:, 1:2]
        radius_star = radius * (1 + lobe_strength * torch.sin(self.n_lobes * angle))
        angle_star = angle + swirl_strength * radius_star
        Y = self._to_cartesian(radius_star, angle_star)
        return Y

    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        final_radius, final_angle = self._to_polar(y)
        lobe_strength = x[:, 0:1]
        swirl_strength = x[:, 1:2]

        original_angle = final_angle - swirl_strength * final_radius
        denominator = (1 + lobe_strength * torch.sin(self.n_lobes * original_angle))
        denominator[denominator == 0] = 1e-10
        original_radius = final_radius / denominator

        U = self._to_cartesian(original_radius, original_angle)
        return U

    def meshgrid_of_covariates(self, n_points_per_dimension: int) -> torch.Tensor:
        """Creates a meshgrid of covariates."""
        X0 = torch.linspace(0, 1, n_points_per_dimension, **self.tensor_parameters)
        X1 = torch.linspace(0, 1, n_points_per_dimension, **self.tensor_parameters)
        return torch.stack(torch.meshgrid(X0, X1), dim=-1).reshape(2, -1).T