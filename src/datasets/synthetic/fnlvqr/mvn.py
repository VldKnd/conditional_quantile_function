from datasets.protocol import Dataset
import os
import torch
from typing import Tuple


class FNLVQR_MVN(Dataset):

    def __init__(
        self,
        *args,
        tensor_parameters: dict = dict(dtype=torch.float64, device=torch.device("cpu")),
        number_of_features: int = 2,
        number_of_classes: int = 2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.tensor_parameters = tensor_parameters
        self.kwargs = kwargs

        path_to_parameters = f"{os.path.dirname(__file__)}/parameters/fnlvqr_mvn_{number_of_classes}_classes_{number_of_features}_features.pth"
        try:
            parameters = torch.load(path_to_parameters)
        except FileNotFoundError:
            print(
                f"Parameters file not found at {path_to_parameters}. Creating new parameters file at {path_to_parameters}."
            )
            self.create_parameters_file(
                path_to_parameters, number_of_features, number_of_classes
            )
            parameters = torch.load(path_to_parameters)

        self.projection_matrix = parameters["projection_matrix"].to(
            **self.tensor_parameters
        )
        self.latent_covariance_matrix = parameters["latent_covariance_matrix"].to(
            **self.tensor_parameters
        )
        self.number_of_features = parameters["number_of_features"]
        self.number_of_classes = parameters["number_of_classes"]

    def create_parameters_file(
        self, path_to_parameters: str, number_of_features: int, number_of_classes: int
    ):
        projection_matrix = torch.randn(number_of_features, number_of_classes)
        latent_covariance_matrix = torch.randn(number_of_classes, number_of_classes)

        torch.save(
            {
                "projection_matrix": projection_matrix,
                "latent_covariance_matrix": latent_covariance_matrix,
                "number_of_features": number_of_features,
                "number_of_classes": number_of_classes
            }, path_to_parameters
        )

    def sample_latent_variables(self, n_points: int) -> torch.Tensor:
        """
        Samples latent variables.
        """
        return torch.randn(n_points,
                           self.number_of_classes).to(**self.tensor_parameters)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        return torch.rand(n_points,
                          self.number_of_features).to(**self.tensor_parameters)

    def sample_x_y_u(self,
                     n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples triple (x, y, u), where y = f(x, u).

        Returns:
            (x, y, u) - Union[torch.Tensor[n, k], torch.Tensor[n, p], torch.Tensor[n, q]]
        """
        u = self.sample_latent_variables(n_points)
        x = self.sample_covariates(n_points)

        y = x.matmul(self.projection_matrix) + u.matmul(self.latent_covariance_matrix)
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
        return x.matmul(self.projection_matrix
                        ) + u.matmul(self.latent_covariance_matrix)
