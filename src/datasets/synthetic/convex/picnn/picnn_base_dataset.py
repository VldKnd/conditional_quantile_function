from datasets.protocol import Dataset
import torch
from typing import Tuple
import os


class PICNN_BaseDataset(Dataset):

    def __init__(self, tensor_parameters: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_parameters = tensor_parameters
        try:
            self.model = self.load_model()
        except FileNotFoundError:
            print(f"Model file not found when initializing {self.__class__.__name__}.")
            self.model = None
            raise FileNotFoundError

    def load_model(self):
        raise NotImplementedError("This dataset is not yet implemented properly.")

    def sample_latent_variables(self, n_points: int) -> torch.Tensor:
        raise NotImplementedError(
            "Sampling of latent variables is not implemented for this dataset."
        )

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        raise NotImplementedError(
            "Sampling of covariates is not implemented for this dataset."
        )

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        batch_size = 256

        for i in range(0, u.shape[0], batch_size):
            u_batch = u[i:i + batch_size]
            x_batch = x[i:i + batch_size]
            y_batch = self.model.push_u_given_x(u_batch, x_batch)
            if i == 0:
                y_batch_all = y_batch
            else:
                y_batch_all = torch.cat([y_batch_all, y_batch], dim=0)

        return y_batch_all

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Returns:
            (x, y) - Union[torch.Tensor[n, k], torch.Tensor[n, p]]
        """
        x = self.sample_covariates(n_points=n_points)
        u = self.sample_latent_variables(n_points=n_points)
        y = self.push_u_given_x(u, x)
        return x, y

    def sample_x_y_u(self,
                     n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples triple (x, y, u), where y = f(x, u).

        Returns:
            (x, y, u) - Union[torch.Tensor[n, k], torch.Tensor[n, p], torch.Tensor[n, q]]
        """
        x = self.sample_covariates(n_points=n_points)
        u = self.sample_latent_variables(n_points=n_points)
        y = self.push_u_given_x(u, x)
        return x, y, u
