from typing import Union, Tuple

import numpy as np
import torch

from datasets.protocol import Dataset


class NPYDataset(Dataset):

    def __init__(self,
                 path_to_file: str,
                 tensor_parameters: dict,
                 seed: int = 31337,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_parameters = tensor_parameters
        self.seed = seed

        self.path_to_file = path_to_file

        with np.load(self.path_to_file) as npzf:
            for part_name in ("X_train", "Y_train", "X_test", "Y_test",
                              "X_cal", "Y_cal", "X_train_raw", "X_test_raw",
                              "X_cal_raw"):
                setattr(
                    self, part_name,
                    torch.tensor(npzf[part_name], **self.tensor_parameters))

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
        Sample the covariates from the uniform distribution between 1 and 5.
        """
        raise NotImplementedError(
            "This is a real-world dataset with a finite number of samples.")

    def sample_conditional(self, n_points: int,
                           x: torch.Tensor) -> torch.Tensor:
        """
        Sample the conditional distribution of the response given the covariates.
        """
        raise NotImplementedError(
            "This is a real-world dataset with a finite number of samples.")

    def sample_joint(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the joint distribution of the covariates and the response.
        """
        return self.X_train[:n_points], self.Y_train[:n_points]

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Push forward the conditional distribution of the covariates given the response.
        """
        raise NotImplementedError(
            "This is a real-world dataset with a finite number of samples.")

    def meshgrid_of_covariates(self,
                               n_points_per_dimension: int) -> torch.Tensor:
        """
        Create a meshgrid of covariates.
        """
        x = torch.linspace(-1., 1., n_points_per_dimension)
        return x.unsqueeze(1)
