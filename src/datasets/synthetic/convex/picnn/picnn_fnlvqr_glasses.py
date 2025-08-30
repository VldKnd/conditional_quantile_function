from datasets.synthetic.convex.picnn.picnn_base_dataset import PICNN_BaseDataset
from pushforward_operators import AmortizedNeuralQuantileRegression
import torch
import os


class PICNN_FNLVQR_Glasses(PICNN_BaseDataset):

    def load_model(self):
        return AmortizedNeuralQuantileRegression.load_class(
            f"{os.path.dirname(__file__)}/parameters/amortized_neural_quantile_regression_fnlvqr_glasses.pth"
        ).to(**self.tensor_parameters)

    def sample_latent_variables(self, n_points: int) -> torch.Tensor:
        """
        Samples latent variables.
        """
        return torch.randn(n_points, 1).to(**self.tensor_parameters)

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        """
            Returns:
            torch.Tensor[n, k]
        """
        x = torch.rand(n_points, 1)
        return x.to(**self.tensor_parameters)
