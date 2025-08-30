import torch.nn as nn
import torch
from pushforward_operators.protocol import PushForwardOperator
import torch.utils.data


class MeanQuantileRegression(PushForwardOperator, nn.Module):

    def __init__(self, response_dimension: int, *args, **kwargs):
        super().__init__()
        self.init_dict = {"response_dimension": response_dimension}
        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

    def warmup_scalers(self, dataloader: torch.utils.data.DataLoader):
        """Run over the data (no grad) to populate BatchNorm running stats."""
        self.Y_scaler.train()

        with torch.no_grad():
            for _, Y in dataloader:
                _ = self.Y_scaler(Y)

        self.Y_scaler.eval()

    def fit(self, dataloader: torch.utils.data.DataLoader, *args, **kwargs):
        self.warmup_scalers(dataloader=dataloader)
        return self

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(y)

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        return torch.ones_like(u) * self.Y_scaler.running_mean.detach().unsqueeze(0)

    def save(self, path: str):
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.state_dict(),
                "class_name": "MeanQuantileRegression"
            }, path
        )

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self

    @classmethod
    def load_class(
        cls, path: str, map_location: torch.device = torch.device('cpu')
    ) -> "MeanQuantileRegression":
        data = torch.load(path, map_location=map_location)
        mean_quantile_regression = cls(**data["init_dict"])
        mean_quantile_regression.load_state_dict(data["state_dict"])
        return mean_quantile_regression
