import torch
from typing import Tuple


class FunnelDistribution:

    def __init__(
        self,
        tensor_parameters: dict,
        conditional_dimension: int = 1,
        target_dimension: int = 1,
        sigma: float = 3.0,
        seed: int = 31337,
        *args,
        **kwargs
    ):
        """
        Neal's funnel distribution (classical or multidimensional).

        Args:
            k (int): Number of funnel directions (scale variables v_j).
            m (int): Number of x's per v_j (block size).
            sigma (float): Std of Gaussian prior for v_j.
            device (str): Device to use ("cpu" or "cuda").
        """
        self.tensor_parameters = tensor_parameters
        self.conditional_dimension = conditional_dimension
        self.target_dimension = target_dimension
        self.sigma = sigma
        self.seed = seed

    def sample_covariates(self, n_points: int):
        condition = torch.randn(n_points, self.conditional_dimension) * self.sigma
        condition = condition.to(**self.tensor_parameters)
        return condition

    def sample_conditional(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = list(x.shape)
        x_flat = x.flatten(0, -2)
        n_points = x_flat.shape[0]

        standard_deviation = torch.exp(x_flat / 2.).unsqueeze(-1)
        target = torch.randn(
            n_points, self.conditional_dimension, self.target_dimension
        )
        target = target.to(**self.tensor_parameters)
        target = target * standard_deviation
        target = target.flatten(start_dim=1)

        return x_flat.reshape(input_shape[:-1] +
                              [-1]), target.reshape(input_shape[:-1] + [-1])

    def sample_joint(self, n_points: int):
        """
        Sample from the joint distribution (v, x).

        Args:
            n_points (int): Number of points to sample.

        Returns:
            v: Tensor of shape (n_points, k) — scale variables.
            x: Tensor of shape (n_points, k, m) — Gaussian blocks conditioned on v.
        """
        condition = torch.randn(n_points, self.conditional_dimension) * self.sigma
        condition = condition.to(**self.tensor_parameters)

        standard_deviation = torch.exp(condition / 2.).unsqueeze(-1)
        target = torch.randn(
            n_points, self.conditional_dimension, self.target_dimension
        )
        target = target.to(**self.tensor_parameters)
        target = target * standard_deviation
        target = target.flatten(start_dim=1)

        return condition, target
