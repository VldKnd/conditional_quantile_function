from typing import Literal
from utils import get_quantile_level_numerically
from distributions import sample_ball
import torch

def compare_coverage_in_latent_space(
        discrete_quantile: torch.Tensor,
        alpha: float,
        distribution_type: Literal["gaussian", "sphere"] = "gaussian"
    ):
    if distribution_type not in {"gaussian", "sphere"}:
        raise RuntimeError(f"Distribution type {distribution_type} is not supported")

    if distribution_type == "gaussian":
        samples = torch.randn(10**6, discrete_quantile.shape[1])
    elif distribution_type == "sphere":
        samples = sample_ball(10**6, discrete_quantile.shape[1])

    quantile_level_radius = get_quantile_level_numerically(samples=samples, alpha=alpha)
    return torch.mean(discrete_quantile.norm(dim=-1) - quantile_level_radius)