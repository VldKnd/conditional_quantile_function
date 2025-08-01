from typing import Literal
import scipy.stats as stats
import torch

def compare_coverage_in_latent_space(
        discrete_quantile: torch.Tensor,
        alpha: float,
        distribution_type: Literal["gaussian", "sphere"] = "gaussian"
    ):
    """
    Computes the coverage of a discrete quantile in the latent space.

    Args:
        discrete_quantile (torch.Tensor): The discrete quantile.
        alpha (float): The alpha level.
        distribution_type (Literal["gaussian", "sphere"], optional): The distribution type. Defaults to "gaussian".

    Raises:
        RuntimeError: If the distribution type is not supported.

    Returns:
        float: The coverage of the discrete quantile in the latent space.
    """
    if distribution_type not in {"gaussian", "sphere"}:
        raise RuntimeError(f"Distribution type {distribution_type} is not supported")

    if distribution_type == "gaussian":
        scipy_quantile = stats.chi2.ppf([alpha], df=discrete_quantile.shape[-1])
        quantile_level_radius = torch.from_numpy(scipy_quantile**(1/2))

    elif distribution_type == "sphere":
        quantile_level_radius = alpha**(1/discrete_quantile.shape[-1])

    return torch.mean(discrete_quantile.norm(dim=-1) - quantile_level_radius)