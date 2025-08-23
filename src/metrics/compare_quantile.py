from typing import Literal
import scipy.stats as stats
import torch


def compare_quantile_in_latent_space(
    discrete_quantile: torch.Tensor,
    quantile_level: float,
    distribution_type: Literal["gaussian", "sphere"] = "gaussian"
):
    """
    Computes quantile levels created by the discrete quantile in the latent space.

    Args:
        discrete_quantile (torch.Tensor): The discrete quantile.
        quantile_level (float): The quantile_level level.
        distribution_type (Literal["gaussian", "sphere"], optional): The distribution type. Defaults to "gaussian".

    Raises:
        RuntimeError: If the distribution type is not supported.

    Returns:
        float: Distance from the true quantile level in the latent space.
    """
    if distribution_type not in {"gaussian", "sphere"}:
        raise RuntimeError(f"Distribution type {distribution_type} is not supported")

    if distribution_type == "gaussian":
        scipy_quantile = stats.chi2.ppf(
            [quantile_level], df=discrete_quantile.shape[-1]
        )
        quantile_level_radius = torch.from_numpy(scipy_quantile**(1 / 2)
                                                 ).to(discrete_quantile)

    elif distribution_type == "sphere":
        quantile_level_radius = quantile_level**(1 / discrete_quantile.shape[-1])

    discrete_quantile_direction = discrete_quantile / discrete_quantile.norm(
        dim=-1, keepdim=True
    )
    return torch.mean(
        (discrete_quantile -
         discrete_quantile_direction * quantile_level_radius).norm(dim=-1)
    )
