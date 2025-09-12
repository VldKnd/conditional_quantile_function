from typing_extensions import Literal
import torch
import scipy.stats as stats


def get_quantile_level_analytically(
    alpha: torch.Tensor, distribution: Literal["gaussian", "uniform_ball"],
    dimension: int
) -> torch.Tensor:
    """Function finds the radius, that is corresponding to alpha-quantile of the samples.

    The function is based on the fact, that the distribution of the distances is symmetric around the origin.
    So, we can find the radius, that is corresponding to alpha-quantile of the samples.

    Args:
        alpha (torch.Tensor): Level of the quantile.
        distribution (Literal["gaussian", "ball"]): Distribution of the samples.
        dimension (int): Dimension of the samples.

    Returns:
        torch.Tensor: The radius of the quantile level.
    """
    if distribution == "gaussian":
        scipy_quantile = stats.chi2.ppf(alpha.cpu().detach().numpy(), df=dimension)
        return torch.from_numpy(scipy_quantile**(1 / 2)).to(alpha)
    elif distribution == "uniform_ball":
        return alpha**(1 / dimension)
    else:
        raise ValueError(f"Distribution {distribution} is not supported.")


def get_quantile_level_numerically(samples: torch.Tensor, alpha: float) -> float:
    """Function finds the radius, that is corresponding to alpha-quantile of the samples.

    The function is based on the fact, that the distribution of the distances is symmetric around the origin.
    So, we can find the radius, that is corresponding to alpha-quantile of the samples.

    Args:
        samples (torch.Tensor): Samples from the distribution.
        alpha (float): Level of the quantile.

    Returns:
        float: The radius of the quantile level.
    """
    distances = torch.norm(samples, dim=-1).reshape(-1)
    distances, _ = distances.sort()
    return distances[int(alpha * len(distances))]
