import torch

def get_quantile_level_numerically(samples: torch.Tensor, alpha: float) -> float:
    """Function finds the radius, that is corresponding to alpha-quantile of the samples.

    The function is based on the fact, that the distribution of the distances is symmetric around the origin.
    So, we can find the radius, that is corresponding to alpha-quantile of the samples.

    Args:
        samples (torch.Tensor): Samples from the distribution.
        alpha (float): Level of the quantile.
    """
    distances = torch.norm(samples, dim=-1).reshape(-1)
    distances, _ = distances.sort()
    return distances[int(alpha * len(distances))]