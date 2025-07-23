def w2(Y_ground_truth: torch.Tensor, Y_approximate: torch.Tensor) -> float:
    """Wasserstein 2

    Args:
        Y_ground_truth (torch.Tensor[n, p]): ground truth values from the distribution
        Y_approximate (torch.Tensor[n, p]): Values sampled from approximated distribution

    Returns:
        float: wasserstein 2 distance
    """
    ...

def quantile_coverage(alpha: float, U_approximated: torch.Tensor) -> float:
    """Computes closness of coverage to alpha

    Args:
        alpha (float): Level of quantile
        U_approx (torch.Tensor): Discrete approximated quantile level

    Returns:
        float: Coefficient that estimates closness of fit to true quantile level.
    """
    ...