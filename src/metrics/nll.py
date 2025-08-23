import torch


def compute_gaussian_negative_log_likelihood(X: torch.Tensor) -> float:
    """
    Computes the likelihood of the model.

    Args:
        X: The input tensor, assumed to be sampled from normal distribution.

    Returns:
        The negative log likelihood.
    """
    constant = torch.log(torch.sqrt(torch.tensor(2 * torch.pi)))
    return (((X)**2 / 2) + constant).mean()
