import torch


def percentage_of_monotonicity_violation(
    tensor: torch.Tensor, pushforward_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Computes the percentage of co-monotonicity violation between two sets of points.
    """
    return (
        1 - tensor.unsqueeze(1).sub(tensor.unsqueeze(0)).mul(
            pushforward_tensor.unsqueeze(1).sub(pushforward_tensor.unsqueeze(0))
        ).sum(dim=-1).ge(0).float().mean()
    ) * 100
