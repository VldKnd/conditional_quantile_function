import torch


def compute_hausdorff_distance(
    A_set: torch.Tensor,
    B_set: torch.Tensor,
) -> float:
    """
    Computes the Hausdorff distance between two sets.

    Args:
        A_set: torch.Tensor,
        B_set: torch.Tensor

    Returns:
        Hausdorff distance
    """

    distances = torch.cdist(A_set, B_set)
    inf_A_to_B, _ = distances.min(dim=-1)
    inf_B_to_A, _ = distances.min(dim=-2)
    sup_inf_A = torch.max(inf_A_to_B)
    sup_inf_B = torch.max(inf_B_to_A)
    return max(sup_inf_A.item(), sup_inf_B.item())
