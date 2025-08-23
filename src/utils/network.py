import torch


def get_total_number_of_parameters(model: torch.nn.Module) -> int:
    """Get the total number of parameters in the model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())
