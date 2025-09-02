import torch
import ot
from tqdm import tqdm


def wassertein2(ground_truth: torch.Tensor, approximation: torch.Tensor):
    """
    Computes the Wasserstein distance between two sets of points.

    Args:
        ground_truth (torch.Tensor): The ground truth points.
        approximation (torch.Tensor): The approximation points.

    Returns:
        float: The Wasserstein distance between the two sets of points.
    """
    return ot.solve_sample(X_a=ground_truth, X_b=approximation).value


if __name__ == "__main__":
    distances = []
    pbar = tqdm(range(10000))

    for i in pbar:
        ground_truth = torch.rand(10000, 5)
        switch = (torch.rand(10000, 5) > 0.5).float()
        approximation = (torch.rand(10000, 5) +
                         0.5) * switch + (torch.rand(10000, 5) - 0.5) * (1 - switch)
        distances.append(wassertein2(ground_truth, approximation))
        pbar.set_description(
            f"Mean: {torch.tensor(distances).mean()}, Std: {torch.tensor(distances).std()}"
        )
