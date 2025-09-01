import torch
import ot
from tqdm import tqdm


def sliced_wasserstein2(
    ground_truth: torch.Tensor, approximation: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Sliced Wasserstein distance between two sets of points.

    Args:
        ground_truth (torch.Tensor): The ground truth points.
        approximation (torch.Tensor): The approximation points.

    Returns:
        float: The Sliced Wasserstein distance between the two sets of points.
    """
    return ot.sliced_wasserstein_distance(
        X_s=ground_truth, X_t=approximation, n_projections=1000
    )


if __name__ == "__main__":
    statistics = {"10": [], "100": [], "1000": [], "10000": []}

    distances = []
    pbar = tqdm(range(10000))

    for i in pbar:
        ground_truth = torch.rand(10000, 5)
        switch = (torch.rand(10000, 5) > 0.5).float()
        approximation = (torch.rand(10000, 5) +
                         0.5) * switch + (torch.rand(10000, 5) - 0.5) * (1 - switch)
        distances.append(sliced_wasserstein2(ground_truth, approximation))
        pbar.set_description(
            f"Mean: {torch.tensor(distances).mean()}, Std: {torch.tensor(distances).std()}"
        )
