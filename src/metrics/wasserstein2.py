import torch
import ot

def wassertein2(ground_truth: torch.Tensor, approximation: torch.Tensor, **kwargs):
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
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]:
        ground_truth = torch.rand(1000, 5)
        approximation = torch.rand(1000, 5) * alpha + 10 * (1 - alpha)
        print(f"{wassertein2(ground_truth, approximation)}")
