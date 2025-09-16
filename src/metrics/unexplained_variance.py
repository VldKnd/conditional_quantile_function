import torch


def percentage_of_unexplained_variance(
    ground_truth: torch.Tensor, approximation: torch.Tensor
) -> torch.Tensor:
    """
    Computes the unexplained variance percentage between two sets of points.
    """
    ground_truth_flat = ground_truth.flatten(start_dim=0, end_dim=-2)
    approximation_flat = approximation.flatten(start_dim=0, end_dim=-2)
    return (
        ground_truth_flat.sub(approximation_flat).norm(dim=-1).pow(2).mean() /
        ground_truth_flat.mean(dim=0, keepdim=True).sub(ground_truth_flat
                                                        ).norm(dim=-1).pow(2).mean()
    ) * 100


if __name__ == "__main__":
    ground_truth = torch.rand(1000, 5)
    for alpha in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        approximation = ground_truth * (1 - alpha) + torch.rand(1000, 5) * alpha
        print(percentage_of_unexplained_variance(ground_truth, approximation))
