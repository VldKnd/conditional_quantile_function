import torch


def unexplained_variance_percentage(
    ground_truth: torch.Tensor, approximation: torch.Tensor
) -> float:
    """
    Computes the unexplained variance percentage between two sets of points.
    """
    return (
        ground_truth.sub(approximation).norm(dim=-1).pow(2).mean() /
        ground_truth.mean(dim=-1, keepdim=True).sub(ground_truth).norm(dim=-1
                                                                       ).pow(2).mean()
    ).item() * 100


if __name__ == "__main__":
    ground_truth = torch.rand(1000, 5)
    for alpha in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        approximation = ground_truth * (1 - alpha) + torch.rand(1000, 5) * alpha
        print(unexplained_variance_percentage(ground_truth, approximation))
