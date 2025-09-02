import math
import torch
from tqdm import tqdm


def log_kde_gaussian(
    query_points: torch.Tensor, density_points: torch.Tensor, bandwidth: float = 0.5
):
    d = query_points.shape[-1]
    bandwidth_squared = torch.tensor(bandwidth**2).to(query_points)
    query_points_squeezed = query_points.reshape(-1, d)
    density_points_squeezed = density_points.reshape(-1, d)
    number_of_points_from_density, _ = density_points_squeezed.shape

    pairwise_differences = (
        query_points_squeezed.unsqueeze(1) - density_points_squeezed.unsqueeze(0)
    )

    exponent = -(pairwise_differences.norm(dim=-1)**2 / (2 * bandwidth_squared))

    log_const = (
        - 0.5 * d * torch.log(2 * torch.pi * bandwidth_squared) +\
        - math.log(number_of_points_from_density)
    )

    log_density_estimates = torch.logsumexp(exponent, dim=1) + log_const
    return log_density_estimates.reshape(query_points.shape[:-1])


def kernel_density_estimate_kl_divergence(
    ground_truth: torch.Tensor, approximation: torch.Tensor, sample: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Kernel Density Estimate Kullback-Leibler divergence between two sets of points.
    """
    ground_truth_log_density = log_kde_gaussian(sample, ground_truth)
    approximation_log_density = log_kde_gaussian(sample, approximation)
    return ground_truth_log_density.exp(
    ).mul(ground_truth_log_density - approximation_log_density).mean()


def kernel_density_estimate_l1_divergence(
    ground_truth: torch.Tensor, approximation: torch.Tensor, sample: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Kernel Density Estimate L1 divergence between two sets of points.
    """
    ground_truth_log_density = log_kde_gaussian(sample, ground_truth)
    approximation_log_density = log_kde_gaussian(sample, approximation)
    return (approximation_log_density.exp() -
            ground_truth_log_density.exp()).abs().mean()


if __name__ == "__main__":
    n_samples = 1000
    distances_l1 = torch.tensor([])
    distances_kl = torch.tensor([])
    pbar = tqdm(range(10000))

    for i in pbar:
        sample = torch.rand(n_samples, 5)
        ground_truth = torch.rand(n_samples, 5)
        switch = (torch.rand(n_samples, 5) > 0.5).float()
        approximation = (torch.rand(n_samples, 5) + 0.5
                         ) * switch + (torch.rand(n_samples, 5) - 0.5) * (1 - switch)
        distances_l1 = torch.cat(
            [
                distances_l1,
                torch.tensor(
                    [
                        kernel_density_estimate_l1_divergence(
                            ground_truth, approximation, sample
                        )
                    ]
                )
            ]
        )
        distances_kl = torch.cat(
            [
                distances_kl,
                torch.tensor(
                    [
                        kernel_density_estimate_kl_divergence(
                            ground_truth, approximation, sample
                        )
                    ]
                )
            ]
        )
        pbar.set_description(
            f"Mean L1: {distances_l1.mean():.3f}, "
            f"Std: {distances_l1.std():.3f}, "
            f"Mean KL: {distances_kl.mean():.3f}, "
            f"Std: {distances_kl.std():.3f}"
        )
