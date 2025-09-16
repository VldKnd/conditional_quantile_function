import math
from typing import Union
import torch


def compute_scott_bandwidth(
    data: torch.Tensor,
    multiplier: float = 1.0,
) -> Union[float, torch.Tensor]:
    if data.ndim != 2:
        raise ValueError("data must be shape (n_samples, d)")

    n, d = data.shape
    std_per_dim = data.std(dim=0, unbiased=False)

    factor = (4.0 / (d + 2.0))**(1.0 / (d + 4.0))
    bw = std_per_dim * factor * (n**(-1.0 / (d + 4.0)))
    return bw * multiplier


def _log_kde_gaussian_chunked(
    query: torch.Tensor,
    data: torch.Tensor,
    bandwidth: torch.Tensor,
    chunk_size: int = 1000,
) -> torch.Tensor:
    """
    Compute log KDE at `query` points given `data` points with Gaussian kernel.
    - bandwidth may be a scalar (isotropic) or tensor of shape (d,) (per-dim diag).
    - chunk_size controls memory usage (number of query points processed at once).
    Returns: log densities shape (num_query,)
    """
    query = query.reshape(-1, query.shape[-1])
    data = data.reshape(-1, data.shape[-1])
    m, d = query.shape
    n = data.shape[0]

    bw = bandwidth.reshape(1, 1, d)
    log_det_term = -0.5 * torch.log(2 * math.pi * (bandwidth**2)).sum().item()
    log_ns = -math.log(n)

    out = []
    for start in range(0, m, chunk_size):

        end = min(start + chunk_size, m)
        query_chunk = query[start:end]
        diff = (query_chunk.unsqueeze(1) - data.unsqueeze(0)) / bw
        exponent = -(diff.double()**2).sum(dim=2) / 2.0

        lse = torch.logsumexp(exponent.to(dtype=torch.float64), dim=1)
        log_density_chunk = lse + (log_det_term + log_ns)
        out.append(log_density_chunk.to(dtype=torch.float32))

    return torch.cat(out, dim=0).reshape(*query.shape[:-1])


def log_kde_gaussian(
    query_points: torch.Tensor,
    density_points: torch.Tensor,
    bandwidth: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    return _log_kde_gaussian_chunked(
        query=query_points,
        data=density_points,
        bandwidth=bandwidth,
        chunk_size=chunk_size,
    )


def kernel_density_estimate_kl_divergence(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
    chunk_size: int = 1000,
) -> torch.Tensor:
    n = ground_truth.shape[0]
    sample_size = min(n, 1000)

    denisty_points_bandwidth = compute_scott_bandwidth(ground_truth)
    query_points_bandwidth = compute_scott_bandwidth(approximation)
    idx = torch.randint(0, n, (sample_size, ))
    sample = ground_truth[idx]

    log_p = log_kde_gaussian(
        sample, ground_truth, bandwidth=denisty_points_bandwidth, chunk_size=chunk_size
    )
    log_q = log_kde_gaussian(
        sample, approximation, bandwidth=query_points_bandwidth, chunk_size=chunk_size
    )

    return (log_p - log_q).mean()


def kernel_density_estimate_l1_divergence(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
    chunk_size: int = 1000,
) -> torch.Tensor:
    """
    Estimate L1 distance between densities: integral |p(x)-q(x)| dx.
    Use Monte Carlo with samples drawn from P, Q, or mixture (default).
    If sampling from mixture, importance weights cancel if sample equally from both.
    Returns scalar estimate.
    """
    n = ground_truth.shape[0]
    sample_size = min(n, 1000)

    idx = torch.randint(0, n, (sample_size, ))
    sample = ground_truth[idx]

    denisty_points_bandwidth = compute_scott_bandwidth(ground_truth)
    query_points_bandwidth = compute_scott_bandwidth(approximation)

    log_p = log_kde_gaussian(
        sample, ground_truth, bandwidth=denisty_points_bandwidth, chunk_size=chunk_size
    )
    log_q = log_kde_gaussian(
        sample, approximation, bandwidth=query_points_bandwidth, chunk_size=chunk_size
    )

    p = torch.exp(log_p)
    q = torch.exp(log_q)
    return (p - q).abs().mean()


if __name__ == "__main__":
    n_samples = 100000
    d = 5

    P = torch.randn(n_samples, d)
    Q = torch.randn(n_samples, d)

    kl = kernel_density_estimate_kl_divergence(P, Q)
    l1 = kernel_density_estimate_l1_divergence(P, Q)

    print("KL estimate:", kl.item())
    print("L1 estimate:", l1.item())
