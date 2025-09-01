import torch


def scott_bandwidth_matrix(data: torch.Tensor):
    d = data.shape[-1]
    data_squeezed = data.reshape(-1, d)
    n, d = data_squeezed.shape
    data_centered = data_squeezed - data_squeezed.mean(dim=0, keepdim=True)
    Sigma = (data_centered.T @ data_centered) / (n - 1)
    H = (n**(-2.0 / (d + 4))) * Sigma
    return H


def log_kde_gaussian(points: torch.Tensor, points_to_evaluate_density: torch.Tensor):
    H = scott_bandwidth_matrix(points_to_evaluate_density)

    L = torch.linalg.cholesky(H)
    logdetH = 2.0 * torch.log(torch.diag(L)).sum()
    H_inv = torch.cholesky_inverse(L)

    d = points.shape[-1]
    points_squeezed = points.reshape(-1, d)
    points_to_evaluate_density_squeezed = points.reshape(-1, d)

    number_of_points, d = points_squeezed.shape
    number_of_points_from_density, d = points_to_evaluate_density.shape

    # Differences: (m, n, d)
    diffs = (
        points_squeezed.unsqueeze(1) - points_to_evaluate_density_squeezed.unsqueeze(0)
    )

    # Quadratic form: (m, n)
    # q = -0.5 * (diffs @ H_inv * diffs).sum(-1)
    # compute efficiently:
    tmp = torch.matmul(diffs, H_inv)  # (m, n, d)
    q = -0.5 * (tmp * diffs).sum(dim=-1)  # (m, n)

    # log kernel values
    log_const = -0.5 * (d * torch.log(2 * torch.tensor(pi)) + logdetH)  # scalar
    logkern = log_const + q  # (m, n)

    # log-sum-exp across sample components, then subtract log n
    lse = torch.logsumexp(logkern, dim=1) - torch.log(torch.tensor(float(n)))
    return lse


def kernel_density_estimate_kl_divergence(
    ground_truth: torch.Tensor, approximation: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Kernel Density Estimate Kullback-Leibler divergence between two sets of points.
    """
    dimension = ground_truth_bandwidth.shape[-1]
    pairwise_distances = ground_truth.unsqueeze(0) - approximation.unsqueeze(
        1
    )  # batch-gt x batch_approx x d
    bandwidth = 1.
    log_density = torch.logsumexp()

    # return torch.nn.functional.kl_div(approximation, ground_truth, reduction="mean")


def kernel_density_estimate_l1_divergence(
    ground_truth: torch.Tensor, approximation: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Kernel Density Estimate L1 divergence between two sets of points.
    """
    # return torch.nn.functional.l1_loss(approximation, ground_truth, reduction="mean")
