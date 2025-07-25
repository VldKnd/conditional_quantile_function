import torch

def sinkhorn(x, y, epsilon, n, niter):
    """
    Computes the Sinkhorn distance between two sets of points.

    Args:
        x (torch.Tensor): The ground truth points.
        y (torch.Tensor): The approximation points.
        epsilon (float): The regularization parameter.
        n (int): The number of ground truth points.
        niter (int): The number of iterations.

    Returns:
        float: The Sinkhorn distance between the two sets of points.
    """
    n, d = x.shape
    m, d = y.shape
    cost_matrix = torch.cdist(x1=x.unsqueeze(0), x2=y.unsqueeze(0), p=2) / d
    cost_matrix = cost_matrix.squeeze(0)

    mu = 1. / n * torch.ones(n, 1)
    nu = 1. / m * torch.ones(m, 1)
    thresh = 10**(-5)

    u, v = 0. * mu, 0. * nu

    for _ in range(niter):
        u_prev = u

        u = epsilon * (torch.log(mu) - torch.logsumexp((-cost_matrix + u + v) / epsilon, dim=1, keepdim=True)) + u
        v = epsilon * (torch.log(nu) - torch.logsumexp((-cost_matrix + u + v).T / epsilon, dim=1, keepdim=True)) + v

        if ((u - u_prev).abs().sum() < thresh):
            break

    pi = torch.exp((-cost_matrix + u + v) / epsilon)
    cost = torch.sum(pi * cost_matrix)
    return cost


def wassertein2(ground_truth: torch.Tensor, approximation: torch.Tensor, **kwargs):
    """
    Computes the Wasserstein distance between two sets of points.

    Args:
        ground_truth (torch.Tensor): The ground truth points.
        approximation (torch.Tensor): The approximation points.

    Returns:
        float: The Wasserstein distance between the two sets of points.
    """
    return sinkhorn(x=ground_truth, y=approximation, epsilon=1e-2, n=ground_truth.shape[0], niter=1000, **kwargs)

if __name__ == "__main__":
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]:
        ground_truth = torch.rand(10, 5)
        approximation = torch.randn(10, 5) * alpha
        print(f"{wassertein2(ground_truth, approximation)}")
