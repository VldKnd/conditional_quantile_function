import numpy as np
import torch
from scipy.stats import norm

from src.protocols.dataset import Dataset


class TicTacDataset(Dataset):
    """
    Adapted from paper "TIC-TAC: A Framework for Improved Covariance Estimation in Deep Heteroscedastic Regression"
    https://arxiv.org/abs/2310.18953v2
    https://github.com/vita-epfl/TIC-TAC/blob/main/Multivariate/sampler.py
    """

    def __init__(self, in_dim=5, out_dim=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Numpy Objects
        # Mean and Sigma over the joint distribution of X and Y.
        self.scale = out_dim
        self.mean = np.random.uniform(low=-self.scale,
                                      high=self.scale,
                                      size=(in_dim + out_dim))
        self.sigma_corr = self.get_correlation(dim=in_dim + out_dim)
        self.sigma_covar = self.sigma_corr * self.scale

        # PyTorch Objects
        self.sigma_11 = torch.from_numpy(self.sigma_covar[:out_dim, :out_dim])
        self.sigma_12 = torch.from_numpy(self.sigma_covar[:out_dim, -in_dim:])
        self.sigma_21 = torch.from_numpy(self.sigma_covar[-in_dim:, :out_dim])
        self.sigma_22 = torch.from_numpy(self.sigma_covar[-in_dim:, -in_dim:])

        self.y_sigma = self.sigma_11 - torch.matmul(
            torch.matmul(self.sigma_12, torch.linalg.inv(self.sigma_22)), self.sigma_21)
        
        # Z Sigma which is heteroscedastic noise
        self.z_sigma = torch.from_numpy(self.get_correlation(dim=out_dim))

    def sample_covariates(self, n_points):
        return super().sample_covariates(n_points)

    def sample_conditional(self, n_points, x):
        return super().sample_conditional(n_points, x)

    def sample_joint(self, n_points):
        mean = torch.from_numpy(self.mean)

        # Sample X from uniform with the same mu and corr using gaussian copula
        # Shape: N x in_dim
        self.samples['x'] = self.get_standard_uniform_samples()
        self.samples['x'] = self.correlation_to_covariance(self.samples['x'])
        self.samples['x'] = torch.from_numpy(self.samples['x'])

        # Obtain Y given X
        # Shape: N x out_dim
        self.samples['y_mean'] = mean[:self.out_dim].view(1, self.out_dim) \
            + torch.matmul(
                torch.matmul(
                    self.sigma_12,
                    torch.linalg.inv(self.sigma_22)).expand(self.num_samples, self.out_dim, self.in_dim),
                (self.samples['x'] - mean[-self.in_dim:].view(1, self.in_dim)).unsqueeze(2)).squeeze()

        # Obtain Q_Covariance
        # Shape: N x out x out
        self.samples['z_sigma'] = self.z_sigma + torch.diag_embed(
            torch.sqrt(torch.abs(
                self.samples['x'][:, :self.out_dim] - mean[-self.in_dim:][:self.out_dim])
                ))

        self.samples['q_covariance'] = self.y_sigma + self.samples['z_sigma']

        # Obtain Q
        # Shape: N x out_dim
        self.samples['q'] = np.stack(
            [np.random.multivariate_normal(m.numpy(), c.numpy()) for m, c in zip(
                self.samples['y_mean'], self.samples['q_covariance'])])
        
        self.samples['q'] = torch.from_numpy(self.samples['q'])
    
    def get_standard_uniform_samples(self) -> float:
        correlation = self.sigma_corr[-self.in_dim:, -self.in_dim:]

        # https://stats.stackexchange.com/questions/66610/generate-pairs-of-random-numbers-uniformly-distributed-and-correlated
        spearman_rho = 2 * np.sin(correlation * np.pi / 6)
        A = np.random.multivariate_normal(mean=np.zeros(self.in_dim), cov=spearman_rho, size=self.num_samples)
        U = norm.cdf(A)

        return U


    def correlation_to_covariance(self, uniform_samples: float) -> float:
        # https://stats.stackexchange.com/questions/139804/the-meaning-of-scale-and-location-in-the-pearson-correlation-context
        # Correlation is independent of scale and we can scsale each variable independtly since uniform_b_minus_a is always positive

        means = self.mean[-self.in_dim:]
        var = self.scale
        
        uniform_b_minus_a = np.sqrt(12 * var) # scalar
        uniform_b_plus_a = 2 * means # vector: dim
        uniform_a = (uniform_b_plus_a - uniform_b_minus_a) / 2

        uniform_samples = uniform_a + (uniform_b_minus_a * uniform_samples)

        return uniform_samples


def get_correlation(self, dim: int) -> float:
    # https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor

    a = 2

    A = np.matrix(
        [np.random.randn(dim) + np.random.randn(1) * a for i in range(dim)])
    A = A * np.transpose(A)
    D_half = np.diag(np.diag(A)**-0.5)
    C = D_half * A * D_half

    return C
