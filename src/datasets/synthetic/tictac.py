# import numpy as np
import math
import torch
# from scipy.stats import norm

from datasets.protocol import Dataset


class TicTacDataset(Dataset):
    """
    Work in progress, not yet debugged.
    Adapted from paper "TIC-TAC: A Framework for Improved Covariance Estimation in Deep Heteroscedastic Regression"
    https://arxiv.org/abs/2310.18953v2
    https://github.com/vita-epfl/TIC-TAC/blob/main/Multivariate/sampler.py
    """

    def __init__(self, in_dim=5, out_dim=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed + 1)

        # Numpy Objects
        # Mean and Sigma over the joint distribution of X and Y.
        self.scale = out_dim
        self.mean = (
            2
            * self.scale
            * torch.rand(size=(in_dim + out_dim,), generator=self.rng).to(
                **self.tensor_parameters
            )
            - self.scale
        )
        self.sigma_corr = get_correlation(dim=in_dim + out_dim, seed=self.seed + 2).to(
            **self.tensor_parameters
        )
        self.sigma_covar = self.sigma_corr * self.scale

        # PyTorch Objects
        self.sigma_11 = self.sigma_covar[:out_dim, :out_dim]
        self.sigma_12 = self.sigma_covar[:out_dim, -in_dim:]
        self.sigma_21 = self.sigma_covar[-in_dim:, :out_dim]
        self.sigma_22 = self.sigma_covar[-in_dim:, -in_dim:]

        self.y_sigma = self.sigma_11 - torch.matmul(
            torch.matmul(self.sigma_12, torch.linalg.inv(self.sigma_22)), self.sigma_21
        )

        # Z Sigma which is heteroscedastic noise
        self.z_sigma = get_correlation(dim=out_dim, seed=self.seed + 3).to(
            **self.tensor_parameters
        )

        self.samples = dict()

    def sample_covariates(self, n_points):
        return super().sample_covariates(n_points)

    def sample_conditional(self, n_points, x):
        return super().sample_conditional(n_points, x)

    def sample_joint(self, n_points):
        mean = self.mean

        # Sample X from uniform with the same mu and corr using gaussian copula
        # Shape: N x in_dim
        self.samples["x"] = self.get_standard_uniform_samples(n_points)
        self.samples["x"] = self.correlation_to_covariance(self.samples["x"])

        # Obtain Y given X
        # Shape: N x out_dim
        self.samples["y_mean"] = (
            self.mean[: self.out_dim].view(1, self.out_dim)
            + torch.matmul(
                torch.matmul(self.sigma_12, torch.linalg.inv(self.sigma_22)).expand(
                    n_points, self.out_dim, self.in_dim
                ),
                (
                    self.samples["x"] - self.mean[-self.in_dim :].view(1, self.in_dim)
                ).unsqueeze(2),
            ).squeeze()
        )

        # Obtain Q_Covariance
        # Shape: N x out x out
        self.samples["z_sigma"] = self.z_sigma + torch.diag_embed(
            torch.sqrt(
                torch.abs(
                    self.samples["x"][:, : self.out_dim]
                    - self.mean[-self.in_dim :][: self.out_dim]
                )
            )
        )

        self.samples["q_covariance"] = self.y_sigma + self.samples["z_sigma"]

        # Obtain Q
        # Shape: N x out_dim
        old_rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed * 2)
        self.samples["q"] = torch.stack(
            [
                torch.distributions.MultivariateNormal(loc=m, covariance_matrix=c)
                .sample([1])
                .squeeze()
                for m, c in zip(self.samples["y_mean"], self.samples["q_covariance"])
            ]
        )
        torch.set_rng_state(old_rng_state)

        print(f"{self.samples['y_mean'].shape=}, {self.samples['q_covariance'].shape=}")
        t = (
            torch.distributions.MultivariateNormal(
                loc=self.samples["y_mean"][0],
                covariance_matrix=self.samples["q_covariance"][0],
            )
            .sample([1])
            .squeeze()
        )
        print(f"{t.shape=}, {self.samples['q'].shape=}")

        # self.samples['q'] = torch.stack([
        #    self.rng.multivariate_normal(m.numpy(), c.numpy()) for m, c in zip(
        #        self.samples['y_mean'], self.samples['q_covariance'])
        # ])

        return self.samples["x"], self.samples["q"]

    def get_standard_uniform_samples(self, num_samples) -> float:
        correlation = self.sigma_corr[-self.in_dim :, -self.in_dim :]

        # https://stats.stackexchange.com/questions/66610/generate-pairs-of-random-numbers-uniformly-distributed-and-correlated
        spearman_rho = 2 * torch.sin(correlation * math.pi / 6)

        old_rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        A = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.in_dim).to(**self.tensor_parameters),
            covariance_matrix=spearman_rho,
        ).sample([num_samples])
        torch.set_rng_state(old_rng_state)
        norm_ = torch.distributions.Normal(
            torch.tensor([0.0]).to(**self.tensor_parameters),
            torch.tensor([1.0]).to(**self.tensor_parameters),
        )
        U = norm_.cdf(A)

        return U

    def correlation_to_covariance(self, uniform_samples: float) -> float:
        # https://stats.stackexchange.com/questions/139804/the-meaning-of-scale-and-location-in-the-pearson-correlation-context
        # Correlation is independent of scale and we can scsale each variable independtly since uniform_b_minus_a is always positive

        means = self.mean[-self.in_dim :]
        var = self.scale

        uniform_b_minus_a = math.sqrt(12 * var)  # scalar
        uniform_b_plus_a = 2 * means  # vector: dim
        uniform_a = (uniform_b_plus_a - uniform_b_minus_a) / 2

        uniform_samples = uniform_a + (uniform_b_minus_a * uniform_samples)

        return uniform_samples


def get_correlation(dim: int, seed: int) -> float:
    # https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor

    a = 2
    old_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    A = torch.stack([torch.randn((dim,)) + torch.randn((1,)) * a for i in range(dim)])
    torch.set_rng_state(old_rng_state)
    A = A * torch.transpose(A, 0, 1)
    D_half = torch.diag(torch.diag(A) ** -0.5)
    C = D_half * A * D_half

    return C
