import torch

from src.protocols.dataset import Dataset


class BananaDataset(Dataset):
    """
    Creating data in the form of a banana with x values distributed between 1 and 5.

    X: 1D, distributed between 1 and 5.
    Y: 2D, derived from x and random noise.
    """

    def __init__(self, seed=31337, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.set_state(self.seed)

    def sample_covariates(self, n_points):
        self.rng.set_state(self.seed)
        x = torch.rand(size=(n_points, 1), generator=self.rng) * 2 + 0.5
        return x

    def sample_conditional(self, n_points, x):
        self.rng.set_state(self.seed)
        u = torch.randn(size=(n_points, 2), generator=self.rng)
        y = torch.concatenate(
            [
                u[:, 0:1] * x,
                u[:, 1:2] / x + (u[:, 0:1] ** 2 + x**3),
            ],
            dim=1,
        )
        return y

    def sample_joint(self, n_points):
        x = self.sample_covariates(n_points=n_points)
        y = self.sample_conditional(n_points=n_points, x=x)
        return x, y
