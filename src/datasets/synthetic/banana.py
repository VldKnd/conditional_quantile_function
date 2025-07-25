import numpy as np

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

    def sample_covariates(self, n_points):
        rng = np.random.default_rng(seed=self.seed)
        x = rng.uniform(0.5, 2.5, size=(n_points, 1))
        return x

    def sample_conditional(self, n_points, x):
        rng = np.random.default_rng(seed=self.seed)
        u = rng.normal(0, 1, size=(n_points, 2))
        y = np.concatenate(
            [
                u[:, 0:1] * x,
                u[:, 1:2] / x + (u[:, 0:1] ** 2 + x**3),
            ],
            axis=1,
        )
        return y

    def sample_joint(self, n_points):
        x = self.sample_covariates(n_points=n_points)
        y = self.sample_conditional(n_points=n_points, x=x)
        return x, y
