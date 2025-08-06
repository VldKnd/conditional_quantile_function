import math
import numpy as np
import torch

from datasets.protocol import Dataset


class EightGaussians(Dataset):

    def sample_covariates(self, n_points: int) -> torch.Tensor:
        x = torch.rand(size=(n_points, 1)) * 2 + 0.5
        return x.to(**self.tensor_prameters)

    def sample_conditional(self, n_points, x):
        n_x, d = x.shape
        print(f"{x.shape=}")
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / math.sqrt(2), 1. / math.sqrt(2)),
                   (1. / math.sqrt(2), -1. / math.sqrt(2)),
                   (-1. / math.sqrt(2), 1. / math.sqrt(2)),
                   (-1. / math.sqrt(2), -1. / math.sqrt(2))]
        centers = torch.tensor(data=[(scale * x, scale * y)
                                     for x, y in centers],
                               **self.tensor_prameters)

        idx = np.random.randint(8, size=(n_x, n_points))
        print(idx[:5])

        selected_centers = centers[idx]

        print(f"{selected_centers.shape=}")

        points = torch.randn(size=(n_x, n_points, 2)) * 0.5

        y = (points + selected_centers) * torch.abs(x).reshape(-1, 1,
                                                               1) / 1.414

        return y


"""         indices = []
        for i in range(n_points):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            y.append(point)
            indices.append(idx)
        y = np.array(y, dtype='float32')
        y /= 1.414
        y_tensor = torch.from_numpy(y) * torch.abs(x)
        return y_tensor
 """
