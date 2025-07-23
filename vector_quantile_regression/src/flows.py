import sys
import torch
import numpy as np
from tqdm.auto import tqdm
import gc

sys.path.insert(0, "./third_party/cp-flow")
from lib.flows import SequentialFlow, DeepConvexFlow, ActNorm
from lib.icnn import PICNN


class CPFlow:
    def __init__(
        self,
        dim_y: int,
        dim_cond: int,
        hidden_dim: int,
        num_hidden_layers: int,
        n_blocks: int,
        device: str = "cuda:0",
    ):
        self.device = device
        self.dim_y = dim_y
        icnns = [
            PICNN(
                dim=dim_y,
                dimh=hidden_dim,
                dimc=dim_cond,
                num_hidden_layers=num_hidden_layers,
                symm_act_first=True,
                softplus_type="gaussian_softplus",
                zero_softplus=True,
            )
            for _ in range(n_blocks)
        ]
        layers = [None] * (2 * n_blocks + 1)
        layers[0::2] = [ActNorm(dim_y) for _ in range(n_blocks + 1)]
        layers[1::2] = [
            DeepConvexFlow(icnn, dim_y, unbiased=False)
            for _, icnn in zip(range(n_blocks), icnns)
        ]
        self.flow = SequentialFlow(layers)

    def fit(self, scores_cal: np.ndarray, train_params: dict):
        num_epochs = train_params.get("num_epochs", 100)
        batch_size = train_params.get("batch_size", 128)
        lr = train_params.get("lr", 1e-3)
        print_every = train_params.get("print_every", 10)

        self.flow = self.flow.to(self.device)

        train_loader = torch.utils.data.DataLoader(
            torch.tensor(scores_cal, dtype=torch.float32, device=self.device),
            batch_size=batch_size,
            shuffle=True,
        )

        optim = torch.optim.Adam(self.flow.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, num_epochs * len(train_loader), eta_min=0
        )

        loss_acc = 0
        t = 0

        self.flow.train()
        for _ in tqdm(range(num_epochs)):
            for cond, y in train_loader:
                y = y.to(self.device)
                cond = cond.to(self.device)

                loss = -self.flow.logp(y, context=cond).mean()
                optim.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.flow.parameters(), max_norm=10
                ).item()

                optim.step()
                sch.step()

                loss_acc += loss.item()
                del loss
                gc.collect()
                torch.clear_autocast_cache()

                t += 1
                if t == 1:
                    print("init loss:", loss_acc)
                if t % print_every == 0:
                    print(t, loss_acc / print_every)
        self.is_fitted_ = True
        return self

    def reverse_transform(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            u = self.flow.reverse(y, context=cond)

        return u

    def forward_transform(self, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            y = self.flow.forward_transform(u, context=cond)
        return y

    def sample_y(self, n_samples: int, cond: torch.Tensor) -> torch.Tensor:
        u = torch.randn(n_samples, self.dim_y, device=self.device, dtype=torch.float32)
        y = self.forward_transform(u, cond)
        return y

    def logp_cond(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.flow.logp(y, context=cond)
