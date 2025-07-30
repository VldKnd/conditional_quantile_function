import torch
import numpy as np
from tqdm.auto import tqdm
import gc

from src.cpflow.flows import SequentialFlow, ActNorm
from src.cpflow.cpflows import DeepConvexFlow
from src.cpflow.icnn import PICNN
from src.protocols.pushforward_operator import PushForwardOperator, TrainParams


class CPFlow(PushForwardOperator):
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

    def fit(self, train_loader: torch.utils.data.DataLoader, train_params: TrainParams, *args, **kwargs):
        num_epochs = train_params.get("num_epochs", 100)
        lr = train_params.get("lr", 1e-3)
        print_every = train_params.get("print_every", 10)

        self.flow = self.flow.to(self.device)

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

    def push_forward_u_given_x(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            y = self.flow.reverse(U, context=X)

        return y

    def push_backward_y_given_x(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            u, _ = self.flow.forward_transform(Y, context=X)
        return u

    def sample_y_given_x(self, n_samples: int, X: torch.Tensor) -> torch.Tensor:
        u = torch.randn(n_samples, self.dim_y, device=self.device, dtype=torch.float32)
        y = self.push_forward_u_given_x(u, X)
        return y

    def logp_cond(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.flow.logp(Y, context=X)
    
    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        ...

    def load(self, path: str) -> "PushForwardOperator":
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        ...
