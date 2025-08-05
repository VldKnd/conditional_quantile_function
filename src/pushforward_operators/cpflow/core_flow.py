import torch
import torch.nn as nn
from tqdm import trange
import gc

from pushforward_operators.cpflow.flows import SequentialFlow, ActNorm
from pushforward_operators.cpflow.cpflows import DeepConvexFlow
from pushforward_operators.cpflow.icnn import PICNN
from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters


class CPFlow(PushForwardOperator, nn.Module):
    def __init__(
        self,
        response_dimension: int,
        feature_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.config = dict(
            response_dimension=response_dimension,
            feature_dimension=feature_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            n_blocks=n_blocks,
        )

        icnns = [
            PICNN(
                dim=response_dimension,
                dimh=hidden_dimension,
                dimc=feature_dimension,
                num_hidden_layers=number_of_hidden_layers,
                symm_act_first=True,
                softplus_type="gaussian_softplus",
                zero_softplus=True,
            )
            for _ in range(n_blocks)
        ]
        layers = [None] * (2 * n_blocks + 1)
        layers[0::2] = [ActNorm(response_dimension) for _ in range(n_blocks + 1)]
        layers[1::2] = [
            DeepConvexFlow(icnn, response_dimension, unbiased=False)
            for _, icnn in zip(range(n_blocks), icnns)
        ]
        self.flow = SequentialFlow(layers)

    def fit(self, dataloader: torch.utils.data.DataLoader, train_parameters: TrainParameters, *args, **kwargs):
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)
        verbose = train_parameters.verbose

        convex_potential_flow_optimizer = torch.optim.Adam(self.flow.parameters(), **train_parameters.optimizer_parameters)
        if train_parameters.scheduler_parameters:
            convex_potential_flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(convex_potential_flow_optimizer, total_number_of_optimizer_steps, **train_parameters.scheduler_parameters)
        else:
            convex_potential_flow_scheduler = None


        accumulated_loss_function = 0
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)

        for _ in progress_bar:
            for cond, y in dataloader:
                loss = -self.flow.logp(y, context=cond).mean()
                convex_potential_flow_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.flow.parameters(), max_norm=10
                ).item()

                convex_potential_flow_optimizer.step()
                if convex_potential_flow_scheduler is not None:
                    convex_potential_flow_scheduler.step()

                accumulated_loss_function += loss.item()

                del loss
                gc.collect()
                torch.clear_autocast_cache()

                if verbose:
                    progress_bar.set_description(
                        (
                            f"Epoch: {progress_bar.n + 1}, Loss: {accumulated_loss_function / (progress_bar.n + 1):.3f}"
                        ) + \
                        (
                            f", LR: {convex_potential_flow_scheduler.get_last_lr()[0]:.6f}"
                            if convex_potential_flow_scheduler is not None
                            else ""
                        )
                    )

        progress_bar.close()
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
        u = torch.randn(
            n_samples,
            self.config["response_dimension"],
            device=X.device,
            dtype=torch.float32,
        )
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
        torch.save(
            {
                "state_dict": self.flow.state_dict(),
                **self.config,
            },
            path,
        )

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path, map_location=map_location)
        with torch.no_grad():
            _dtype = list(self.flow.parameters())[0].dtype
            _device = list(self.flow.parameters())[0].device
            y = torch.rand(8, self.config["response_dimension"], dtype=_dtype, device=_device)
            x = torch.rand(8, self.config["feature_dimension"], dtype=_dtype, device=_device)
        self.flow.forward_transform(y, context=x)
        self.flow.load_state_dict(data["state_dict"])
        self.config.update(data)
        self.is_fitted_ = True
        return self
