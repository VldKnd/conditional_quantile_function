import torch
import torch.nn as nn
from tqdm import trange
import gc
import time

from pushforward_operators.convex_potential_flow.flows import SequentialFlow, ActNorm
from pushforward_operators.convex_potential_flow.cpflows import DeepConvexFlow
from pushforward_operators.convex_potential_flow.icnn import PICNN
from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters


class ConvexPotentialFlow(PushForwardOperator, nn.Module):

    def __init__(
        self,
        response_dimension: int,
        feature_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.init_dict = dict(
            response_dimension=response_dimension,
            feature_dimension=feature_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            n_blocks=n_blocks,
        )
        self.model_information_dict = {
            "class_name": "ConvexPotentialFlow",
        }
        icnns = [
            PICNN(
                dim=response_dimension,
                dimh=hidden_dimension,
                dimc=feature_dimension,
                num_hidden_layers=number_of_hidden_layers,
                symm_act_first=True,
                softplus_type="softplus",
                zero_softplus=True,
            ) for _ in range(n_blocks)
        ]
        layers = [None] * (2 * n_blocks + 1)
        layers[0::2] = [ActNorm(response_dimension) for _ in range(n_blocks + 1)]
        layers[1::2] = [
            DeepConvexFlow(icnn, response_dimension, unbiased=False)
            for _, icnn in zip(range(n_blocks), icnns)
        ]
        self.flow = SequentialFlow(layers)

    def fit(
        self, dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters, *args, **kwargs
    ):
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)
        verbose = train_parameters.verbose

        convex_potential_flow_optimizer = torch.optim.Adam(
            self.flow.parameters(), **train_parameters.optimizer_parameters
        )
        if train_parameters.scheduler_parameters:
            convex_potential_flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                convex_potential_flow_optimizer, total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            convex_potential_flow_scheduler = None

        accumulated_loss_function = 0
        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )

        training_time_start = time.perf_counter()

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

        elapsed_training_time = time.perf_counter() - training_time_start
        training_time_per_epoch = elapsed_training_time / number_of_epochs_to_train
        self.model_information_dict["training_time"] = elapsed_training_time
        self.model_information_dict["time_per_epoch"] = training_time_per_epoch
        self.model_information_dict["number_of_epochs_to_train"
                                    ] = number_of_epochs_to_train
        self.model_information_dict["training_batch_size"] = dataloader.batch_size
        return self

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This ConvexPotentialFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            y = self.flow.reverse(u, context=x)

        return y

    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        if not self.is_fitted_:
            raise ValueError(
                "This ConvexPotentialFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            u, _ = self.flow.forward_transform(y, context=x)
        return u

    def sample_y_given_x(self, n_samples: int, X: torch.Tensor) -> torch.Tensor:
        u = torch.randn(
            n_samples,
            self.init_dict["response_dimension"],
            device=X.device,
            dtype=torch.float32,
        )
        y = self.push_u_given_x(u=u, x=X)
        return y

    def logp_cond(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This ConvexPotentialFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.flow.logp(Y, context=X)

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.flow.state_dict(),
                "model_information_dict": self.model_information_dict,
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
            y = torch.rand(
                8, self.init_dict["response_dimension"], dtype=_dtype, device=_device
            )
            x = torch.rand(
                8, self.init_dict["feature_dimension"], dtype=_dtype, device=_device
            )
        self.flow.forward_transform(y, context=x)
        self.flow.load_state_dict(data["state_dict"])
        self.model_information_dict = data.get("model_information_dict", {})
        self.init_dict.update(data)
        self.is_fitted_ = True
        return self

    @classmethod
    def load_class(
        cls, path: str, map_location: torch.device = torch.device('cpu')
    ) -> "ConvexPotentialFlow":
        data = torch.load(path, map_location=map_location)
        convex_potential_flow = cls(**data["init_dict"])
        with torch.no_grad():
            _dtype = list(convex_potential_flow.flow.parameters())[0].dtype
            _device = list(convex_potential_flow.flow.parameters())[0].device
            y = torch.rand(
                8,
                convex_potential_flow.init_dict["response_dimension"],
                dtype=_dtype,
                device=_device
            )
            x = torch.rand(
                8,
                convex_potential_flow.init_dict["feature_dimension"],
                dtype=_dtype,
                device=_device
            )
        convex_potential_flow.flow.forward_transform(y, context=x)
        convex_potential_flow.flow.load_state_dict(data["state_dict"])
        convex_potential_flow.model_information_dict = data.get(
            "model_information_dict", {}
        )
        convex_potential_flow.is_fitted_ = True
        return convex_potential_flow
