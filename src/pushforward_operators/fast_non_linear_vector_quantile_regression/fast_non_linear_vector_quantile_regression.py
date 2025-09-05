from scipy.stats import f
from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
from tqdm import trange
import torch
import torch.nn as nn
from typing import List


class FastNonLinearVectorQuantileRegression(PushForwardOperator):

    def __init__(
        self, feature_dimension: int, response_dimension: int, hidden_dimension: int,
        number_of_hidden_layers: int, embedding_dimension: int, epsilon: float, *args,
        **kwargs
    ):
        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "embedding_dimension": embedding_dimension,
            "epsilon": epsilon,
        }
        self.feature_network = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension), nn.Softplus(),
            *[nn.Linear(hidden_dimension, hidden_dimension),
              nn.Softplus()] * number_of_hidden_layers,
            nn.Linear(hidden_dimension, embedding_dimension),
            torch.nn.BatchNorm1d(
                num_features=embedding_dimension,
                affine=False,
                track_running_stats=True
            )
        )
        self.fitted = False
        self.embedding_dimension = embedding_dimension
        self.epsilon = epsilon
        self.response_dimension = response_dimension
        self.b_u = None
        self.phi = None

    def make_multidimensional_grid(
        self, number_of_dimensions, number_of_points_per_dimension
    ):
        linespaces = [
            torch.linspace(0, 1, number_of_points_per_dimension)
            for _ in range(number_of_dimensions)
        ]
        meshgrid_of_points = torch.meshgrid(*linespaces)
        return torch.stack(meshgrid_of_points, dim=-1)

    def make_progress_bar_message(
        self, training_information: list[dict], epoch_idx: int,
        last_learning_rate: float | None
    ):
        running_mean_objective = sum(
            [information["objective"] for information in training_information[-10:]]
        ) / len(training_information[-10:])

        return f"Epoch: {epoch_idx}, Objective: {running_mean_objective:.3f}, LR: {last_learning_rate:.6f}"

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device and dtype.
        """
        self.feature_network.to(*args, **kwargs)

        if self.phi is not None:
            self.phi = self.phi.to(*args, **kwargs)
        if self.b_u is not None:
            self.b_u = self.b_u.to(*args, **kwargs)

        return self

    def train(self):
        """
        Sets the model to training mode.
        """
        self.feature_network.train()

    def eval(self):
        """
        Sets the model to evaluation mode.
        """
        self.feature_network.eval()

    def fit(
        self, dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters, *args, **kwargs
    ):
        """Fits the pushforward operator to the data.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
            verbose (bool): Whether to print verbose output.
        """

        dataloader.shuffle = False
        Y_sample = next(iter(dataloader))[1]
        U_tensor = self.make_multidimensional_grid(
            number_of_dimensions=self.response_dimension,
            number_of_points_per_dimension=int(1024**(1 / self.response_dimension))
        ).to(Y_sample)

        psi_tensor = torch.full(size=(len(dataloader.dataset), ), fill_value=0.1)
        psi_tensor = psi_tensor.to(U_tensor)
        psi_tensor.requires_grad = True

        b_tensor = torch.zeros(*(U_tensor.shape[:-1] + (self.embedding_dimension, )))
        b_tensor = b_tensor.to(U_tensor)
        b_tensor.requires_grad = True

        total_number_of_optimizer_steps = train_parameters.number_of_epochs_to_train

        self.feature_network.to(Y_sample.device)
        self.feature_network.train()

        network_optimizer = torch.optim.AdamW(
            [dict(params=self.feature_network.parameters())],
            **train_parameters.optimizer_parameters
        )
        b_psi_optimizer = torch.optim.AdamW(
            [dict(params=[b_tensor, psi_tensor])],
            **train_parameters.optimizer_parameters
        )

        if train_parameters.scheduler_parameters:
            network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                network_optimizer, total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
            b_psi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                b_psi_optimizer, total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            network_scheduler = None
            b_psi_scheduler = None

        training_information = []

        progress_bar = trange(
            1,
            train_parameters.number_of_epochs_to_train + 1,
            desc="Training",
            disable=not train_parameters.verbose
        )

        for epoch_idx in progress_bar:
            psi_batch_index = 0
            for X_batch, Y_batch in dataloader:
                psi_batch = psi_tensor[psi_batch_index:psi_batch_index + len(X_batch)]
                psi_batch_index += len(X_batch)

                b_psi_optimizer.zero_grad()
                network_optimizer.zero_grad()

                psi_batch_reshaped = psi_batch.reshape(
                    *([1] * self.response_dimension), -1
                )

                phi_batch = self.epsilon * torch.logsumexp(
                    (
                        U_tensor @ Y_batch.T -
                        b_tensor @ self.feature_network(X_batch).T - psi_batch_reshaped
                    ) / self.epsilon,
                    dim=-1
                )

                objective = torch.mean(psi_batch) + torch.mean(phi_batch)

                objective.backward()
                b_psi_optimizer.step()
                network_optimizer.step()

                if network_scheduler is not None and b_psi_scheduler is not None:
                    network_scheduler.step()
                    b_psi_scheduler.step()

                    if train_parameters.verbose:
                        training_information.append(
                            {
                                "objective": objective.item(),
                                "epoch_index": epoch_idx
                            }
                        )

                        progress_bar_message = self.make_progress_bar_message(
                            training_information=training_information,
                            epoch_idx=epoch_idx,
                            last_learning_rate=network_scheduler.get_last_lr()[0]
                        )

                        progress_bar.set_description(progress_bar_message)

        progress_bar.close()
        assert False
        with torch.no_grad():
            log_phi_tensor = torch.zeros_like(Y_tensor, **device_type_and_specification)

            for i in range(0, X_tensor.shape[0], batch_size):
                X_batch = X_tensor[i:i + batch_size]
                Y_batch = Y_tensor[i:i + batch_size]
                psi_batch = psi_tensor[i:i + batch_size]
                log_phi_tensor[i:i + batch_size] = (
                    U_tensor @ Y_batch.T - b_tensor @ self.feature_network(X_batch).T -
                    psi_batch.reshape(1, -1)
                ) / self.epsilon

            phi_tensor = self.epsilon * torch.logsumexp(
                log_phi_tensor, dim=1, keepdim=True
            )

        # self.phi = phi_tensor.detach()
        # self.b_u = b_tensor.detach()
        # self.u = U_tensor.detach()
        # self.feature_network.zero_grad()
        # self.feature_network.eval()
        # progress_bar.close()
        # self.fitted = True
        # return self

    @torch.inference_mode()
    def push_forward_u_given_x(
        self,
        U: torch.Tensor,
        X: torch.Tensor,
        batch_size: int = 1_024,
        k: int = 10,
    ) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        for name in ["b_u", "phi", "u"]:
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))

        self.feature_network.to(device).to(dtype)

        outputs: List[torch.Tensor] = []

        for i in range(0, X.shape[0], batch_size):
            X_b = X[i:i + batch_size]
            U_b = U[i:i + batch_size]

            potential = (self.b_u @ self.feature_network(X_b).T).add_(self.phi)

            grads = self._estimate_gradients_knn(
                u_samples=self.u,
                f_samples=potential,
                query_points=U_b,
                k=k,
            )
            outputs.append(grads)

        return torch.cat(outputs, dim=0)

    def _estimate_gradients_knn(
        self,
        u_samples: torch.Tensor,
        f_samples: torch.Tensor,
        query_points: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        N, d = u_samples.shape

        if k < d + 1:
            raise ValueError(f"k must be ≥ d+1 (got k={k}, d={d})")
        if k > N:
            raise ValueError(f"k ({k}) > number of samples ({N})")

        distances_sq = torch.cdist(query_points, u_samples, p=2).pow_(2)
        knn_dists, knn_idx = torch.topk(distances_sq, k=k, largest=False)

        neighbours_u = u_samples[knn_idx]
        neighbours_f = f_samples.T.gather(1, knn_idx)

        ones = torch.ones_like(neighbours_f)[..., None]
        A = torch.cat((neighbours_u, ones), dim=-1)
        b = neighbours_f.unsqueeze(-1)

        betas = torch.linalg.lstsq(A, b).solution.squeeze(-1)
        return betas[..., :d]

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save(
            {
                "u": self.u,
                "b_u": self.b_u,
                "phi": self.phi,
                "feature_network.state_dict": self.feature_network.state_dict()
            }, path
        )

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path, map_location=map_location)
        self.u = data["u"]
        self.b_u = data["b_u"]
        self.phi = data["phi"]
        self.feature_network.load_state_dict(data["feature_network.state_dict"])
        self.feature_network.eval()
        self.fitted = True
        return self
