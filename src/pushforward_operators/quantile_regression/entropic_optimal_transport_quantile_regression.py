import torch.nn as nn
import torch
from tqdm import trange
from typing import Optional
from infrastructure.classes import TrainParameters
from pushforward_operators.protocol import PushForwardOperator



class EntropicOTQuantileRegression(PushForwardOperator, nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int = 100,
        number_of_hidden_layers: int = 1,
        *,
        epsilon: float = 1e-7,
        activation_function_name: str = "Softplus",
        number_of_samples_for_entropy_dual_estimation: int = 2048,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.n_entropy_samples = int(number_of_samples_for_entropy_dual_estimation)

        act_cls = getattr(nn, activation_function_name, None)
        if act_cls is None:
            raise ValueError(
                f"Activation '{activation_function_name}' is not available in torch.nn"
            )
        self._act_cls = act_cls

        self.y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

        layers = [
            nn.Linear(feature_dimension + response_dimension, hidden_dimension),
            self._act_cls(),
        ]
        for _ in range(number_of_hidden_layers):
            layers.extend([nn.Linear(hidden_dimension, hidden_dimension), self._act_cls()])
        layers.append(nn.Linear(hidden_dimension, 1))
        self.phi_potential_network = nn.Sequential(*layers)

    def warmup_Y_scaler(self, dataloader: torch.utils.data.DataLoader, num_passes: int = 1):
        """Run over the data (no grad) to populate BatchNorm running stats."""
        self.y_scaler.train()
        with torch.no_grad():
            for _ in range(num_passes):
                for _, Y in dataloader:
                    _ = self.y_scaler(Y)
        self.y_scaler.eval()

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        *_,
        **__,
    ) -> "EntropicOTQuantileRegression":
        """Fits the operator on the given data loader."""

        self.train()
        self.warmup_Y_scaler(dataloader)
        device_has_been_set = False

        num_epochs = train_parameters.number_of_epochs_to_train
        total_steps = num_epochs * len(dataloader)
        optimiser = torch.optim.AdamW(
            self.parameters(), **train_parameters.optimizer_parameters
        )
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if train_parameters.scheduler_parameters:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=total_steps, **train_parameters.scheduler_parameters
            )

        pbar = trange(1, num_epochs + 1, disable=not train_parameters.verbose, desc="Train")
        running_objective: list[float] = []

        for epoch_idx in pbar:
            for X_batch, Y_batch in dataloader:
                if not device_has_been_set:
                    self.to(X_batch.device)
                    device_has_been_set = True

                optimiser.zero_grad(set_to_none=True)

                Y_scaled = self.y_scaler(Y_batch)
                U_batch = torch.randn_like(Y_scaled)

                phi = self.phi_potential_network(torch.cat([X_batch, U_batch], dim=1))
                psi = self._estimate_entropy_dual_psi(
                    X_tensor=X_batch,
                    U_tensor=torch.randn(
                        self.n_entropy_samples,
                        Y_batch.shape[1],
                        device=Y_batch.device,
                        dtype=Y_batch.dtype,
                    ),
                    Y_tensor=Y_scaled,
                )

                objective = phi.mean() + psi.mean()
                objective.backward()
                optimiser.step()
                if scheduler is not None:
                    scheduler.step()

                if train_parameters.verbose:
                    running_objective.append(objective.item())
                    avg_last_10 = sum(running_objective[-10:]) / len(running_objective[-10:])
                    lr_str = (
                        f", LR: {scheduler.get_last_lr()[0]:.6f}" if scheduler else ""
                    )
                    pbar.set_description(
                        f"Epoch {epoch_idx} | Objective: {avg_last_10:.3f}{lr_str}"
                    )

        pbar.close()
        self.eval()
        return self

    def _estimate_entropy_dual_psi(
        self,
        *,
        X_tensor: torch.Tensor,
        U_tensor: torch.Tensor,
        Y_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Numerically stable estimation of the entropy‑dual psi.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        n = X_tensor.shape[0]
        m = U_tensor.shape[0]

        U_expanded = U_tensor.unsqueeze(0).expand(n, -1, -1)  # (n, m, d)
        X_expanded = X_tensor.unsqueeze(1).expand(-1, m, -1)  # (n, m, p)
        XU = torch.cat([X_expanded, U_expanded], dim=-1)  # (n, m, p + d)

        phi_vals = self.phi_potential_network(XU).squeeze(-1)  # (n, m)
        cost = Y_tensor @ U_tensor.T  # (n, m)

        slackness = cost - phi_vals  # (n, m)

        # Log‑sum‑exp with row‑wise stabilisation
        row_max, _ = torch.max(slackness, dim=1, keepdim=True)
        stable = (slackness - row_max) / self.epsilon
        log_mean_exp = torch.logsumexp(stable, dim=1, keepdim=True) - torch.log(
            torch.tensor(m, device=slackness.device, dtype=slackness.dtype)
        )
        psi_estimate = self.epsilon * (log_mean_exp + row_max / self.epsilon)
        return psi_estimate  # (n, 1)

    def push_forward_u_given_x(
        self, U: torch.Tensor, X: torch.Tensor, *, create_graph: bool = True
    ) -> torch.Tensor:
        """Applies the learned push‑forward ∇φ to a noise sample *U|X*.

        If you only need samples (no gradients through the result), wrap the call
        in ``with torch.no_grad():`` for memory savings.
        """
        requires_grad_backup = U.requires_grad
        U.requires_grad = True

        phi_value = self.phi_potential_network(torch.cat([X, U], dim=-1)).sum()
        grad_U = torch.autograd.grad(phi_value, U, create_graph=create_graph)[0]

        grad_U = grad_U * torch.sqrt(self.y_scaler.running_var + self.epsilon) + self.y_scaler.running_mean

        U.requires_grad = requires_grad_backup
        return grad_U

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({"state_dict": self.state_dict(), "epsilon": self.epsilon, "activation_function_name": self.activation_function_name}, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        self.epsilon = data["epsilon"]
        self.activation_function_name = data["activation_function_name"]
        return self