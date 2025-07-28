from protocols.pushforward_operator import PushForwardOperator, TrainParams
import torch
import torch.nn as nn
from tqdm import trange

class UnconstrainedOTQuantileRegression(PushForwardOperator):
    def __init__(self, input_dimension: int, embedding_dimension: int = 5, hidden_dimension: int = 100, number_of_hidden_layers: int = 1, epsilon: float = 1e-7):
        self.phi_potential_network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.Softplus(),
            *[nn.Linear(hidden_dimension, hidden_dimension), nn.Softplus()] * number_of_hidden_layers,
            nn.Linear(hidden_dimension, embedding_dimension),
            torch.nn.BatchNorm1d(num_features=embedding_dimension, affine=False, track_running_stats=True)
        )
        self.epsilon = epsilon

    def fit(self, dataloader: torch.utils.data.DataLoader, train_params: TrainParams = TrainParams(verbose=False), *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_params (TrainParams): Training parameters.
        """
        num_epochs = train_params.get("num_epochs", 100)
        lr = train_params.get("lr", 1e-3)
        _, Y_tensor = next(iter(dataloader))
        device_and_dtype_specifications = {
            "device": Y_tensor.device,
            "dtype": Y_tensor.dtype
        }
        self.phi_potential_network.to(**device_and_dtype_specifications)
        phi_potential_network_optimizer = torch.optim.Adam(self.phi_potential_network.parameters(), lr=lr)

        training_information = []
        progress_bar = trange(1, num_epochs+1, desc="Training", disable=not train_params["verbose"])

        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    self.phi_potential_network.zero_grad()
                    U_batch = torch.randn_like(Y_batch)

                    phi = self.phi_potential_network(torch.cat([X_batch, U_batch], dim=1))
                    psi = self.estimate_entropy_dual_psi(
                            X_tensor=X_batch,
                            U_tensor=torch.randn(
                                    2048, Y_batch.shape[1],
                                    **device_and_dtype_specifications
                            ),
                            Y_tensor=Y_batch
                    )

                    objective = torch.mean(phi) + torch.mean(psi)

                    objective.backward()
                    phi_potential_network_optimizer.step()

                training_information.append({
                        "objective": objective.item(),
                        "epoch_index": epoch_idx
                })

                running_mean_objective = sum([information["objective"] for information in training_information[-10:]]) / len(training_information[-10:])
                progress_bar.set_description(f"Epoch: {epoch_idx}, objective: {running_mean_objective:.3f}")

        _ = self.phi_potential_network.eval()


    def estimate_entropy_dual_psi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor, Y_tensor: torch.Tensor):
        """Estimates the entropy dual psi.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        ...

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({"feature_network.state_dict": self.feature_network.state_dict(), "epsilon": self.epsilon}, path)

    def load(self, path: str):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path)
        self.feature_network.load_state_dict(data["feature_network.state_dict"])
        self.feature_network.eval()
        self.epsilon = data["epsilon"]
        return self