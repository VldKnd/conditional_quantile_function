import torch.nn as nn
import torch
from tqdm import trange
from typing import Literal
from infrastructure.classes import TrainParameters
from pushforward_operators.protocol import PushForwardOperator
import ot
from geomloss import SamplesLoss
from pushforward_operators.picnn import FFNN

def optimal_transport_plan(ground_truth: torch.Tensor, approximation: torch.Tensor) -> torch.Tensor:
    """
    Computes the Wasserstein distance between two sets of points.

    Args:
        ground_truth (torch.Tensor): The ground truth points.
        approximation (torch.Tensor): The approximation points.

    Returns:
        float: The Wasserstein distance between the two sets of points.
    """
    return ot.solve_sample(X_a=ground_truth, X_b=approximation).plan

class MongeMapNetwork(nn.Module):
    def __init__(self,
            feature_dimension: int,
            response_dimension: int,
            hidden_dimension: int,
            number_of_hidden_layers: int,
            activation_function_name: str = "Softplus",
    ):
        super().__init__()
        self.monge_map_network = FFNN(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_function_name=activation_function_name,
            output_dimension=response_dimension
        )

    def pushforward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return tensor - self.monge_map_network(condition, tensor)

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.monge_map_network(tensor, condition)

class MongeGapTransport(PushForwardOperator, nn.Module):
    def __init__(self,
            feature_dimension: int,
            response_dimension: int,
            hidden_dimension: int,
            number_of_hidden_layers: int,
            activation_function_name: str = "Softplus",
            potential_to_estimate_with_neural_network: Literal["y", "u"] = "y",
        ):
        super().__init__()
        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "activation_function_name": activation_function_name,
            "potential_to_estimate_with_neural_network":potential_to_estimate_with_neural_network
        }

        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)
        self.X_scaler = nn.BatchNorm1d(feature_dimension, affine=False)
        self.potential_to_estimate_with_neural_network = potential_to_estimate_with_neural_network 
        self.monge_map_network = MongeMapNetwork(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_function_name=activation_function_name
        )
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",   
            p=2,
            blur=0.05,
            scaling=0.9,
            debias=True
        )
                
    def warmup_scalers(self, dataloader: torch.utils.data.DataLoader):
        """Run over the data (no grad) to populate BatchNorm running stats."""
        self.X_scaler.train(), self.Y_scaler.train()

        with torch.no_grad():
            for X, Y in dataloader:
                _, _ = self.Y_scaler(Y), self.X_scaler(X)

        self.X_scaler.eval(), self.Y_scaler.eval()

    def make_progress_bar_message(self, training_information: list[dict], epoch_idx:int, last_learning_rate: float | None):
        running_mean_objective = sum([information["objective"] for information in training_information[-10:]]) / len(training_information[-10:])
                        
        return  (
            f"Epoch: {epoch_idx}, "
            f"Objective: {running_mean_objective:.3f}"
        ) + \
        (
            f", LR: {last_learning_rate[0]:.6f}"
            if last_learning_rate is not None
            else ""
        )

    def fit(self, dataloader: torch.utils.data.DataLoader, train_parameters: TrainParameters, *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_parameters (TrainParameters): Training parameters.
        """
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        verbose = train_parameters.verbose
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)
        monge_map_network_optimizer = torch.optim.AdamW(
            params=self.monge_map_network.parameters(),
            **train_parameters.optimizer_parameters
        )
        if train_parameters.scheduler_parameters:
            monge_map_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=monge_map_network_optimizer, 
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            monge_map_network_scheduler = None

        training_information = []
        self.warmup_scalers(dataloader=dataloader)
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)
        
        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    X_scaled = self.X_scaler(X_batch)
                    Y_scaled = self.Y_scaler(Y_batch)
                    U_batch = torch.randn_like(Y_scaled)

                    Y_pushforward = self.monge_map_network.pushforward(Y_scaled, X_scaled)

                    fitting_cost = self.sinkhorn(U_batch, Y_pushforward)
                    c_optimality_cost = (
                        torch.mean(
                            torch.norm(Y_scaled - Y_pushforward, dim=1)**2
                        ) -  self.sinkhorn(Y_scaled, Y_pushforward)
                    )

                    # jvp_vector, vjp_vector = torch.randn_like(U_batch), torch.randn_like(U_batch)

                    # _, monge_map_network_jvp = torch.autograd.functional.jvp(
                    #     lambda y: self.monge_map_network(y, X_scaled),
                    #     Y_scaled, jvp_vector, create_graph=True
                    # )

                    # _, monge_map_network_vjp = torch.autograd.functional.vjp(
                    #     lambda y: self.monge_map_network(y, X_scaled),
                    #     Y_scaled, vjp_vector, create_graph=True
                    # )

                    # jaccobian_cost = torch.mean(torch.norm(monge_map_network_jvp - monge_map_network_vjp, dim=1))

                    monge_map_network_optimizer.zero_grad()
                    monge_map_network_objective = fitting_cost + 0.1 * c_optimality_cost
                    monge_map_network_objective.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.monge_map_network.parameters(), max_norm=10)
                    monge_map_network_optimizer.step()
                    
                    if monge_map_network_scheduler is not None:
                        monge_map_network_scheduler.step()

                    if verbose: 

                        training_information.append({
                                "objective": monge_map_network_objective.item(),
                                "epoch_index": epoch_idx
                        })

                        last_learning_rate = (
                            monge_map_network_scheduler.get_last_lr() 
                            if monge_map_network_scheduler is not None
                            else None
                        )
                        
                        progress_bar_message = self.make_progress_bar_message(
                            training_information=training_information,
                            epoch_idx=epoch_idx,
                            last_learning_rate=last_learning_rate
                        )

                        progress_bar.set_description(progress_bar_message)

        progress_bar.close()
        return self
    
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        y_scaled, x_scaled = self.Y_scaler(y), self.X_scaler(x)
        return self.monge_map_network.pushforward(y_scaled, x_scaled)
    
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        raise NotImplementedError("Not implemented")

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({
            "init_dict": self.init_dict,
            "state_dict": self.state_dict(),
            "class_name":"MongeGapTransport"
        }, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self
    
    @classmethod
    def load_class(cls, path: str, map_location: torch.device = torch.device('cpu')) -> "MongeGapTransport":
        data = torch.load(path, map_location=map_location)
        monge_gap_transport = cls(**data["init_dict"])
        monge_gap_transport.load_state_dict(data["state_dict"])
        return monge_gap_transport