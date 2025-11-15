# Variational Encoder Kingma and Welling https://arxiv.org/abs/1906.02691

import time
import torch
from tqdm import trange
from pushforward_operators import PushForwardOperator
from infrastructure.classes import TrainParameters


class VariationalEncoder(PushForwardOperator, torch.nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        latent_space_regularizer: float = 0.01,
    ):
        super().__init__()

        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "latent_space_regularizer": latent_space_regularizer,
        }
        self.model_information_dict = {
            "class_name": "VariationalEncoder",
        }
        self.activation_function = torch.nn.SiLU()
        self.latent_space_regularizer = latent_space_regularizer

        encoder_hidden_layers = []
        for _ in range(number_of_hidden_layers):
            encoder_hidden_layers.append(
                torch.nn.Linear(hidden_dimension, hidden_dimension)
            )
            encoder_hidden_layers.append(self.activation_function)

        decoder_hidden_layers = []
        for _ in range(number_of_hidden_layers):
            decoder_hidden_layers.append(
                torch.nn.Linear(hidden_dimension, hidden_dimension)
            )
            decoder_hidden_layers.append(self.activation_function)

        self.encoder_network = torch.nn.Sequential(
            torch.nn.Linear(response_dimension, hidden_dimension),
            self.activation_function, *encoder_hidden_layers,
            torch.nn.Linear(hidden_dimension, 2 * response_dimension)
        )

        self.decoder_network = torch.nn.Sequential(
            torch.nn.Linear(response_dimension, hidden_dimension),
            self.activation_function, *decoder_hidden_layers,
            torch.nn.Linear(hidden_dimension, response_dimension)
        )

    def make_progress_bar_message(
        self,
        training_information: list[dict],
        epoch_idx: int,
        last_learning_rate: str = None
    ):
        running_mean_objective = sum(
            [
                information["potential_loss"]
                for information in training_information[-10:]
            ]
        ) / len(training_information[-10:])

        return (f"Epoch: {epoch_idx}, "
                f"Objective: {running_mean_objective:.3f}") + \
        (
            f", LR: {last_learning_rate[0]:.6f}"
            if last_learning_rate is not None
            else ""
        )

    def fit(
        self, dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters, *args, **kwargs
    ):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_parameters (TrainParameters): Training parameters.
        """
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        verbose = train_parameters.verbose
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)

        encoder_network_optimizer = torch.optim.AdamW(
            self.encoder_network.parameters(), **train_parameters.optimizer_parameters
        )

        decoder_network_optimizer = torch.optim.AdamW(
            self.decoder_network.parameters(), **train_parameters.optimizer_parameters
        )

        if train_parameters.scheduler_parameters:
            encoder_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                encoder_network_optimizer, total_number_of_optimizer_steps // 2,
                **train_parameters.scheduler_parameters
            )
            decoder_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                decoder_network_optimizer, total_number_of_optimizer_steps // 2,
                **train_parameters.scheduler_parameters
            )
        else:
            encoder_network_scheduler = None
            decoder_network_scheduler = None

        training_information = []
        training_information_per_epoch = []

        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )

        for epoch_idx in progress_bar:
            start_of_epoch = time.perf_counter()
            potential_losses_per_epoch = []

            for batch_index, (X_batch, Y_batch) in enumerate(dataloader):

                encoder_network_optimizer.zero_grad()
                decoder_network_optimizer.zero_grad()

                encoder_input = Y_batch

                # q(z|x,y)
                mu_q, logvar_q = self.encoder_network(encoder_input).chunk(2, dim=1)
                logvar_q = logvar_q.clamp(-10, 10)
                std_q = torch.exp(0.5 * logvar_q)
                z = mu_q + torch.randn_like(std_q) * std_q

                decoder_input = z
                reconstruction = self.decoder_network(decoder_input)

                # losses
                recon = (reconstruction - Y_batch).pow(2).sum(dim=1).mean()

                kl = 0.5 * torch.sum(
                    -1 - logvar_q + mu_q.pow(2) + torch.exp(logvar_q), dim=-1
                ).mean()

                loss = recon + self.latent_space_regularizer * kl
                loss.backward()

                if batch_index % 2 == 1:
                    encoder_network_optimizer.step()
                    if encoder_network_scheduler is not None:
                        encoder_network_scheduler.step()
                else:
                    decoder_network_optimizer.step()
                    if decoder_network_scheduler is not None:
                        decoder_network_scheduler.step()

                potential_losses_per_epoch.append(loss.item())

                training_information.append(
                    {
                        "potential_loss":
                        loss.item(),
                        "batch_index":
                        batch_index,
                        "epoch_index":
                        epoch_idx,
                        "time_elapsed_since_last_epoch":
                        time.perf_counter() - start_of_epoch,
                    }
                )

                if verbose:
                    last_learning_rate = (
                        decoder_network_scheduler.get_last_lr()
                        if decoder_network_scheduler is not None else None
                    )

                    progress_bar_message = self.make_progress_bar_message(
                        training_information=training_information,
                        epoch_idx=epoch_idx,
                        last_learning_rate=last_learning_rate
                    )

                    progress_bar.set_description(progress_bar_message)

            training_information_per_epoch.append(
                {
                    "potential_loss":
                    torch.mean(torch.tensor(potential_losses_per_epoch)),
                    "epoch_training_time": time.perf_counter() - start_of_epoch
                }
            )

        progress_bar.close()

        self.model_information_dict["number_of_epochs_to_train"
                                    ] = number_of_epochs_to_train
        self.model_information_dict["training_batch_size"] = dataloader.batch_size
        self.model_information_dict["training_information"
                                    ] = training_information_per_epoch
        return self

    @torch.no_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # encoder_input = torch.cat([y, x], dim=1)
        encoder_input = y
        latent_parameters = self.encoder_network(encoder_input)
        mu, _ = latent_parameters.chunk(2, dim=1)
        return mu.detach()

    @torch.no_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # decoder_input = torch.cat([u, x], dim=1)
        decoder_input = u
        y_reconstructed = self.decoder_network(decoder_input)
        return y_reconstructed.detach()

    def save(self, path: str):
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.state_dict(),
                "model_information_dict": self.model_information_dict,
            }, path
        )

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self

    @classmethod
    def load_class(
        cls, path: str, map_location: torch.device = torch.device('cpu')
    ) -> "VariationalEncoder":
        data = torch.load(path, map_location=map_location)
        neural_quantile_regression = cls(**data["init_dict"])
        neural_quantile_regression.load_state_dict(data["state_dict"])
        neural_quantile_regression.model_information_dict = data.get(
            "model_information_dict", {}
        )
        return neural_quantile_regression
