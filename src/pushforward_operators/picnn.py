import torch
from torch import nn
from torch import Tensor

class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(-1)
        return nn.functional.linear(x, torch.nn.functional.softplus(self.weight), self.bias) * gain

class PICNN(nn.Module):
    def __init__(self,
            feature_dimension: int,
            response_dimension: int,
            hidden_dimension: int,
            number_of_hidden_layers: int,
            activation_function_name: str,
            output_dimension: int = 1,
            *args, 
            **kwargs,
        ):
        super().__init__()
        x_dimension = feature_dimension
        y_dimension = response_dimension
        u_dimension, z_dimension = hidden_dimension, hidden_dimension
        
        # Activations:
        self.number_of_hidden_layers = number_of_hidden_layers
        self.z_activation = nn.Softplus()
        self.u_activation = getattr(nn, activation_function_name)()
        self.positive_activation = nn.Softplus()

        # First layer
        self.first_linear_layer_tilde = nn.Linear(x_dimension, u_dimension)
        self.first_linear_layer_yu = nn.Linear(x_dimension, y_dimension)
        self.first_linear_layer_y = nn.Linear(y_dimension, z_dimension, bias=False)
        self.first_linear_layer_u = nn.Linear(x_dimension, z_dimension, bias=False)

        # Iterations:
        self.linear_layer_tilde = nn.ModuleList([
                nn.Linear(u_dimension, u_dimension)
                for _ in range(number_of_hidden_layers)
        ])
        self.linear_layer_uz = nn.ModuleList([
            nn.Linear(u_dimension, z_dimension)
                for _ in range(number_of_hidden_layers)
        ])
        self.linear_layer_z = nn.ModuleList([
            PosLinear(z_dimension, z_dimension)
                for _ in range(number_of_hidden_layers)
        ])
        self.linear_layer_uy = nn.ModuleList([
            nn.Linear(u_dimension, y_dimension)
                for _ in range(number_of_hidden_layers)
        ])
        self.linear_layer_y = nn.ModuleList([
            nn.Linear(y_dimension, z_dimension, bias=False)
                for _ in range(number_of_hidden_layers)
        ])
        self.linear_layer_u = nn.ModuleList([
            nn.Linear(u_dimension, z_dimension, bias=False)
                for _ in range(number_of_hidden_layers)
        ])

        # Last layer:
        self.last_linear_layer_uz = nn.Linear(u_dimension, z_dimension)
        self.last_linear_layer_z = PosLinear(z_dimension, output_dimension)
        self.last_linear_layer_uy = nn.Linear(u_dimension, y_dimension)
        self.last_linear_layer_y = nn.Linear(y_dimension, output_dimension, bias=False)
        self.last_linear_layer_u = nn.Linear(u_dimension, output_dimension, bias=False)


    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        # First layer:
        u = self.u_activation(
            self.first_linear_layer_tilde(condition)
        )
        z = self.z_activation(
            self.first_linear_layer_y(
                tensor * self.first_linear_layer_yu(condition)
            ) +
            self.first_linear_layer_u(condition)
        )

        # Iterations:
        for iteration_number in range(self.number_of_hidden_layers):
            u, z = (
                self.u_activation(
                    self.linear_layer_tilde[iteration_number](u)
                ),
                self.z_activation(
                    self.linear_layer_z[iteration_number](
                        z * self.positive_activation((self.linear_layer_uz[iteration_number](u)))
                    ) + \
                    self.linear_layer_y[iteration_number](
                        tensor * self.linear_layer_uy[iteration_number](u)
                    ) + \
                    self.linear_layer_u[iteration_number](u)
                )
            )

        output = self.last_linear_layer_z(
            z * self.positive_activation(self.last_linear_layer_uz(u))
        ) + \
        self.last_linear_layer_y(
            tensor * self.last_linear_layer_uy(u)
        ) + \
        self.last_linear_layer_u(u)

        return output

class PISCNN(PICNN):
    """Strongly convex variant of PICNN.
    alpha: regularization parameter
    """
    def __init__(self, *args, **kwargs):
        super(PISCNN, self).__init__(*args, **kwargs)
        self.log_alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        output = super().forward(condition, tensor)
        return output + 0.5 * torch.exp(self.log_alpha) * torch.norm(tensor, dim=-1, keepdim=True)**2

class FFNN(nn.Module):
    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        activation_function_name: str,
        *args, 
        **kwargs,
    ):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.))
        self.activation_function_name = activation_function_name
        self.activation_function = getattr(nn, activation_function_name)()

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.potential_network = nn.Sequential(
            nn.Linear(response_dimension + feature_dimension, hidden_dimension),
            self.activation_function,
            *hidden_layers,
            nn.Linear(hidden_dimension, 1)
        )

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        input_tensor = torch.cat([condition, tensor], dim=-1)
        output_tensor = self.potential_network(input_tensor)
        return output_tensor

class SCFFNN(FFNN):
    """Feed Forward Neural Network with added norm to enforce strong convexity.
    alpha: regularization parameter
    """
    def __init__(self, *args, **kwargs):
        super(SCFFNN, self).__init__(*args, **kwargs)
        self.log_alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        output = super().forward(condition, tensor)
        return output + 0.5 * torch.exp(self.log_alpha) * torch.norm(tensor, dim=-1, keepdim=True)**2

network_name_to_network_type: dict[str, nn.Module] = {
    "SCFFNN":SCFFNN,
    "FFNN":FFNN,
    "PICNN":PICNN,
    "PISCNN":PISCNN
}
