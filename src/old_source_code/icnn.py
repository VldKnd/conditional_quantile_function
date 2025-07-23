import torch
from torch import nn

class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""

    def __init__(self, input_dimension: int, hidden_dimension: int, num_hidden_layers: int, output_dimension: int):
        super().__init__()

        Wzs = []
        Wzs.append(nn.Linear(input_dimension, hidden_dimension))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(torch.nn.Linear(hidden_dimension, hidden_dimension, bias=False))
        Wzs.append(torch.nn.Linear(hidden_dimension, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(input_dimension, hidden_dimension))
        Wxs.append(nn.Linear(input_dimension, output_dimension, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)