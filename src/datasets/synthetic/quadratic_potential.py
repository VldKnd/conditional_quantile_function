import torch
from torch import nn
import torch.nn.functional as F

class ConvexQuadraticPotential(nn.Module):
    """
    f(x,u) = sum_b softplus( [x,u]^T A_b [x,u] + b_b^T [x,u] + c_b )
    with strict convexity in u for every fixed x.

    We enforce: A_b = [[S_xx, A_xu],
                       [A_xu^T, A_uu]]
    where A_uu = L L^T + eps*I  (PD), ensuring strict convexity in u.
    """
    def __init__(self,
            number_of_functions: int,
            response_size: int,
            covariate_size: int,
            epsilon: float = 1e-3,
        ):
        super().__init__()
        self.init_dict = {
            "number_of_functions": number_of_functions,
            "response_size": response_size,
            "covariate_size": covariate_size,
            "epsilon": epsilon
        }
        self.number_of_functions = number_of_functions
        self.response_size = response_size
        self.covariate_size = covariate_size
        self.epsilon = epsilon

        self.A_xx_raw = nn.Parameter(
            torch.randn(self.covariate_size, self.covariate_size, self.number_of_functions)
        )
        self.A_xu = nn.Parameter(
            torch.randn(self.covariate_size, self.response_size, self.number_of_functions)
        )
        self.L_uu_raw = nn.Parameter(
            torch.randn(self.response_size, self.response_size, self.number_of_functions)
        )

        self.b_full = nn.Parameter(
            torch.randn(self.covariate_size + self.response_size, self.number_of_functions)
        )
        self.c_full = nn.Parameter(torch.randn(self.number_of_functions))

    def get_full_A(self):
        S_xx = 0.5 * (self.A_xx_raw + self.A_xx_raw.transpose(0, 1))

        tril_mask = torch.tril(torch.ones_like(self.L_uu_raw))
        L = self.L_uu_raw * tril_mask
        A_uu = torch.einsum('ipb,jpb->ijb', L, L)
        I = torch.eye(self.response_size, device=A_uu.device, dtype=A_uu.dtype).unsqueeze(-1)
        A_uu = A_uu + self.epsilon * I

        top = torch.cat([S_xx, self.A_xu], dim=1)
        bottom = torch.cat([self.A_xu.transpose(0, 1), A_uu], dim=1)
        A_full = torch.cat([top, bottom], dim=0)
        return A_full

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        x: (..., dx)
        u: (..., du)
        returns: (..., 1) strictly convex in u
        """
        z = torch.cat([x, u], dim=-1)
        A = self.get_full_A()

        quadratic_term = torch.einsum('...m,mnb,...n->...b', z, A, z)
        linear_term = torch.einsum('...m,mb->...b', z, self.b_full) 
        constant_term = self.c_full.view(*([1] * (linear_term.ndim - 1)), self.number_of_functions)

        quadratic_forms = F.softplus(quadratic_term + linear_term + constant_term)
        out = torch.sum(quadratic_forms, dim=-1, keepdim=True)

        return out
    
    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "init_dict": self.init_dict}, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.__init__(**data["init_dict"])
        self.load_state_dict(data["state_dict"])
        return self
    
    @classmethod
    def load(cls, path: str, map_location: torch.device = torch.device('cpu')) -> "ConvexQuadraticPotential":
        data = torch.load(path, map_location=map_location)
        quadratic_potential = cls(**data["init_dict"])
        quadratic_potential.load_state_dict(data["state_dict"])
        return quadratic_potential