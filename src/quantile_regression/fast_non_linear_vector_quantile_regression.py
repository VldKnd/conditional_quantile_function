from protocols.pushforward_operator import PushForwardOperator, TrainParams
from tqdm import tqdm
import torch
import torch.nn as nn

class FastNonLinearVectorQuantileRegression(PushForwardOperator):
    def __init__(self, input_dimension:int, embedding_dimension: int = 5, hidden_dimension: int = 100, number_of_hidden_layers: int = 1):
        self.feature_network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.Softplus(),
            *[nn.Linear(hidden_dimension, hidden_dimension), nn.Softplus()] * number_of_hidden_layers,
            nn.Linear(hidden_dimension, embedding_dimension),
            torch.nn.BatchNorm1d(num_features=embedding_dimension, affine=False, track_running_stats=True)
        )
        self.fitted = False
        self.embedding_dimension = embedding_dimension
        self.b_u = None
        self.phi = None

    def fit(self, dataloader: torch.utils.data.DataLoader, train_params: TrainParams = TrainParams(verbose=False), *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_params (TrainParams): Training parameters.
        """

        X_Y_tuple = [(X_batch, Y_batch) for X_batch, Y_batch in dataloader]
        X_tensor = torch.cat([X_batch for X_batch, _ in X_Y_tuple], dim=0)
        Y_tensor = torch.cat([Y_batch for _, Y_batch in X_Y_tuple], dim=0)
        self.fit_tensor(X_tensor=X_tensor, Y_tensor=Y_tensor, verbose=train_params["verbose"], train_params=train_params, *args, **kwargs)
        return self

    def fit_tensor(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor, verbose: bool = False, train_params: TrainParams = TrainParams(verbose=False), *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
            verbose (bool): Whether to print verbose output.
        """
        epsilon = 0.001
        num_epochs = train_params["num_epochs"]
        device_type_and_specification = {
            "device": Y_tensor.device,
            "dtype": Y_tensor.dtype
        }
        m, _ = Y_tensor.shape
        U_tensor = torch.randn_like(Y_tensor, **device_type_and_specification)

        psi_tensor = torch.full(size=(Y_tensor.shape[0], 1), fill_value=0.1, **device_type_and_specification)
        psi_tensor.requires_grad = True

        b_tensor = torch.zeros(*(m, self.embedding_dimension), **device_type_and_specification)
        b_tensor.requires_grad = True

        self.feature_network.to(**device_type_and_specification)
        self.feature_network.train()

        network_optimizer = torch.optim.Adam([dict(params=self.feature_network.parameters())], lr=train_params["learning_rate"])
        b_psi_optimizer = torch.optim.Adam([dict(params=[b_tensor, psi_tensor])], lr=train_params["learning_rate"])

        for _ in tqdm(range(num_epochs), disable=not train_params["verbose"]):
                b_psi_optimizer.zero_grad()
                network_optimizer.zero_grad()

                phi_tensor = epsilon * torch.logsumexp(
                        (
                                U_tensor @ Y_tensor.T  - # (m, q) @ (q, N)
                                b_tensor @ self.feature_network(X_tensor).T - # (m, 1) @ (1, N)
                                psi_tensor.reshape(1, -1) # (N, 1).reshape(1, N)
                        ) / epsilon
                        , dim = 1
                ) # (m, 1)

                objective = torch.mean(psi_tensor) + torch.mean(phi_tensor)

                objective.backward()
                b_psi_optimizer.step()
                network_optimizer.step()

        with torch.no_grad():
                phi_tensor = epsilon * torch.logsumexp(
                        (
                                U_tensor @ Y_tensor.T  - # (m, q) @ (q, N)
                                b_tensor @ self.feature_network(X_tensor).T - # (m, 1) @ (1, N)
                                psi_tensor.reshape(1, -1) # (N, 1).reshape(1, N)
                        ) / epsilon
                        , dim = 1, keepdim=True
                )

        self.phi = phi_tensor.detach()
        self.b_u = b_tensor.detach()
        self.u = U_tensor.detach()
        self.feature_network.zero_grad()
        self.feature_network.eval()
        self.fitted = True
        return self

    def push_forward_u_given_x(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        device_type_and_specification = {
            "device": X.device,
            "dtype": X.dtype
        }

        self.b_u = self.b_u.to(**device_type_and_specification)
        self.phi = self.phi.to(**device_type_and_specification)
        self.u = self.u.to(**device_type_and_specification)
        self.feature_network.to(**device_type_and_specification)

        potential = self.b_u @ self.feature_network(X).T + self.phi
        potential = potential.to(**device_type_and_specification)
        pushforward_of_u = self.estimate_gradients_for_points_knn(self.u, potential, points_of_interest=U, k=30)
        return pushforward_of_u

    def estimate_gradient_knn(self, u_samples: torch.Tensor, f_samples: torch.Tensor, point: torch.Tensor, k: int) -> torch.Tensor:
        """
        Estimates the gradient of a scalar function f: R^d -> R at a single
        given point using k-nearest neighbors and ordinary least-squares.

        This function fits a local linear model (a hyperplane) to the k-nearest
        neighbors of the target point. The gradient of that model is the
        estimated gradient.

        Args:
            u_samples (np.ndarray): An (N, d) array of N input samples in d dimensions.
            f_samples (np.ndarray): An (N,) array of N corresponding scalar output samples.
            point (tuple or np.ndarray): The d-dimensional point (u_1, ..., u_d) at
                which to estimate the gradient.
            k (int): The number of nearest neighbors to use for the estimation.

        Returns:
            np.ndarray: The estimated (d,) gradient vector.
        """
        f_samples = f_samples.flatten()

        if u_samples.shape[0] != f_samples.shape[0]:
            raise ValueError("u_samples and f_samples must have the same number of samples.")

        N, d = u_samples.shape
        point = torch.tensor(point)

        if point.shape != (d,):
            raise ValueError(f"The 'point' must be a d-dimensional vector, but got shape {point.shape} for d={d}.")

        if k < d + 1:
            raise ValueError(f"k must be at least d+1 (which is {d+1}) to fit a hyperplane in R^{d}.")
        if k > N:
            raise ValueError(f"k ({k}) cannot be larger than the number of samples ({N}).")

        distances_sq = torch.sum((u_samples - point)**2, axis=1)
        knn_indices = torch.argsort(distances_sq)[:k]

        local_u_samples = u_samples[knn_indices]
        local_f_samples = f_samples[knn_indices]

        A_design = torch.hstack([local_u_samples, torch.ones((k, 1))])

        try:
            coeffs = torch.linalg.pinv(A_design) @ local_f_samples
        except torch.linalg.LinAlgError:
            return torch.full(d, torch.nan)

        gradient = coeffs[:d]
        return gradient


    def estimate_gradients_for_points_knn(self, u_samples: torch.Tensor, f_samples: torch.Tensor, points_of_interest: torch.Tensor, k: int) -> torch.Tensor:
        """
        Estimates the gradient for multiple points of interest by calling
        estimate_gradient_knn for each point.

        Args:
            u_samples (np.ndarray): An (N, d) array of input samples.
            f_samples (np.ndarray): An (M, N) array of scalar output samples.
            points_of_interest (np.ndarray): An (M, d) array of M points where
                the gradient is to be estimated.
            k (int): The number of nearest neighbors to use.

        Returns:
            np.ndarray: An (M, d) array containing the M estimated gradients.
        """
        if points_of_interest.ndim == 1:
            points_of_interest = points_of_interest.reshape(1, -1)

        all_gradients = torch.stack([
            self.estimate_gradient_knn(u_samples, f_samples[:, i], point, k)
            for i, point in enumerate(points_of_interest)
        ])

        return all_gradients

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({"u": self.u, "b_u": self.b_u, "phi": self.phi, "feature_network.state_dict": self.feature_network.state_dict()}, path)

    def load(self, path: str):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path)
        self.u = data["u"]
        self.b_u = data["b_u"]
        self.phi = data["phi"]
        self.feature_network.load_state_dict(data["feature_network.state_dict"])
        self.feature_network.eval()
        self.fitted = True
        return self