import numpy as np
from scipy.optimize import linprog
from protocols.pushforward_operator import PushForwardOperator, TrainParams
import torch

class LinearVectorQuantileRegression(PushForwardOperator):
    def __init__(self, num_latent_points_to_generate: int = 500):
        self.u = None
        self.b_u = None
        self.num_latent_points_to_generate = num_latent_points_to_generate

    def fit(self, dataloader: torch.utils.data.DataLoader, train_params: TrainParams = TrainParams(verbose=False), *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_params (TrainParams): Training parameters.
        """
        X_Y_tuple = [(X_batch, Y_batch) for X_batch, Y_batch in dataloader]
        X_tensor = torch.cat([X_batch for X_batch, _ in X_Y_tuple], dim=0)
        Y_tensor = torch.cat([Y_batch for _, Y_batch in X_Y_tuple], dim=0)
        self.fit_tensor(X_tensor, Y_tensor, train_params["verbose"])
        return self

    def fit_tensor(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor, verbose: bool = False):
        """Fits the pushforward operator to the data.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
            verbose (bool): Whether to print verbose output.
        """
        n, d = Y_tensor.shape
        U = np.random.normal(size=(self.num_latent_points_to_generate, d))
        m = U.shape[0]
        X = X_tensor.cpu().numpy()
        Y = Y_tensor.cpu().numpy()

        nu = np.ones((n, 1)) / n
        mu = np.ones((m, 1)) / m

        result = self.solve_linear_vector_quantile_regression_primal(X, Y, U, nu, mu, verbose)
        print(result['eqlin']['marginals'].shape)
        self.b_u = result['eqlin']['marginals'][n:].reshape((U.shape[0], X.shape[1]), order='F')
        self.u = U
        return self

    def solve_linear_vector_quantile_regression_primal(self, X: np.ndarray, Y: np.ndarray, U: np.ndarray, nu: np.ndarray, mu: np.ndarray, verbose: bool = False):
        """
        Solves the primal form of the Linear Vector Quantile Regression (VQR) linear program.

        This function implements the discrete formulation of the VQR problem as
        described in equation (4.1) of "VECTOR QUANTILE REGRESSION" by
        G. Carlier, V. Chernozhukov, and A. Galichon.

        Args:
            X (np.ndarray): The n x p matrix of regressors.
            Y (np.ndarray): The n x q matrix of response variables.
            U (np.ndarray): The m x q matrix of points for the reference distribution.
            nu (np.ndarray): The n x 1 vector of probability weights for observations (Y, X).
            mu (np.ndarray): The m x 1 vector of probability weights for the reference points U.
            verbose (bool): Whether to print verbose output.

        Returns:
            scipy.optimize.OptimizeResult: The result object from scipy.optimize.linprog.
            The optimal `pi` matrix can be recovered by `res.x.reshape((n, m)).T`.
        """
        n, _ = Y.shape
        m, _ = U.shape

        UY_transpose = U @ Y.T
        c = -UY_transpose.flatten('F')

        A_eq_a = np.kron(np.eye(n), np.ones((1, m)))

        E_X = nu.T @ X
        mu_EX = mu @ E_X

        A_eq_b = np.kron(X.T, np.eye(m))
        b_eq_b = mu_EX.flatten('F')

        A_eq = np.vstack([A_eq_a, A_eq_b])
        b_eq = np.concatenate([nu.flatten(), b_eq_b])

        bounds = (0, None)

        if verbose:
            print("Solving the linear program...")

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        return result

    def push_forward_u_given_x(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        X_numpy = X.cpu().numpy()
        U_numpy = U.cpu().numpy()
        phi_u = -self.b_u @ X_numpy.T
        pushforward_of_u = self.estimate_gradients_for_points_knn(self.u, phi_u.T, points_of_interest=U_numpy, k=5)
        return pushforward_of_u

    def estimate_gradient_knn(self, u_samples: np.ndarray, f_samples: np.ndarray, point: np.ndarray, k: int) -> np.ndarray:
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
        point = np.asarray(point)

        if point.shape != (d,):
            raise ValueError(f"The 'point' must be a d-dimensional vector, but got shape {point.shape} for d={d}.")

        if k < d + 1:
            raise ValueError(f"k must be at least d+1 (which is {d+1}) to fit a hyperplane in R^{d}.")
        if k > N:
            raise ValueError(f"k ({k}) cannot be larger than the number of samples ({N}).")

        distances_sq = np.sum((u_samples - point)**2, axis=1)
        knn_indices = np.argsort(distances_sq)[:k]

        local_u_samples = u_samples[knn_indices]
        local_f_samples = f_samples[knn_indices]

        A_design = np.hstack([local_u_samples, np.ones((k, 1))])

        try:
            coeffs = np.linalg.pinv(A_design) @ local_f_samples
        except np.linalg.LinAlgError:
            return np.full(d, np.nan)

        gradient = coeffs[:d]
        return gradient


    def estimate_gradients_for_points_knn(self, u_samples: np.ndarray, f_samples: np.ndarray, points_of_interest: np.ndarray, k: int) -> np.ndarray:
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

        all_gradients = np.array([
            self.estimate_gradient_knn(u_samples, f_samples[i], point, k)
            for i, point in enumerate(points_of_interest)
        ])

        return all_gradients

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        np.save(path, np.concatenate([self.u, self.b_u], axis=0), allow_pickle=True)


    def load(self, path: str):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = np.load(path, allow_pickle=True)
        self.num_latent_points_to_generate = data.shape[0] // 2
        self.u = data[:self.num_latent_points_to_generate]
        self.b_u = data[self.num_latent_points_to_generate:]
        return self