import numpy as np
from scipy.optimize import linprog

def solve_vqr_primal(Y, X, U, nu, mu):
    """
    Solves the primal form of the Vector Quantile Regression (VQR) linear program.

    This function implements the discrete formulation of the VQR problem as
    described in equation (4.1) of "VECTOR QUANTILE REGRESSION" by
    G. Carlier, V. Chernozhukov, and A. Galichon.

    Args:
        Y (np.ndarray): The n x q matrix of response variables.
        X (np.ndarray): The n x p matrix of regressors.
        U (np.ndarray): The m x q matrix of points for the reference distribution.
        nu (np.ndarray): The n x 1 vector of probability weights for observations (Y, X).
        mu (np.ndarray): The m x 1 vector of probability weights for the reference points U.

    Returns:
        scipy.optimize.OptimizeResult: The result object from scipy.optimize.linprog.
        The optimal `pi` matrix can be recovered by `res.x.reshape((n, m)).T`.
    """
    n, _ = Y.shape
    m, _ = U.shape

    UY_transpose = U @ Y.T
    c = -UY_transpose.flatten('F')

    # Constraint (a): Marginal distribution of (X, Y) is nu.
    A_eq_a = np.kron(np.eye(n), np.ones((1, m)))

    E_X = nu.T @ X
    mu_EX = mu @ E_X

    A_eq_b = np.kron(X.T, np.eye(m))
    b_eq_b = mu_EX.flatten('F')

    A_eq = np.vstack([A_eq_a, A_eq_b])
    b_eq = np.concatenate([nu.flatten(), b_eq_b])

    bounds = (0, None)

    print("Solving the linear program...")
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    return res