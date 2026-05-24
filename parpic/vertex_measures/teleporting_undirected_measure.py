import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import matrix_power

def teleporting_undirected_measure(adjacency_matrix, alpha, t, epsilon=1e-8):
    """Compute the teleporting random walk vertex measure for an undirected graph.

    Parameters
    ----------
    adjacency_matrix : np.ndarray or sp.spmatrix
        Adjacency matrix of the undirected graph.
    alpha : float
        Exponent parameter for the measure.
    t : int
        Number of steps for the random walk.
    epsilon : float, default=1e-8
        Small constant to avoid division by zero and ensure positivity.

    Returns
    -------
    np.ndarray
        Teleporting random walk vertex measure for each vertex, normalized to sum to 1.

    """
    is_sparse = sp.issparse(adjacency_matrix)
    N = adjacency_matrix.shape[0]
    diags = adjacency_matrix.sum(axis=1).A1 if is_sparse else np.array(adjacency_matrix.sum(axis=1)).flatten()

    if np.allclose(diags, 1.0):
        P = adjacency_matrix
    else:
        diags[diags == 0] = 1e-10
        D_inv = sp.diags(1 / diags) if is_sparse else np.diag(1 / diags)
        P = D_inv @ adjacency_matrix

    P_t = matrix_power(P.copy(), t)

    nu = (np.array(((1/N) * np.ones((1, N)) @ P_t)).T.flatten())**alpha
    nu[nu <= 0] = epsilon
    nu_normalized = nu / np.sum(nu)

    return nu_normalized