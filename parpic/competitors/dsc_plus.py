import numpy as np
from .utils import teleporting_rw, stationary_distribution


def dsc_plus(A, gamma: float = 0.9):
    """Compute the DSC+ symmetrized operator.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    gamma : float, default=0.9
        Teleportation parameter used to build the teleporting random walk.

    Returns
    -------
    np.ndarray
        Symmetrized DSC+ operator

    Notes
    -----
    This implementation estimates the stationary distribution with
    power iteration in ``stationary_distribution``.
    """
    A = teleporting_rw(A, gamma=gamma)
    pi = stationary_distribution(A)
    Pi_sqrt = np.diag(np.sqrt(pi))
    Pi_inv_sqrt = np.diag(1 / np.sqrt(pi))
    L = Pi_sqrt @ A @ Pi_inv_sqrt
    return -(np.eye(L.shape[0]) - (L + L.T) / 2)
