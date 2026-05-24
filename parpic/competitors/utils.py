import numpy as np


def teleporting_rw(P, gamma: float = 0.85):
    """Compute the teleporting random walk operator.

    Parameters
    ----------
    P : np.ndarray
        Input transition matrix (row-stochastic).
    gamma : float, default=0.85
        Teleportation parameter.

    Returns
    -------
    np.ndarray
        Teleporting random walk operator ``gamma * P + (1 - gamma) * 1/n``.
    """
    if not np.allclose(P.sum(axis=1), 1):
        P_norm = P.sum(axis=1, keepdims=True)
        P_norm[P_norm == 0] = 1
        P = P / P_norm
    n = P.shape[0]
    return gamma * P + (1 - gamma) * np.ones((n, n)) / n


def stationary_distribution(P, tol=1e-12, max_iter=10000):
    """Compute stationary distribution of a row-stochastic matrix P.

    Parameters
    ----------
    P : np.ndarray
        Row-stochastic transition matrix.
    tol : float, default=1e-12
        Convergence tolerance for power iteration.
    max_iter : int, default=10000
        Maximum number of iterations for power iteration.

    Returns
    -------
    np.ndarray
        Stationary distribution vector of length n.
    """
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, 1) < tol:
            break
        pi = pi_next
    return pi