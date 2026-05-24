import numpy as np


def sum_deg(A, gamma: float = 0.5, degree_normalized: bool = True) -> np.ndarray:
    """
    Compute the sum of the in-degree and out-degree for each node in a directed graph.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the directed graph.
    gamma : float, default=0.5
        Weighting parameter between in-degree and out-degree.
    degree_normalized : bool, default=True
        Whether to normalize the degree measures to sum to 1.

    Returns
    -------
    np.ndarray
        Sum of in-degree and out-degree for each node, optionally normalized.
    """
    Din = A.sum(axis=0)
    Dout = A.sum(axis=1)
    if degree_normalized:
        Din = Din / (Din.sum() + 1e-10)
        Dout = Dout / (Dout.sum() + 1e-10)
    nu = gamma * Din + (1 - gamma) * Dout
    return nu / np.linalg.norm(nu, 1)
