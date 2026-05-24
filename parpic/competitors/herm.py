import numpy as np


def herm(adjacency_matrix: np.ndarray, normalized: bool = False) -> np.ndarray:
    """Compute the Hermitian adjacency matrix of a directed graph.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix of the directed graph.
    normalized : bool, default=False
        Whether to return the normalized Hermitian adjacency matrix.

    Returns
    -------
    np.ndarray
        Hermitian adjacency matrix of the directed graph.

    """

    n = adjacency_matrix.shape[0]
    H = np.zeros((n, n), dtype=complex)
    both = (adjacency_matrix == 1) & (adjacency_matrix.T == 1)
    u_to_v = (adjacency_matrix == 1) & (adjacency_matrix.T == 0)
    v_to_u = (adjacency_matrix == 0) & (adjacency_matrix.T == 1)
    H[both] = 1
    H[u_to_v] = 1j
    H[v_to_u] = -1j
    if normalized:
        degree = np.abs(H).sum(axis=1)
        degree[degree == 0] = 1
        H = H @ np.diag(1.0 / np.sqrt(degree))
    return H
