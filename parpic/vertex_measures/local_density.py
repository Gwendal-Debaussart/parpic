import numpy as np


def graph_density_local(A, k=5):
    """Compute the local density measure for each vertex in a directed graph.

    The local density of a vertex is defined as the density of the subgraph induced by its k-hop neighbors.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the directed graph.
    k : int, default=5
        Number of hops to consider for the neighborhood.

    Returns
    -------
    np.ndarray
        Local density measure for each vertex, clipped to the range [0.01, 1.0].
    """
    n = A.shape[0]
    local_density = np.zeros(n)

    for i in range(n):
        A_k = np.linalg.matrix_power(A + np.eye(n), k)
        neighbors = np.where(A_k[i] > 0)[0]
        if len(neighbors) < 2:
            local_density[i] = 1.0
            continue

        subgraph = A[np.ix_(neighbors, neighbors)]
        edges = np.sum(subgraph) / 2
        max_edges = len(neighbors) * (len(neighbors) - 1) / 2
        local_density[i] = edges / max_edges if max_edges > 0 else 0.0

    local_density = np.clip(local_density, 0.01, 1.0)
    return local_density
