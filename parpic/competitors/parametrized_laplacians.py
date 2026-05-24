import numpy as np


def parametrized_laplacian(
    A: np.ndarray,
    nu: np.ndarray,
    normalized: bool = True,
    dangling_policy: str = "uniform",
) -> np.ndarray:
    """Compute the parametrized Laplacian of a graph.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the graph.
    nu : np.ndarray
        Vertex measure (positive values).
    normalized : bool, default=True
        Whether to return the normalized Laplacian.
    dangling_policy : str, default="uniform"
        Policy for handling dangling nodes. Supported values are "uniform" and "self".

    Returns
    -------
    np.ndarray
        Parametrized Laplacian matrix.
    """

    D_nu = np.diag(nu)
    L_nu = D_nu @ A + A.T @ D_nu
    row_sums = L_nu.sum(axis=1)
    dangling_idx = np.where(row_sums == 0)[0]

    if dangling_idx.size > 0:
        if dangling_policy == "uniform":
            L_nu[dangling_idx, :] = 1.0 / A.shape[0]
            row_sums = L_nu.sum(axis=1)
        elif dangling_policy == "self":
            for i in dangling_idx:
                L_nu[i, i] = 1.0
            row_sums = L_nu.sum(axis=1)
        else:
            row_sums[row_sums == 0] = 1e-12

    if normalized:
        D_inv = np.diag(1 / row_sums)
        L_nu = D_inv @ L_nu
    return L_nu
