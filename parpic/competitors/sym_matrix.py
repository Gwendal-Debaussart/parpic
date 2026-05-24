import numpy as np

def sym_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric normalized stochastic complement of W.

    Parameters
    ----------
    A : np.ndarray
        Input adjacency matrix.

    Returns
    -------
    np.ndarray
        Symmetric normalized stochastic complement of W.
    """
    A_sym = (A + A.T) / 2.0
    d_sym = np.asarray(A_sym.sum(axis=1)).flatten()
    d_sym[d_sym == 0] = 1e-10
    D_inv_sym = np.diag(1 / d_sym)
    P_sym = D_inv_sym @ A_sym
    return P_sym