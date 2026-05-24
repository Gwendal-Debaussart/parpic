import numpy as np


def dd_sym(A):
    """
    Docstring for dd_sym

    Parameters
    ----------
    A : np.ndarray
        Input adjacency matrix.

    Returns
    -------
    np.ndarray
        Symmetrized Bibliographic operator ``A.T @ A + A @ A.T``.
    """
    M = A.T @ A
    return M + M.T
