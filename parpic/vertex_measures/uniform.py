import numpy as np

def uniform_measure(A):
    """Uniform vertex measure for directed graphs.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the directed graph.

    Returns
    -------
    np.ndarray
        Uniform vertex measure, where each vertex has the same value.
    """
    return np.ones(A.shape[0])