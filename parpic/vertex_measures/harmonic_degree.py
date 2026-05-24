
import numpy as np


def harmonic_degree_measure(A, gamma=0.5, degree_normalized=True):
    """
    Harmonic mean degree measure for directed graphs.

    Formula:
        nu_i = 2 / (gamma / D_in(i) + (1-gamma) / D_out(i))

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
        Harmonic mean degree measure for each vertex.
    """
    Din = A.sum(axis=0) + 1e-10
    Dout = A.sum(axis=1) + 1e-10

    if degree_normalized:
        Din = Din / (Din.sum() + 1e-10)
        Dout = Dout / (Dout.sum() + 1e-10)

    harmonic = 2 / (gamma / (Din + 1e-10) + (1 - gamma) / (Dout + 1e-10))
    return harmonic / np.linalg.norm(harmonic, 1)
