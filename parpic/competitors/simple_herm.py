import numpy as np

def simple_herm(A: np.ndarray, d: int) -> np.ndarray:
    """
    Compute the simple Hermitian matrix of A.

    Parameters
    ----------
    A : np.ndarray
        Input adjacency matrix.
    d : int
        Number used to determine the complex phase.

    Returns
    -------
    np.ndarray
        Simple Hermitian matrix of A, defined as ``I - D^{-1/2} (A * omega + A^H * omega^*) D^{-1/2}``, where ``omega = exp(2j * pi / d)`` and ``D`` is the diagonal degree matrix of the symmetrized adjacency matrix.
    """
    omega = np.exp(2j * np.pi / d)
    M = A * omega
    M = M + M.conj().T

    out_deg = np.sum(A, axis=1)
    in_deg  = np.sum(A, axis=0)
    degrees = out_deg + in_deg
    degrees[degrees == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    return  (np.eye(A.shape[0]) - D_inv_sqrt @ M @ D_inv_sqrt)