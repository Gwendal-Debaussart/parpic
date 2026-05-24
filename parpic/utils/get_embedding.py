import numpy as np


def get_embedding(X, dim: int, method="eigen"):
    """Compute an embedding from an operator matrix.

    Parameters
    ----------
    X : np.ndarray
        Input matrix to embed.
    dim : int
        Target embedding dimension.
    method : str, default="eigen"
        Embedding method. Supported values are "eigen", "r-svd",
        "l-svd", and "natural".

    Returns
    -------
    np.ndarray
        Embedded representation with ``dim`` columns (or fewer if limited
        by matrix shape).
    """
    if method.lower() == "eigen":
        _, eigenvecs = np.linalg.eigh(X)
        norms = np.linalg.norm(eigenvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        eigenvecs = eigenvecs[:, ::-1]
        eigenvecs = eigenvecs / norms
        return eigenvecs[:, :dim].real
    elif method.lower() == "r-svd":
        _, _, Vt = np.linalg.svd(X, full_matrices=True)
        return Vt[:, :dim]
    elif method.lower() == "l-svd":
        U, _, _ = np.linalg.svd(X, full_matrices=True)
        return U[:, :dim]
    elif method.lower() == "natural":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return (X / norms)[:, :dim].real
    else:
        raise ValueError(f"Unknown method: {method}")
