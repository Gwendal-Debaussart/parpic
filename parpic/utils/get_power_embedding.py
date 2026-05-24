import numpy as np


def get_power_embedding(
    operator: np.ndarray,
    t: int,
    projection_type: str = "full",
    n_cols: int = None,
) -> np.ndarray:
    """
    Compute embedding via power iteration.

    Parameters
    ----------
    operator : np.ndarray
        Square operator matrix used to compute the embedding.
    t : int
        Number of power iterations to perform.
    projection_type : str, default="full"
        Type of projection to apply during power iteration. Supported values are:
        - "full": No projection, compute the full power of the operator.
        - "random": Random projection to reduce dimensionality during power iteration.
    n_cols : int, default=None
        Number of columns for random projection if ``projection_type`` is "random". If None, it is set to the max of 100 and the square root of the number of nodes.

    Returns
    -------
    np.ndarray
        Node embedding matrix of shape ``(n_samples, n_cols)``.
    """
    n = operator.shape[0]
    if projection_type == "full":
        return np.linalg.matrix_power(operator, t)
    elif projection_type == "random":
        if n_cols is None:
            n_cols = max(min(n, 100), int(np.sqrt(n)))
        Z = np.random.rand(n, n_cols)
        Z = Z / np.linalg.norm(Z, axis=0, keepdims=True)
        for _ in range(t):
            Z = operator @ Z
        return Z
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")
