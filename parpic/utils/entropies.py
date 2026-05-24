import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def compute_entropy(
    operator: np.ndarray,
    t_max: int,
    entropy_type: str = "random-walk",
    n_probes: int = 0,
) -> np.ndarray:
    """Compute entropy values from a graph operator.

    Parameters
    ----------
    operator : np.ndarray
        Square operator matrix used to compute entropy trajectories.
    t_max : int
        Maximum time step (inclusive).
    entropy_type : str, default="random-walk"
        Entropy variant to compute. Supported values are "random-walk"
        and "eigen".
    n_probes : int, default=0
        Number of probe nodes for stochastic estimation in the
        "random-walk" case. If 0, the exact computation is used.

    Returns
    -------
    np.ndarray
        Entropy value at each time step from 1 to ``t_max``.
    """
    if entropy_type == "random-walk":
        entrop = diffusion_entropy(operator, max_t=t_max, n_probes=n_probes)
        return entrop
    elif entropy_type == "eigen":
        entrop = spectral_entropy(operator, max_t=t_max)
        return entrop
    else:
        raise ValueError(f"Unknown entropy type: {entropy_type}")

def diffusion_entropy(P: np.ndarray, max_t: int = 50, n_probes: int = 0, n_jobs: int = None) -> np.ndarray:
    """Compute diffusion-based entropy trajectory from a transition matrix, parallelized and using sparse matrices."""
    n = P.shape[0]
    if not sp.issparse(P):
        P = sp.csr_matrix(P)
    entropies = []
    if n_probes > 0:
        source_indices = np.random.choice(n, size=min(n_probes, n), replace=False)
        distributions = np.zeros((len(source_indices), n))
        for idx, i in enumerate(source_indices):
            distributions[idx, i] = 1.0
        def step_entropy(distributions):
            distributions_clipped = np.clip(distributions, 1e-15, 1.0)
            H_estimates = - np.sum(distributions_clipped * np.log(distributions_clipped), axis=1)
            H_total = np.sum(H_estimates) * (n / n_probes)
            return H_total
        n_jobs = n_jobs or min(multiprocessing.cpu_count(), len(source_indices))
        for _ in range(max_t):
            chunks = np.array_split(distributions, n_jobs)
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(step_entropy, chunks))
            H_total = np.sum(results)
            entropies.append(H_total)
            distributions = distributions @ P.T
    elif n_probes == 0:
        Pt = sp.eye(n, format='csr')
        def entropy_sparse(Pt):
            Pt_clipped = np.clip(Pt.data, 1e-15, 1.0)
            return -np.sum(Pt.data * np.log(Pt_clipped))
        for _ in range(1, max_t + 1):
            Pt = Pt @ P
            entropies.append(entropy_sparse(Pt))
    return np.array(entropies)


def spectral_entropy(operator: np.ndarray, max_t: int = 50) -> np.ndarray:
    """Compute spectral entropy trajectory from operator eigenvalues.

    Parameters
    ----------
    operator : np.ndarray
        Square operator matrix.
    max_t : int, default=50
        Maximum time step (inclusive).

    Returns
    -------
    np.ndarray
        Array of Shannon entropies over normalized ``|lambda|^t`` weights,
        where ``lambda`` are eigenvalues of ``operator``.
    """
    eigenvalues, _ = np.linalg.eig(operator)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    abs_eigenvalues = np.abs(eigenvalues)

    entropies = []
    for t in range(1, max_t + 1):
        lambda_t = abs_eigenvalues**t
        lambda_t_sum = np.sum(lambda_t)
        probs = lambda_t / lambda_t_sum
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)
    return np.array(entropies)
