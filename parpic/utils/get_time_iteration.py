import numpy as np
import numpy as np
from kneed import KneeLocator
from .entropies import diffusion_entropy


def get_time_iteration(
    operator: np.ndarray,
    t_max: int = 250,
    n_probes: int = None,
) -> int:
    """
    Get optimal time iteration using entropy-based knee detection.

    Parameters
    ----------
    operator : np.ndarray
        Square operator matrix used to compute entropy trajectories.
    t_max : int, default=250
        Maximum time step (inclusive) for entropy trajectory computation.
    n_probes : int, default=None
        Number of probe nodes for stochastic estimation in the diffusion entropy.
        If None, it is set to the minimum of 100 and the square root of the number of nodes.

    Returns
    -------
    int
        Optimal time iteration determined by knee detection on the entropy trajectory.

    Notes
    -----
    This function computes the diffusion-based entropy trajectory for the given operator and uses the KneeLocator to find the optimal time iteration where the entropy curve has a significant change in slope. If no knee is detected, it defaults to the logarithm of the number of nodes.
    """
    n = operator.shape[0]
    if n_probes is None:
        n_probes = max(min(n, 100), int(np.sqrt(n)))
    entropy_vals = diffusion_entropy(operator, max_t=t_max, n_probes=n_probes)
    knee_locator = KneeLocator(
        range(1, t_max + 1),
        entropy_vals,
        curve="concave",
        direction="increasing",
    )
    optimal_t = knee_locator.knee
    if optimal_t is None:
        optimal_t = int(np.log(n))
    return optimal_t


def get_time_iteration_differences(P, t_max = 250, acceleration_threshold=1e-4):
    """
    Compute diffusion time based on acceleration-based stopping criterion (PIC style).

    Monitors the rate of change (acceleration) in the power iteration process.
    Stops when the acceleration falls below threshold, indicating convergence.

    Parameters:
    -----------
    P : ndarray
        Transition matrix
    t_max : int
        Maximum iterations
    acceleration_threshold : float
        Threshold for stopping based on acceleration

    Returns:
    --------
    int : Optimal stopping time
    """
    n = P.shape[0]
    prev_prev_power = np.eye(n)
    prev_power = np.eye(n) @ P  # t=1
    current_power = prev_power @ P  # t=2

    for t in range(3, t_max + 1):
        current_power = current_power @ P
        diff_current = np.linalg.norm(current_power - prev_power, ord='fro')
        diff_prev = np.linalg.norm(prev_power - prev_prev_power, ord='fro')
        acceleration = abs(diff_current - diff_prev)
        if acceleration < acceleration_threshold:
            return t

        prev_prev_power = prev_power
        prev_power = current_power

    return t_max