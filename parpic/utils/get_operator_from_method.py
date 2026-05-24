import numpy as np


def get_operator_from_method(method, A):
    """
    Given a method dictionary, return the corresponding operator function.

    Parameters
    ----------
    method : dict
        Method configuration dictionary. Expected keys include:
        - "function": The operator function to apply.
        - "params": (Optional) Parameters to pass to the operator function.
        - "vertex_measure": (Optional) A dictionary with keys "function" and "params" for vertex measure computation.
        - "input_type": (Optional) Type of input expected by the operator function (e.g., "adjacency", "transition"). Default is "adjacency".
    A : np.ndarray
        Adjacency matrix of the graph.

    Returns
    -------
    np.ndarray
        The operator matrix computed from the input adjacency matrix and method configuration.
    """
    parameters = method.get("params", {})
    vertex_measure = method.get("vertex_measure", None)
    if method.get("input_type", "adjacency").lower() == "transition":
        diags = A.sum(axis=1)
        diags[diags == 0] = 1e-12
        D_inv = np.diag(1 / diags)
        X = D_inv @ A
    else:
        X = A
    if vertex_measure is not None:
        nu = vertex_measure["function"](A, **vertex_measure.get("params", {}))
        return method["function"](X, nu, **parameters)
    else:
        return method["function"](X, **parameters)
