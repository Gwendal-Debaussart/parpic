from torch_geometric.datasets import PolBlogs
import numpy as np
import os

def polblogs():
    """
    PolBlogs network dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features (None).
    """
    path = os.path.abspath(os.path.dirname(__file__))
    dataset = PolBlogs(root = os.path.join(path, "source/", "polblogs"))
    data = dataset[0]
    A = np.zeros((data.num_nodes, data.num_nodes))
    A[data.edge_index[0, :], data.edge_index[1, :]] = 1
    y = data.y.numpy()
    return A, y, None