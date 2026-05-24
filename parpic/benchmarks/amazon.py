from torch_geometric.datasets import Amazon
import numpy as np
import os

def amazon() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Amazon Computers dataset from PyTorch Geometric.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features.
    """
    path = os.path.abspath(os.path.dirname(__file__))
    dataset = Amazon(root=os.path.join(path, "source/", "amazon"), name="Computers")
    data = dataset[0]

    num_nodes = data.num_nodes
    edge_index = data.edge_index.numpy()
    A = np.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1
    y = data.y.numpy()
    x = data.x.numpy()
    return A, y, x