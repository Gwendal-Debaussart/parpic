from torch_geometric.datasets import EmailEUCore
import numpy as np


def email_eu():
    """
    Email-Eu-Core dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features (None).
    """
    mails = EmailEUCore(root="data/EmailEUCore")
    data = mails[0]
    num_nodes = data.num_nodes
    edge_index = data.edge_index.cpu().numpy()

    A = np.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1
    y = data.y.numpy()
    return A, y, None
