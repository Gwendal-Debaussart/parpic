import numpy as np
from sklearn.neighbors import kneighbors_graph
import os
from datasets import load_from_disk


def features_to_knn(features: np.ndarray) -> np.ndarray:
    """
    Convert features to directed unweighted kNN graph adjacency matrix.

    For each node i, creates outgoing edges to its k nearest neighbors j.

    Parameters
    ----------
    features : np.ndarray
        Node feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Adjacency matrix of the kNN graph, shape (n_samples, n_samples).
    """

    k = int(np.ceil(np.log(features.shape[0])))
    knn_graph = kneighbors_graph(
        features, n_neighbors=k, mode="connectivity", include_self=False
    )
    return knn_graph.toarray()

def load_dataset_from_local(name: str):
    """
    Load a dataset from the local disk.

    Parameters
    ----------
    name : str
        Name of the dataset to load (e.g., "vertebral", "wdbc", "yeast").
    Returns
    -------
    tuple
        A tuple containing the features and labels of the dataset.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    data = load_from_disk(os.path.join(path, "source/", name))
    df_new = data.with_format("numpy")
    features = df_new["train"]["data"]
    labels = df_new["train"]["labels"]
    return features, labels
