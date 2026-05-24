from sklearn.datasets import load_iris
from .utilities import features_to_knn

def iris():
    """Load the Iris dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features.
    """
    data = load_iris()
    features = data.data
    labels = data.target
    adjacency_matrix = features_to_knn(features)
    return adjacency_matrix, labels, features