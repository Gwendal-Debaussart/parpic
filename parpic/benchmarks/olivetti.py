from sklearn.datasets import fetch_olivetti_faces
from .utilities import features_to_knn
from sklearn.decomposition import PCA


def olivetti():
    """Load the Olivetti faces dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features.
    """
    data = fetch_olivetti_faces()
    features = data.data
    true_labels = data.target
    features = PCA(n_components=150).fit_transform(features)
    adjacency = features_to_knn(features)
    return adjacency, true_labels, features
