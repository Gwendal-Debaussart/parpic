from sklearn.datasets import fetch_openml
from .utilities import features_to_knn
from sklearn.decomposition import PCA


def mnist():
    """
    Load the MNIST dataset.

    Returns:
    tuple
        A tuple containing the adjacency matrix, labels, and features.
    """

    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.to_numpy()
    features = PCA(n_components=100).fit_transform(X)
    true_labels = mnist.target.to_numpy().astype(int)
    adjacency = features_to_knn(features)
    return adjacency, true_labels, features
