from sklearn.datasets import load_wine
from .utilities import features_to_knn
from scipy import stats

def wine():
    """
    Load Wine dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features.

    Notes
    -----
    The features are standardized using z-score normalization before constructing the adjacency matrix.
    """
    data = load_wine(as_frame=True)
    features = data.data
    labels = data.target
    features = stats.zscore(features, axis=0)
    adjacency_matrix = features_to_knn(features)
    return adjacency_matrix, labels, features