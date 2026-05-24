from .utilities import load_dataset_from_local, features_to_knn


def control_chart():
    """
    Load the Control Chart dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features.
    """
    features, labels = load_dataset_from_local("control_chart")
    adjacency = features_to_knn(features)
    return adjacency, labels, features
