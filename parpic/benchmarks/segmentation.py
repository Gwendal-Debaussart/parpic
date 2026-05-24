from .utilities import features_to_knn, load_dataset_from_local

def segmentation():
    """
    Load the Segmentation dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features.
    """
    features, labels = load_dataset_from_local("segmentation")
    adjacency = features_to_knn(features)
    return adjacency, labels, features