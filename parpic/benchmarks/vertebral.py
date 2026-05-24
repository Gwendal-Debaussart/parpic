from .utilities import features_to_knn, load_dataset_from_local

def vertebral():
    """
    Load the Vertebral dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features.
    """
    features, labels = load_dataset_from_local("vertebral")
    adjacency = features_to_knn(features)
    return adjacency, labels, features