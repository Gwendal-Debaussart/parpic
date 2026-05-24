import networkx as nx


def directed_sbm(block_sizes, P, seed =42):
    """Generate a directed stochastic block model dataset.

    Parameters
    ----------
    block_sizes : list[int]
        Number of nodes in each block.
    P : array-like
        Inter/intra-block edge probability matrix.
    seed : int, default=42
        Random seed passed to NetworkX SBM generation.

    Returns
    -------
    tuple
        ``(A, y, None)`` where ``A`` is the adjacency matrix,
        ``y`` are block labels, and the third value is ``None``
        for feature compatibility with other loaders.
    """
    G = nx.stochastic_block_model(
        sizes=block_sizes, p=P, directed=True, selfloops=False, seed=seed
    )
    y = []
    for i, size in enumerate(block_sizes):
        y.extend([i] * size)
    return nx.adjacency_matrix(G).todense(), y, None
