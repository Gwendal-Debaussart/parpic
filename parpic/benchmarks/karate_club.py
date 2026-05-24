from networkx.generators import karate_club_graph
import networkx as nx


def karate_club():
    """
    Generate the karate club benchmark dataset.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix, labels, and features (None).
    """
    adjacency_matrix = nx.adjacency_matrix(karate_club_graph()).todense()
    y = [data["club"] for _, data in karate_club_graph().nodes(data=True)]
    return adjacency_matrix, y, None
