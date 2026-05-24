from sklearn.cluster import KMeans
import numpy as np

def get_clustering(embedding, num_clusters):
  """Cluster embedding vectors with KMeans.

  Parameters
  ----------
  embedding : np.ndarray
      Node embedding matrix of shape ``(n_samples, n_features)``.
  num_clusters : int
      Number of clusters to fit.

  Returns
  -------
  np.ndarray
      Predicted cluster labels of length ``n_samples``.
  """
  k_means = KMeans(n_clusters = num_clusters).fit(embedding)
  predicted_labels = k_means.labels_
  return predicted_labels