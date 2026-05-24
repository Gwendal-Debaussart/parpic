"""
Metrics for evaluating clustering performance, both supervised and unsupervised.
A metric is expected to be a dict containing:
  - name
    o A string representing the name of the metric, used for display and saving results.
  - function:
    o A callable that computes the metric given the true labels, predicted labels, and optionally features.
  - type:
    o A string indicating whether the metric is "supervised" (requires true labels) or "unsupervised" (does not require true labels).
"""

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
)


def metric_list():
    return [
        {
            "name": "Adjusted Rand Index",
            "function": adjusted_rand_score,
            "type": "supervised",
        },
        {
            "name": "Adjusted Mutual Information",
            "function": adjusted_mutual_info_score,
            "type": "supervised",
        },
        {
            "name": "Calinski-Harabasz Index",
            "function": calinski_harabasz_score,
            "type": "unsupervised",
        },
        {
            "name": "Silhouette Score",
            "function": silhouette_score,
            "type": "unsupervised",
        },
    ]
