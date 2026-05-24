def evaluate_labels(true_labels, features, pred_labels, metric):
    """Evaluate predicted labels with one or multiple metrics.

    Parameters
    ----------
    true_labels : np.ndarray
        Ground-truth labels.
    features : np.ndarray or None
        Feature matrix used by unsupervised metrics. Can be ``None``.
    pred_labels : np.ndarray
        Predicted cluster labels.
    metric : dict or list[dict]
        Metric configuration(s). Each dict is expected to contain at least
        ``name``, ``function``, and ``type`` where ``type`` is "supervised"
        or "unsupervised".

    Returns
    -------
    float or dict
        A single metric value if ``metric`` is a dict, or a mapping
        ``{metric_name: value}`` if ``metric`` is a list.
    """
    if type(metric) == list:
        scores = {}
        for m in metric:
            if m.get("type", "") == "supervised":
                scores[m["name"]] = m.get("function")(true_labels, pred_labels)
            elif m.get("type", "") == "unsupervised":
                if features is None:
                    scores[m["name"]] = -1.0
                else:
                    scores[m["name"]] = m.get("function")(features, pred_labels)
        return scores
    if metric.get("type", "") == "supervised":
        return metric["function"](true_labels, pred_labels)
    elif metric.get("type", "") == "unsupervised":
        return metric["function"](features, pred_labels)
