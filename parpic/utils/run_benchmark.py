"""
Utility functions for running the benchmark, including:
- get_existing_repeats: Check how many repeats have already been run for a given dataset and method, to avoid redundant computations.
- save_raw_results: Save raw results of benchmark runs in a structured format (CSV) for later analysis.
- run_one_repeat: Run a single repeat of the benchmark for a given dataset, method, and metrics.
- run_benchmark_for_dataset: Run the benchmark for a specific dataset across multiple methods and repeats, while checking for existing results to avoid redundant runs.

The logic for running the benchmark is as follows:
1. For each dataset and method, check how many repeats have already been run using `get_existing_repeats`.
2. If more repeats are needed, run the benchmark for the required number of repeats using `run_one_repeat`, which computes the operator, embedding, clustering, and evaluates the metrics.
3. Save the raw results using `save_raw_results`, which appends to existing CSV files if they exist, or creates new ones if not.
"""

import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Optional, Union
from benchmarks.load import load_dataset
from .evaluate_labels import evaluate_labels
from .get_operator_from_method import get_operator_from_method
from .get_embedding import get_embedding
from .get_clustering import get_clustering
from .get_time_iteration import get_time_iteration
from .get_power_embedding import get_power_embedding


def get_existing_repeats(
    dataset_name: str,
    method_name: str,
    metrics: list = [],
    save_dir="tables/benchmark_raw/",
):
    """
    Get the number of existing repeats for the given arguments. If a metric list is provided,
    only counts repeats that have results for all specified metrics.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    method_name : str
        Name of the method.
    metrics : list, optional
        List of metric names to check for. If empty, counts all repeats regardless of metrics.
    save_dir : str, optional
        Directory where raw results CSV files are stored. Default is "tables/benchmark_raw/".

    Returns
    -------
    int
        Number of existing repeats for the given dataset, method, and metrics.
    """

    file_path = os.path.join(save_dir, f"{dataset_name}__raw.csv")
    if not os.path.exists(file_path):
        return 0
    df = pd.read_csv(file_path)
    existing = df[df["method"] == method_name]
    if existing.empty:
        return 0

    if len(metrics) == 0:
        return existing["repeat"].max() + 1

    existing = existing[existing["metric"].isin(metrics)]
    if existing.empty:
        return 0

    for metric in existing["metric"].unique():
        metric_repeats = existing[existing["metric"] == metric]["repeat"].nunique()
        if metric_repeats < existing["repeat"].nunique():
            return metric_repeats
    return existing["repeat"].nunique()


def save_raw_results(dataset, method_name, repeats, save_dir="tables/benchmark_raw/"):
    """Save raw repeat-level benchmark results (append-only).

    Parameters
    ----------
    dataset : dict
        Dataset configuration dictionary, must contain a "name" key.
    method_name : str
        Name of the method for which results are being saved.
    repeats : list of dict
        List of dictionaries, each containing metric results for one repeat.
    save_dir : str, optional
        Directory where raw results CSV files are stored. Default is "tables/benchmark_raw/".

    Returns
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{dataset['name']}__raw.csv")

    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        mask = df_existing["method"] == method_name
        if mask.any():
            repeat_offset = df_existing.loc[mask, "repeat"].max() + 1
        else:
            repeat_offset = 0
    else:
        repeat_offset = 0

    rows = []
    for i, repeat in enumerate(repeats):
        for metric, value in repeat.items():
            rows.append(
                {
                    "method": method_name,
                    "repeat": i + repeat_offset,
                    "metric": metric,
                    "value": value,
                }
            )
    df_new = pd.DataFrame(rows)

    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df_new.to_csv(filepath, index=False)

    print(f"[{dataset['name']} -- {method_name}] Raw results appended to {filepath}")


def run_one_repeat(
    operator: np.ndarray,
    true_labels: np.ndarray,
    features: np.ndarray,
    num_clusters: int,
    method: dict,
    metrics: list,
    time_iteration: Optional[int] = None,
) -> Optional[Union[float, dict]]:
    """
    Run one repeat of the benchmark with the given arguments.

    Parameters
    ----------
    operator : np.ndarray
        The operator matrix to be used for embedding.
    true_labels : np.ndarray
        The ground truth labels for evaluation.
    features : np.ndarray
        The feature matrix for unsupervised metrics (can be None).
    num_clusters : int
        The number of clusters to find.
    method : dict
        The method configuration, which may include keys like "power_iteration", "projection_type", and "decomposition", see the 'method-list' functions for details.
    metrics : list
        A list of metric configurations to evaluate the predicted labels against the true labels.
    time_iteration : int, optional
        If the method uses power iteration, this specifies the number of iterations to perform. If None, it will be determined automatically.

    Returns
    -------
    dict
        A dictionary mapping metric names to their computed values for this repeat.
    """
    if method.get("power_iteration", False):
        if method.get("time_iteration", None) is not None:
            time_iteration = method["time_iteration"]
        else:
            time_iteration = get_time_iteration(operator, t_max=250)
        method["decomposition"] = "natural"
    else:
        time_iteration = None
    projection_type = method.get("projection_type", "full")
    n_cols = method.get(
        "n_projection_cols", operator.shape[0] // len(np.unique(true_labels))
    )
    if method.get("power_iteration", False):
        if time_iteration is None:
            raise ValueError(
                "time_iteration must be defined for power iteration methods"
            )
        power_matrix = get_power_embedding(
            operator,
            t=time_iteration,
            projection_type=projection_type,
            n_cols=n_cols,
        )
    else:
        power_matrix = operator

    if projection_type == "random":
        embedding = power_matrix
    else:
        if method.get("decomposition", "") == "eigen":
            dim_embedding = num_clusters
        elif method.get("decomposition", "") == "natural":
            dim_embedding = power_matrix.shape[0]

        embedding = get_embedding(
            power_matrix,
            dim=dim_embedding,
            method=method.get("decomposition", "eigen"),
        )
    pred_labels = get_clustering(embedding, num_clusters)

    score_dict = evaluate_labels(
        true_labels=true_labels,
        features=features,
        pred_labels=pred_labels,
        metric=metrics,
    )

    return score_dict


def run_benchmark_for_dataset(
    dataset,
    methods,
    num_repeats=100,
    save_dir="tables/benchmark_raw/",
    metrics=[],
    parallelize_methods: bool = False,
    method_n_jobs: int = 1,
):
    """Run benchmark for one dataset.

    Parameters
    ----------
    dataset : dict
        Dataset configuration dictionary.
    methods : list[dict]
        Method configurations to benchmark.
    num_repeats : int, default=100
        Target number of repeats per method.
    save_dir : str, default="tables/benchmark_raw/"
        Directory where raw benchmark CSV files are written.
    metrics : list, default=[]
        Metrics to evaluate.
    parallelize_methods : bool, default=False
        If True, compute methods in parallel for this dataset.
    method_n_jobs : int, default=1
        Number of parallel jobs when ``parallelize_methods`` is True.
        Use ``-1`` to use all available workers.
    """

    def _compute_repeats_for_method(method):
        """Compute repeats for one method and return data for deferred saving."""
        method_local = dict(method)
        existing_repeats = get_existing_repeats(
            dataset_name=dataset["name"],
            method_name=method_local["name"],
            save_dir=save_dir,
            metrics=[metric["name"] for metric in metrics],
        )
        repeats_needed = num_repeats - existing_repeats
        if repeats_needed <= 0:
            return method_local["name"], [], existing_repeats

        print(
            f"[{dataset['name']}] Running {repeats_needed} repeats for {method_local['name']}"
        )
        operator = get_operator_from_method(method_local, A=adjacency_matrix)

        if method_local.get("projection_type", "full") == "random":
            # Stochastic embedding: re-run each repeat.
            repeats = [
                run_one_repeat(
                    operator=operator,
                    true_labels=true_labels,
                    features=features,
                    num_clusters=num_clusters,
                    method=method_local,
                    metrics=metrics,
                )
                for _ in range(repeats_needed)
            ]
        else:
            # Deterministic embedding: compute once, then only cluster/evaluate.
            if method_local.get("power_iteration", False):
                if method_local.get("time_iteration", None) is not None:
                    time_iteration = method_local["time_iteration"]
                else:
                    time_iteration = get_time_iteration(operator, t_max=250)
                method_local["decomposition"] = "natural"

            if method_local.get("power_iteration", False):
                embedded_operator = get_power_embedding(
                    operator,
                    t=time_iteration,
                    projection_type=method_local.get("projection_type", "full"),
                    n_cols=method_local.get("n_projection_cols", num_clusters),
                )
            else:
                embedded_operator = get_embedding(
                    operator,
                    dim=num_clusters,
                    method=method_local.get("decomposition", "eigen"),
                )

            repeats = [
                evaluate_labels(
                    true_labels=true_labels,
                    features=features,
                    pred_labels=get_clustering(embedded_operator, num_clusters),
                    metric=metrics,
                )
                for _ in range(repeats_needed)
            ]

        return method_local["name"], repeats, existing_repeats

    dataset_function = dataset["function"]
    adjacency_matrix, true_labels, features = load_dataset(
        dataset_function, **dataset.get("parameters", {})
    )
    all_results = []
    num_clusters = len(np.unique(true_labels))

    if parallelize_methods:
        method_results = Parallel(n_jobs=method_n_jobs, prefer="threads")(
            delayed(_compute_repeats_for_method)(method) for method in methods
        )
    else:
        method_results = [_compute_repeats_for_method(method) for method in methods]

    for method_name, repeats, existing_repeats in method_results:
        if len(repeats) == 0:
            continue
        print(
            f"[{dataset['name']}] Completed [{num_repeats - existing_repeats}] repeats for {method_name}"
        )
        save_raw_results(dataset, method_name, repeats, save_dir=save_dir)
        all_results.extend(repeats)

    return all_results
