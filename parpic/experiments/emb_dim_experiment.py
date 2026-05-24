import numpy as np
from pathlib import Path
from utils.run_benchmark import run_benchmark_for_dataset
from utils.format_results import format_results
from benchmarks.load import load_dataset
from benchmarks.load import dataset_list
from joblib import Parallel, delayed
from competitors.parametrized_laplacians import parametrized_laplacian
from vertex_measures import sum_deg
import multiprocessing
import os
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"


def get_dimension_range(n: int, num_points: int = 15) -> list:
    """
    Get a log-ish spaced range of embedding dimensions.
    Denser at low dimensions, sparser at high dimensions.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    num_points : int, default=15
        Approximate number of dimension values to test.
    """
    max_dim = min(n - 1, 500)
    log_dims = np.logspace(0, np.log10(max_dim), num=num_points)
    dims = [d for d in log_dims if 1 <= d < n]
    dims = sorted(set(int(round(d)) for d in log_dims))
    dims = [d for d in dims if 1 <= d < n]
    return dims


def run_sensitivity_experiment(
    dataset,
    methods,
    num_repeats=10,
    save_dir=str(TABLES_DIR / "sensitivity_experiments"),
    metrics=[],
):
    """
    Run sensitivity experiments for multiple datasets and methods.
    """
    os.makedirs(save_dir, exist_ok=True)
    adjacency_matrix, _, _ = load_dataset(
        dataset["function"], **dataset.get("parameters", {})
    )
    dimensions = get_dimension_range(
        adjacency_matrix.shape[0], num_points=adjacency_matrix.shape[0] // 5
    )
    print("dimensions to test:", dimensions)
    methods_dim = []
    for method in methods:
        for dim in dimensions:
            method_dim = {
                    **method,
                    "n_projection_cols": dim,
                }
            method_dim["name"] += f"_d={dim}"
            methods_dim.append(method_dim)
    run_benchmark_for_dataset(dataset, methods_dim, num_repeats, save_dir, metrics)


if __name__ == "__main__":

    methods = [
        {
            "name": "(random-proj) Parametrized Random Walk Laplacian gamma = 0.5",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.5}},
        },
        {
            "name": "(random-proj) Parametrized Random Walk Laplacian gamma = 0.15",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.15}},
        },
    ]
    metrics = [
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
    ]
    datasets = dataset_list()
    exclude = ["cora", "cornell", "texas", "wisconsin", "citeseer", "pubmed"]
    datasets = [d for d in datasets if d["name"] not in exclude]
    n_cores = multiprocessing.cpu_count()
    n_jobs = max(8, n_cores // 4)
    num_repeats = 100
    save_dir = str(TABLES_DIR / "sensitivity_experiments")
    Parallel(n_jobs=n_jobs)(
        delayed(run_sensitivity_experiment)(
            dataset, methods, num_repeats, metrics=metrics, save_dir=save_dir
        )
        for dataset in datasets
    )

    for dataset in datasets:
        format_results(
            dataset["name"],
            results_dir=str(TABLES_DIR / "sensitivity_experiments"),
            format_dir=str(TABLES_DIR / "sensitivity_experiments/formatted"),
        )