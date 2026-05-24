from utils.run_benchmark import run_benchmark_for_dataset
from competitors.parametrized_laplacians import parametrized_laplacian
from vertex_measures.sum_deg import sum_deg
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from benchmarks.load import dataset_list, load_dataset
import os

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"

"""
Sensitivity experiments for varying gamma and t
Checking how this works out, plot heatmaps afterwards
"""

def get_t_range(n: int, num_points: int = 15) -> list:
    """
    Get a log-ish spaced range of embedding dimensions.
    Denser at low dimensions, sparser at high dimensions.

    :param n: Number of nodes in the graph
    :param num_points: Approximate number of dimension values to test
    :return: Sorted list of unique dimensions to test
    """
    max_dim = min(n - 1, 500)
    log_dims = np.logspace(0, np.log10(max_dim), num=num_points)
    dims = [d for d in log_dims if 1 <= d < n]
    dims = sorted(set(int(round(d)) for d in log_dims))
    dims = [d for d in dims if 1 <= d < n]
    return dims

def sensitivity_experiment_per_dataset(
    dataset,
    methods,
    num_repeats=10,
    vertex_function=sum_deg,
    save_dir=str(TABLES_DIR / "sensitivity_experiments_t_gamma"),
    metrics=[],
):
    """
    Sensitivity experiment per dataset, varying gamma and t
    """
    methods_gamma_t = []
    gammas = [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
    adjacency_matrix, _, _ = load_dataset(
        dataset["function"], **dataset.get("parameters", {})
    )
    ts = get_t_range(adjacency_matrix.shape[0], num_points = min(adjacency_matrix.shape[0] // 5, 100))
    ts = range(1, 51)
    os.makedirs(save_dir, exist_ok=True)
    for method in methods:
        for gamma in gammas:
            for t in ts:
                method_tmp = {
                    **method,
                    "vertex_measure": {
                        "function": vertex_function,
                        "params": {"gamma": gamma},
                    },
                    "time_iteration": t,
                }
                method_tmp["name"] += f"_gamma={gamma}_t={t}"
                methods_gamma_t.append(method_tmp)
    run_benchmark_for_dataset(dataset, methods_gamma_t, num_repeats, save_dir, metrics)


if __name__ == "__main__":
    methods = [
        {
            "name": "Parametrized Random Walk Laplacian",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
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
    exclude = ["cora", "cornell", "texas", "wisconsin", "citeseer", "pubmed", "mnist"]
    datasets = [d for d in datasets if d["name"] not in exclude]
    n_cores = multiprocessing.cpu_count()
    n_jobs = max(8, n_cores // 4)
    num_repeats = 100
    save_dir = str(TABLES_DIR / "gamma_t_sensitivity")
    Parallel(n_jobs=n_jobs)(
        delayed(sensitivity_experiment_per_dataset)(
            dataset, methods, num_repeats, metrics=metrics, save_dir=save_dir
        )
        for dataset in datasets
    )
