import multiprocessing
from pathlib import Path
import sys
from benchmarks.load import dataset_list
from utils.method_list import method_list
from utils.run_benchmark import run_benchmark_for_dataset
from utils.format_results import format_results
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
PROJECT_ROOT = BASE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_benchmark(
    datasets,
    methods,
    num_repeats=100,
    n_jobs=-1,
    save_dir=str(TABLES_DIR / "clustering_benchmark_raw"),
    metrics=[],
):
    """
    Run benchmark for multiple datasets and methods.
    """

    Parallel(n_jobs=n_jobs)(
        delayed(run_benchmark_for_dataset)(
            dataset, methods, num_repeats, save_dir, metrics
        )
        for dataset in datasets
    )


if __name__ == "__main__":
    methods = method_list()
    datasets = dataset_list()
    exclude = []
    datasets = [d for d in datasets if d["name"] not in exclude]
    choosen_dataset_name = [dataset["name"] for dataset in datasets]
    print("Datasets selected for benchmarking:", choosen_dataset_name)
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

    n_cores = multiprocessing.cpu_count()
    n_jobs = max(8, n_cores // 4)

    run_benchmark(
        datasets,
        methods,
        num_repeats=100,
        n_jobs=n_jobs,
        save_dir=str(TABLES_DIR / "clustering_benchmark_raw"),
        metrics=metrics,
    )
    for dataset in datasets:
        format_results(
            dataset["name"],
            results_dir=str(TABLES_DIR / "clustering_benchmark_raw"),
            format_dir=str(TABLES_DIR / "clustering_benchmark"),
        )
