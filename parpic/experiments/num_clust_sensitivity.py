"""Sensitivity experiment on the number of clusters using benchmark workflow.

This script generates directed SBM datasets with varying numbers of clusters and
runs the benchmark pipeline for each configuration.
"""

from pathlib import Path
import json
import multiprocessing
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_mutual_info_score

from benchmarks.load import dataset_functions
from competitors.dd_sym import dd_sym
from competitors.dsc_plus import dsc_plus
from competitors.parametrized_laplacians import parametrized_laplacian
from competitors.sym_matrix import sym_matrix
from utils.format_results import format_results, format_all_datasets
from utils.run_benchmark import run_benchmark_for_dataset
from vertex_measures.sum_deg import sum_deg


MODEL_TYPE = "cp"
NUM_CLUSTERS_LIST = [3, 4, 5, 6, 8, 10, 12, 15]
BLOCK_SIZE = 500
NUM_REPEATS = 50

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
RAW_DIR = TABLES_DIR / f"disbm_{MODEL_TYPE}_cluster_sensitivity_raw"
FORMAT_DIR = TABLES_DIR / f"disbm_{MODEL_TYPE}_cluster_sensitivity"
MEANS_JSON = TABLES_DIR / f"disbm_{MODEL_TYPE}_cluster_sensitivity_means.json"
VARS_JSON = TABLES_DIR / f"disbm_{MODEL_TYPE}_cluster_sensitivity_vars.json"


def build_probability_matrix(num_clusters: int, model_type: str) -> np.ndarray:
    """Build a directed SBM block probability matrix for chain/core-periphery."""
    probs = np.zeros((num_clusters, num_clusters))

    if model_type == "chain":
        rho = 0.6
        for i in range(num_clusters):
            probs[i, i] = 0.05
            if i < num_clusters - 1:
                probs[i, i + 1] = rho
            if i > 0:
                probs[i, i - 1] = 0.02
    elif model_type == "cp":
        rho = 0.4
        for i in range(num_clusters):
            probs[i, i] = 0.05
            if i == 0:
                for j in range(1, num_clusters):
                    probs[i, j] = rho
            else:
                for j in range(1, num_clusters):
                    if i != j:
                        probs[i, j] = 0.02
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'chain' or 'cp'.")

    return probs


def build_dataset_config(num_clusters: int, model_type: str, block_size: int) -> dict:
    """Build benchmark dataset config for one cluster-count setting."""
    block_sizes = [block_size] * num_clusters
    probs = build_probability_matrix(num_clusters=num_clusters, model_type=model_type)
    return {
        "name": f"disbm_{model_type}_k{num_clusters}",
        "function": "directed_sbm",
        "parameters": {
            "block_sizes": block_sizes,
            "P": probs,
            "seed": 42,
        },
    }


def method_list() -> list:
    """Methods evaluated in the benchmark formulation."""
    return [
        {
            "name": "Parametrized Random Walk Laplacian",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.5}},
            "time_iteration": 5,
        },
        {
            "name": "Symmetric Random Walk",
            "function": sym_matrix,
            "input_type": "adjacency",
            "projection_type": "random",
            "power_iteration": True,
            "params": {},
            "time_iteration": 5,
        },
        {
            "name": "DD Sym",
            "function": dd_sym,
            "input_type": "adjacency",
            "decomposition": "eigen",
            "projection_type": "full",
            "power_iteration": False,
            "params": {},
        },
        {
            "name": "DSC Plus",
            "function": dsc_plus,
            "input_type": "adjacency",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"gamma": 0.9},
            "time_iteration": 5,
        },
    ]


def metric_list() -> list:
    """Metrics used for this sensitivity benchmark."""
    return [
        {
            "name": "Adjusted Mutual Information",
            "function": adjusted_mutual_info_score,
            "type": "supervised",
        },
    ]


def run_for_dataset(dataset: dict, methods: list, metrics: list, num_repeats: int) -> None:
    """Run one dataset benchmark and write raw repeat-level outputs."""
    run_benchmark_for_dataset(
        dataset=dataset,
        methods=methods,
        num_repeats=num_repeats,
        save_dir=str(RAW_DIR),
        metrics=metrics,
        parallelize_methods=False,
    )





def export_summary_json(datasets: list) -> None:
    """Export compact means/std JSONs keyed by method and number of clusters."""
    means = {}
    vars_ = {}
    method_key_map = {
        "Parametrized Random Walk Laplacian": "par",
        "Symmetric Random Walk": "sym",
        "DD Sym": "dd",
        "DSC Plus": "dsc",
    }

    for dataset in datasets:
        dataset_name = dataset["name"]
        k = dataset_name.split("_k")[-1]
        formatted_path = FORMAT_DIR / f"{dataset_name}__formatted.csv"
        if not formatted_path.exists():
            continue
        df = pd.read_csv(formatted_path)
        df = df[df["metric"] == "Adjusted Mutual Information"]
        for _, row in df.iterrows():
            short_name = method_key_map.get(row["method"])
            if short_name is None:
                continue
            means[f"{short_name}_{k}"] = float(row["mean_val"])
            vars_[f"{short_name}_{k}"] = float(row["std_val"])

    with open(MEANS_JSON, "w") as f:
        json.dump(means, f, indent=2)
    with open(VARS_JSON, "w") as f:
        json.dump(vars_, f, indent=2)


if __name__ == "__main__":
    if "directed_sbm" not in dataset_functions():
        raise RuntimeError("Dataset function 'directed_sbm' is not available.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FORMAT_DIR.mkdir(parents=True, exist_ok=True)

    methods = method_list()
    metrics = metric_list()
    datasets = [
        build_dataset_config(
            num_clusters=num_clusters,
            model_type=MODEL_TYPE,
            block_size=BLOCK_SIZE,
        )
        for num_clusters in NUM_CLUSTERS_LIST
    ]

    print(
        f"Running benchmark formulation for {len(datasets)} cluster settings "
        f"(model={MODEL_TYPE}, repeats={NUM_REPEATS})."
    )

    n_cores = multiprocessing.cpu_count()
    n_jobs = min(len(datasets), max(1, n_cores // 4))
    Parallel(n_jobs=n_jobs)(
        delayed(run_for_dataset)(
            dataset=dataset,
            methods=methods,
            metrics=metrics,
            num_repeats=NUM_REPEATS,
        )
        for dataset in datasets
    )

    format_all_datasets(datasets, raw_dir=str(RAW_DIR), format_dir=str(FORMAT_DIR))

    export_summary_json(datasets)
    print("Benchmark run complete.")
    print(f"Raw results: {RAW_DIR}")
    print(f"Formatted results: {FORMAT_DIR}")
    print(f"Summary means: {MEANS_JSON}")
    print(f"Summary stds: {VARS_JSON}")
