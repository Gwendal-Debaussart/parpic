from utils.get_power_embedding import get_power_embedding
from utils.get_time_iteration import get_time_iteration
from benchmarks.directed_sbm import directed_sbm
from vertex_measures.sum_deg import sum_deg
from competitors.parametrized_laplacians import parametrized_laplacian
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
import numpy as np
import networkx as nx
from benchmarks import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
RHO_DIR = TABLES_DIR / "rho_sensitivity"

DISBM_TYPE = "cp"  # "cp" or "chain"


def process_single_run(args):
    """Process a single run for both symmetric and parametrized embeddings."""
    L_sym, L_par, y, t_sym, t_par = args

    L_emb_sym = get_power_embedding(L_sym, t=t_sym, projection_type="random")
    L_par_emb = get_power_embedding(L_par, t=t_par, projection_type="random")

    k_means_sym = KMeans(n_clusters=len(np.unique(y)), n_init=100).fit(L_emb_sym)
    y_pred_sym = k_means_sym.labels_

    k_means_par = KMeans(n_clusters=len(np.unique(y)), n_init=100).fit(L_par_emb)
    y_pred_par = k_means_par.labels_

    ami_sym = ami(y, y_pred_sym)
    ami_par = ami(y, y_pred_par)

    return ami_sym, ami_par


def process_parameter_combination(first_block_size, rho, verbose=1):
    """Process a single combination of first_block_size and rho."""

    if DISBM_TYPE == "cp":
        block_sizes = np.array([first_block_size, 400, 400])
        probs = [
            [0.05, rho, rho],
            [0.01, 0.05, 0.01],
            [0.01, 0.01, 0.05],
        ]
    else:  # chain
        block_sizes = np.array([400, 400, 400])
        probs = [
            [0.05, rho, 0.0],
            [0.01, 0.05, rho],
            [0.01, 0.01, 0.05],
        ]

    A, y, _ = directed_sbm(block_sizes=block_sizes, P=probs)

    diags = np.array(A.sum(axis=1)).flatten()
    diags[diags == 0] = 1e-12
    D_inv = np.diag(1 / diags)
    P = D_inv @ A

    A_sym_norm = (A + A.T) / 2.0
    d_sym = np.asarray(A_sym_norm.sum(axis=1)).flatten()
    d_sym[d_sym == 0] = 1e-10
    D_inv_sym = np.diag(1 / d_sym)
    L_sym = D_inv_sym @ A_sym_norm

    gamma_val = 0.5
    nu = sum_deg(A, gamma=gamma_val, degree_normalized=True)
    L_par = parametrized_laplacian(P, nu, normalized=True)

    # Time iterations have been fixed to reduce runtime, note that the value selected is based on the get_time_iteration function. As, on avereage the time selected for both symmetric and parametrized is around 6, we select this value for all runs to reduce runtime. This is a bit of a hack, but it allows us to get results in a reasonable time frame.
    # t_par = get_time_iteration(L_par, t_max=250)
    t_par = 6
    t_sym = t_par

    args_list = [(L_sym, L_par, y, t_sym, t_par) for _ in range(10)]

    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count())) as executor:
        results = list(executor.map(process_single_run, args_list))

    ami_scores_sym = [r[0] for r in results]
    ami_scores_par = [r[1] for r in results]

    # Calculate stats.
    reciprocity = (A * A.T).sum() / A.sum()
    mean_ami_sym = np.mean(ami_scores_sym)
    std_ami_sym = np.std(ami_scores_sym)
    mean_ami_par = np.mean(ami_scores_par)
    std_ami_par = np.std(ami_scores_par)

    if verbose:
        print(
            f"rho: {rho}, first_block: {first_block_size} reciprocity: {reciprocity:.4f}"
        )
        print(f"select times: :(t_sym={t_sym}, t_par={t_par})")
        print("---------------------------------")
        print(f"(symmetric) AMI scores over 10 runs: {mean_ami_sym} +- {std_ami_sym}")
        print(f"(param) AMI scores over 10 runs: {mean_ami_par} +- {std_ami_par}")
        print("================================")

    return {
        "first_block_size": first_block_size,
        "rho": rho,
        "mean_ami_par": mean_ami_par,
        "std_ami_par": std_ami_par,
        "mean_ami_sym": mean_ami_sym,
        "std_ami_sym": std_ami_sym,
    }


def main(verbose=1):
    RHO_DIR.mkdir(parents=True, exist_ok=True)
    rho_means = {}
    rho_stds = {}
    if DISBM_TYPE == "cp":
        first_block_size_list = [
            40,
            80,
            120,
            160,
            200,
            240,
            280,
            320,
            360,
            400,
            500,
            600,
            700,
            800,
        ]
    else:
        first_block_size_list = [500]
    rho_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    param_combinations = [
        (first_block_size, rho)
        for first_block_size in first_block_size_list
        for rho in rho_list
    ]

    max_workers = min(os.cpu_count(), len(param_combinations))
    max_workers = max_workers // 4 if max_workers >= 4 else max_workers
    if verbose:
        print(
            f"Processing {len(param_combinations)} parameter combinations with {max_workers} workers..."
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_parameter_combination, fbs, r, verbose): (fbs, r)
            for fbs, r in param_combinations
        }

        for future in as_completed(futures):
            result = future.result()
            first_block_size = result["first_block_size"]
            rho = result["rho"]

            rho_means[f"par_{rho}_{first_block_size}"] = result["mean_ami_par"]
            rho_means[f"sym_{rho}_{first_block_size}"] = result["mean_ami_sym"]
            rho_stds[f"par_{rho}_{first_block_size}"] = result["std_ami_par"]
            rho_stds[f"sym_{rho}_{first_block_size}"] = result["std_ami_sym"]

    np.save(RHO_DIR / f"{DISBM_TYPE}_rho_means.npy", rho_means)
    np.save(RHO_DIR / f"{DISBM_TYPE}_rho_stds.npy", rho_stds)
    if verbose:
        print(
            f"Results saved to {RHO_DIR / f'{DISBM_TYPE}_rho_means.npy'} and {RHO_DIR / f'{DISBM_TYPE}_rho_stds.npy'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rho sensitivity experiment")
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level (0=silent, 1=default)"
    )
    args = parser.parse_args()
    main(verbose=args.verbose)
