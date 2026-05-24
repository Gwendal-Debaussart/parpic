"""
Sensitivity analysis for DiSBM core-periphery structure:
Two-parameter analysis varying rho and first cluster size
"""

import json
from joblib import Parallel, delayed
import time
import numpy as np
from benchmarks.directed_sbm import directed_sbm
from competitors.parametrized_laplacians import parametrized_laplacian
from vertex_measures.sum_deg import sum_deg
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from utils.get_power_embedding import get_power_embedding
import os
import argparse
import gc
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"


def run_single_cp_iteration(run_id, Lds_param, Lds_sym, y, t_val, n_clusters):
    """
    Run one iteration on pre-computed Laplacians.
    Returns scores for both parametrized and symmetric Laplacians.

    Parameters:
    ----------

    run_id: int
        Unique identifier for the run (used for random seed)
    Lds_param: np.ndarray
        Parametrized Laplacian matrix
    Lds_sym: np.ndarray
        Symmetric Laplacian matrix
    y: np.ndarray
        Ground truth labels
    t_val: int
        Time iteration parameter for embedding
    n_clusters: int
        Number of clusters for KMeans

    Returns:
    -------

    tuple: (score_param, score_sym)
        score_param: AMI score for parametrized Laplacian
        score_sym: AMI score for symmetric Laplacian
    """
    try:
        np.random.seed(42 + run_id)
        L_emb_param = get_power_embedding(Lds_param, t=t_val, projection_type="random")
        if hasattr(L_emb_param, 'toarray'):
            L_emb_param = L_emb_param.toarray()
        else:
            L_emb_param = np.asarray(L_emb_param)

        L_emb_sym = get_power_embedding(Lds_sym, t=t_val, projection_type="random")
        if hasattr(L_emb_sym, 'toarray'):
            L_emb_sym = L_emb_sym.toarray()
        else:
            L_emb_sym = np.asarray(L_emb_sym)

        k_means_param = KMeans(n_clusters=n_clusters, n_init=50, random_state=42 + run_id).fit(L_emb_param)
        y_pred_param = k_means_param.labels_
        score_param = ami(y, y_pred_param)

        k_means_sym = KMeans(n_clusters=n_clusters, n_init=50, random_state=1000 + run_id).fit(L_emb_sym)
        y_pred_sym = k_means_sym.labels_
        score_sym = ami(y, y_pred_sym)

        del L_emb_param, L_emb_sym, k_means_param, k_means_sym
        gc.collect()

        return score_param, score_sym

    except Exception as e:
        print(f"Error in run_single_cp_iteration: {e}")
        return 0.0, 0.0


def process_cp_parameter_combination(first_cluster_size, rho_val, gamma_val, t_val, n_runs, n_jobs, verbose=1):
    """Process a single combination of parameters."""
    if verbose:
        print(f"Processing: first_cluster={first_cluster_size}, rho={rho_val:.3f}, gamma={gamma_val:.2f}, t={t_val}")

    try:
        second_cluster_size = 1300
        third_cluster_size = 1300

        block_sizes = [first_cluster_size, second_cluster_size, third_cluster_size]
        probs = [
            [0.05, rho_val, rho_val],
            [0.02, 0.05, 0.02],
            [0.02, 0.02, 0.05],
        ]

        A, y, _ = directed_sbm(block_sizes=block_sizes, P=probs, seed=42)
        A = A.astype(float)
        n_clusters = len(np.unique(y))

        diags = np.array(A.sum(axis=1)).flatten()
        diags[diags == 0] = 1e-12
        D_inv = np.diag(1 / diags)
        P_matrix = D_inv @ A

        nu = sum_deg(A, gamma=gamma_val)
        Lds_param = parametrized_laplacian(P_matrix, nu, normalized=True)

        if hasattr(Lds_param, 'toarray'):
            Lds_param = Lds_param.toarray()
        else:
            Lds_param = np.asarray(Lds_param)

        A_sym = (A + A.T) / 2.0
        d_sym = np.array(A_sym.sum(axis=1)).flatten()
        d_sym[d_sym == 0] = 1e-12
        D_inv_sym = np.diag(1 / d_sym)
        Lds_sym = D_inv_sym @ A_sym

        if hasattr(Lds_sym, 'toarray'):
            Lds_sym = Lds_sym.toarray()
        else:
            Lds_sym = np.asarray(Lds_sym)

        del A, A_sym, P_matrix, D_inv, D_inv_sym, diags, d_sym
        gc.collect()

        results = Parallel(n_jobs=n_jobs, backend='threading', verbose=max(0, int(verbose) - 1))(
            delayed(run_single_cp_iteration)(i, Lds_param, Lds_sym, y, t_val, n_clusters)
            for i in range(n_runs)
        )

        del Lds_param, Lds_sym
        gc.collect()

    except Exception as e:
        print(f"Error generating graph/Laplacians: {e}")
        return {
            'first_cluster_size': first_cluster_size,
            'rho': rho_val,
            'gamma': gamma_val,
            't': t_val,
            'mean_ami_param': 0.0,
            'std_ami_param': 0.0,
            'mean_ami_sym': 0.0,
            'std_ami_sym': 0.0
        }

    scores_param = [r[0] for r in results]
    scores_sym = [r[1] for r in results]

    mean_score_param = np.mean(scores_param)
    std_score_param = np.std(scores_param)
    mean_score_sym = np.mean(scores_sym)
    std_score_sym = np.std(scores_sym)

    gc.collect()

    return {
        'first_cluster_size': first_cluster_size,
        'rho': rho_val,
        'gamma': gamma_val,
        't': t_val,
        'mean_ami_param': mean_score_param,
        'std_ami_param': std_score_param,
        'mean_ami_sym': mean_score_sym,
        'std_ami_sym': std_score_sym
    }


def main(args, verbose=None):
    """Main experiment."""
    verbose = args.verbose if verbose is None else verbose

    if verbose:
        print("="*80)
        print("DiSBM Core-Periphery Sensitivity Analysis")
        print("Two-parameter analysis: rho vs first_cluster_size")
        print("="*80)

    first_cluster_sizes = args.cluster_sizes or [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    rho_values = args.rho_values or [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    gamma_val = args.gamma
    t_val = args.t
    n_runs = args.n_runs
    n_jobs = args.n_jobs

    os.makedirs(args.output_dir, exist_ok=True)

    param_combinations = [(size, rho) for size in first_cluster_sizes for rho in rho_values]

    if verbose:
        print(f"Total combinations to process: {len(param_combinations)}")
        print(f"Runs per combination: {n_runs}")
        print(f"Fixed parameters: gamma={gamma_val}, t={t_val}")
        print(f"Parallel jobs: {n_jobs}")

    start_time = time.time()
    all_results = []
    for i, (size, rho) in enumerate(param_combinations):
        try:
            if verbose:
                print(f"\n[{i+1}/{len(param_combinations)}]", end=" ")
            result = process_cp_parameter_combination(size, rho, gamma_val, t_val, n_runs, n_jobs, verbose=verbose)
            all_results.append(result)
            gc.collect()
        except Exception as e:
            print(f"ERROR at combination {i+1}: {e}")
            continue

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n\n{'='*80}")
        print(f"All experiments completed in {elapsed_time/60:.1f} minutes!")
        print(f"{'='*80}")

    results_dict = {
        'means_param': {},
        'stds_param': {},
        'means_sym': {},
        'stds_sym': {},
        'parameters': {
            'first_cluster_sizes': first_cluster_sizes,
            'rho_values': rho_values,
            'gamma': gamma_val,
            't': t_val,
            'n_runs': n_runs
        }
    }

    for result in all_results:
        key = f"size_{result['first_cluster_size']}_rho_{result['rho']:.3f}"
        results_dict['means_param'][key] = result['mean_ami_param']
        results_dict['stds_param'][key] = result['std_ami_param']
        results_dict['means_sym'][key] = result['mean_ami_sym']
        results_dict['stds_sym'][key] = result['std_ami_sym']

    output_file = os.path.join(args.output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    if verbose:
        print(f"\nResults saved to {output_file}")

    if verbose:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
    all_means_param = [r['mean_ami_param'] for r in all_results]
    all_means_sym = [r['mean_ami_sym'] for r in all_results]

    if verbose:
        print("\nParametrized Laplacian:")
        print(f"  Overall mean AMI: {np.mean(all_means_param):.4f}")
        print(f"  Overall std AMI: {np.std(all_means_param):.4f}")
        print(f"  Min AMI: {np.min(all_means_param):.4f}")
        print(f"  Max AMI: {np.max(all_means_param):.4f}")

    if verbose:
        print("\nSymmetric Laplacian (A + A.T):")
        print(f"  Overall mean AMI: {np.mean(all_means_sym):.4f}")
        print(f"  Overall std AMI: {np.std(all_means_sym):.4f}")
        print(f"  Min AMI: {np.min(all_means_sym):.4f}")
        print(f"  Max AMI: {np.max(all_means_sym):.4f}")


    if verbose:
        best_result_param = max(all_results, key=lambda x: x['mean_ami_param'])
        print(f"\nBest configuration (Parametrized):")
        print(f"  First cluster size: {best_result_param['first_cluster_size']}")
        print(f"  Rho: {best_result_param['rho']:.3f}")
        print(f"  Mean AMI: {best_result_param['mean_ami_param']:.4f} +/- {best_result_param['std_ami_param']:.4f}")

        best_result_sym = max(all_results, key=lambda x: x['mean_ami_sym'])
        print(f"\nBest configuration (Symmetric):")
        print(f"  First cluster size: {best_result_sym['first_cluster_size']}")
        print(f"  Rho: {best_result_sym['rho']:.3f}")
        print(f"  Mean AMI: {best_result_sym['mean_ami_sym']:.4f} +/- {best_result_sym['std_ami_sym']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiSBM CP sensitivity analysis')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for vertex measure')
    parser.add_argument('--t', type=int, default=6, help='Time iteration parameter')
    parser.add_argument('--n_runs', type=int, default=50, help='Number of runs per parameter combination')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--output_dir', type=str, default=str(TABLES_DIR / 'disbm_cp_cluster_sensitivity'),
                        help='Output directory for results')
    parser.add_argument('--cluster_sizes', type=int, nargs='+',
                        help='List of first cluster sizes to test')
    parser.add_argument('--rho_values', type=float, nargs='+',
                        help='List of rho values to test')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=silent, 1=default, >=2 includes joblib progress)')

    args = parser.parse_args()
    main(args)
