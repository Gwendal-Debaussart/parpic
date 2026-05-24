"""Benchmark runtime scaling for ParPIC vs classical spectral clustering.

This script generates synthetic directed SBM graphs for multiple sizes,
measures average runtime, and saves a runtime scaling plot.
"""

from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.directed_sbm import directed_sbm
from competitors.parametrized_laplacians import parametrized_laplacian
from utils.get_embedding import get_embedding
from utils.get_power_embedding import get_power_embedding
from vertex_measures.sum_deg import sum_deg


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "figures" / "runtimes"


def generate_graph(block_size: int):
    """Generate DiSBM graph and operators for one block size."""
    block_sizes = [block_size, block_size, block_size]
    probs = [
        [0.05, 0.01, 0.01],
        [0.01, 0.05, 0.01],
        [0.00, 0.01, 0.05],
    ]
    A, y, _ = directed_sbm(block_sizes=block_sizes, P=probs, seed=42)
    A = np.asarray(A, dtype=float)

    diags = np.array(A.sum(axis=1)).flatten()
    diags[diags == 0] = 1e-10
    D_inv = np.diag(1 / diags)
    P = D_inv @ A

    A_sym_norm = (A + A.T) / 2.0
    d_sym = np.array(A_sym_norm.sum(axis=1)).flatten()
    d_sym[d_sym == 0] = 1e-10
    D_inv_sym = np.diag(1 / d_sym)
    L_sym = D_inv_sym @ A_sym_norm

    gamma_val = 0.5
    nu = sum_deg(A, gamma=gamma_val, degree_normalized=True)
    L_par = parametrized_laplacian(P, nu, normalized=True)

    return A, L_sym, L_par, np.asarray(y)



import multiprocessing

def _power_clustering_trial(L, y, projection_type, t, result_conn):
    n_clusters = len(np.unique(y))
    start = time.time()
    emb = get_power_embedding(L, t=t, projection_type=projection_type)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50).fit(emb)
    y_pred = kmeans.labels_
    elapsed = time.time() - start
    ami_val = ami(y, y_pred)
    result_conn.send((elapsed, ami_val))

def run_power_clustering(L: np.ndarray, y: np.ndarray, n_trials: int = 10, projection_type: str = "random", t: int = 10, time_max: float = 120):
    """Measure runtime and AMI for power-iteration clustering, with optional per-trial timeout."""
    times = []
    amis = []
    n_clusters = len(np.unique(y))
    for _ in range(n_trials):
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_power_clustering_trial, args=(L, y, projection_type, t, child_conn))
        p.start()
        p.join(timeout=time_max)
        if p.is_alive():
            p.terminate()
            p.join()
            times.append(None)
            amis.append(None)
        elif parent_conn.poll():
            elapsed, ami_val = parent_conn.recv()
            times.append(elapsed)
            amis.append(ami_val)
        else:
            times.append(None)
            amis.append(None)
    # Filter out None values
    valid_times = [t for t in times if t is not None]
    valid_amis = [a for a in amis if a is not None]
    return (float(np.mean(valid_times)) if valid_times else None,
            float(np.mean(valid_amis)) if valid_amis else None,
            float(np.std(valid_amis)) if len(valid_amis) > 1 else 0.0,
            times.count(None))



def _classical_spectral_trial(A, y, result_conn):
    import time
    from sklearn.cluster import KMeans
    from utils.get_embedding import get_embedding
    from sklearn.metrics import adjusted_mutual_info_score as ami
    n_clusters = len(np.unique(y))
    start = time.time()
    emb = get_embedding(A, dim=n_clusters, method="eigen")
    kmeans = KMeans(n_clusters=n_clusters, n_init=50).fit(emb)
    y_pred = kmeans.labels_
    elapsed = time.time() - start
    ami_val = ami(y, y_pred)
    result_conn.send((elapsed, ami_val))

def run_classical_spectral(A: np.ndarray, y: np.ndarray, n_trials: int = 10, time_max: float = 120):
    """Measure runtime and AMI for classical spectral + KMeans, with optional per-trial timeout."""
    times = []
    amis = []
    n_clusters = len(np.unique(y))
    for _ in range(n_trials):
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_classical_spectral_trial, args=(A, y, child_conn))
        p.start()
        p.join(timeout=time_max)
        if p.is_alive():
            p.terminate()
            p.join()
            times.append(None)
            amis.append(None)
        elif parent_conn.poll():
            elapsed, ami_val = parent_conn.recv()
            times.append(elapsed)
            amis.append(ami_val)
        else:
            times.append(None)
            amis.append(None)
    valid_times = [t for t in times if t is not None]
    valid_amis = [a for a in amis if a is not None]
    return (float(np.mean(valid_times)) if valid_times else None,
            float(np.mean(valid_amis)) if valid_amis else None,
            float(np.std(valid_amis)) if len(valid_amis) > 1 else 0.0,
            times.count(None))


def plot_runtimes(results: list, output_path: Path, show: bool):
    """Plot and save runtime curves."""
    Ns = [r["N"] for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(Ns, [r["par_time"] for r in results], marker="o", label="ParPIC (low-dim projection)", color="#072AC8")
    ax.plot(
        Ns,
        [r["par_time_full"] for r in results],
        marker="s",
        linestyle="--",
        label="ParPIC (full projection)",
        color="#072AC8",
    )
    ax.plot(Ns, [r["classic_time"] for r in results], marker="^", label="Classical Spectral Clustering", color="#F96C39")

    ax.set_xlabel("Number of nodes (N)")
    ax.set_ylabel("Runtime (seconds)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    print(f"Saved runtime plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate runtime scaling plot")
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[50, 100, 200, 400, 500, 600, 800, 1000, 1500, 2000, 2500], help="Block sizes to test")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of repeated runs per graph size")
    parser.add_argument("--t", type=int, default=10, help="Power iteration depth")
    parser.add_argument("--show", action="store_true", help="Show figure window")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for PDF")
    args = parser.parse_args()

    results = []
    for b in args.block_sizes:
        print(f"Running for block size = {b} (N = {3 * b})")
        A, _, L_par, y = generate_graph(b)
        A_sym = (A + A.T) / 2.0


        time_par, ami_par, std_par, n_timeouts_par = run_power_clustering(L_par, y, n_trials=args.n_trials, projection_type="random", t=args.t, time_max=10)
        time_classic, ami_classic, std_classic, n_timeouts_classic = run_classical_spectral(A_sym, y, n_trials=args.n_trials, time_max=10)
        time_par_full, ami_par_full, std_par_full, n_timeouts_par_full = run_power_clustering(
            L_par,
            y,
            n_trials=args.n_trials,
            projection_type="full",
            t=args.t,
            time_max=10
        )

        results.append(
            {
                "N": 3 * b,
                "par_time": time_par,
                "par_ami": ami_par,
                "classic_time": time_classic,
                "classic_ami": ami_classic,
                "par_time_full": time_par_full,
                "par_ami_full": ami_par_full,
            }
        )

    output_path = Path(args.output) if args.output else FIGURES_DIR / "scaling_runtime.pdf"

    json.dump(
        results,
        open("scaling_directed_sbm.json", "w"),
        indent=4
    )
    plot_runtimes(results=results, output_path=output_path, show=args.show)


if __name__ == "__main__":
    main()
