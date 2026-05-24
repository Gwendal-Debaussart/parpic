"""
Find the maximum block size each method can handle within a 1-minute time budget.
Saves results to a JSON file for easy plotting.
"""

import time
import multiprocessing
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from benchmarks.directed_sbm import directed_sbm
from competitors.parametrized_laplacians import parametrized_laplacian
from utils.get_embedding import get_embedding
from utils.get_time_iteration import get_time_iteration
from utils.get_power_embedding import get_power_embedding
from vertex_measures.sum_deg import sum_deg

RESULTS_PATH = Path(__file__).resolve().parent / "max_blocksize_vs_time_results.json"


def parpic_random(L, y, t):
    return get_power_embedding(L, t=t, projection_type="random")


def parpic_full(L, y, t):
    return get_power_embedding(L, t=t, projection_type="full")


def classical(A, y, t):
    return get_embedding(A, dim=len(np.unique(y)), method="eigen")


METHODS = [
    ("ParPIC-random", parpic_random),
    ("ParPIC-full", parpic_full),
    ("Classical", classical),
]


def generate_graph(block_size: int):
    block_sizes = [block_size]
    N = block_size
    avg_deg = np.log(N)
    p = avg_deg / (N - 1)
    probs = [[p]]
    A, y, _ = directed_sbm(block_sizes=block_sizes, P=probs, seed=42)
    A = np.asarray(A, dtype=float)
    diags = np.array(A.sum(axis=1)).flatten()
    diags[diags == 0] = 1e-10
    D_inv = np.diag(1 / diags)
    P = D_inv @ A
    gamma_val = 0.5
    nu = sum_deg(A, gamma=gamma_val, degree_normalized=True)
    L_par = parametrized_laplacian(P, nu, normalized=True)
    return A, L_par, y


def trial(b, method_name, t, method_func):
    try:
        A, L_par, y = generate_graph(b)
        if method_name.startswith("ParPIC"):
            emb = method_func(L_par, y, t)
        else:
            A_sym = A + A.T
            emb = method_func(A_sym, y, t)
        KMeans(n_clusters=1, n_init=10).fit(emb)
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def find_runtimes(method_name, method_func, t=10, time_budget=120):
    block_sizes = [
        500,
        800,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4250,
        4500,
        5000,
        7000,
        10000,
        15000,
        20000,
    ]
    results = []
    time_budget = 120  # 2 minutes
    for b in block_sizes:
        N = b
        print(f"[{method_name}] Trying block size {b} (N={N})...")
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(target=trial, args=(b, method_name, t, method_func))
        start = time.time()
        p.start()
        p.join(timeout=time_budget)
        elapsed = time.time() - start
        if p.is_alive():
            print(f"  Timeout after {time_budget} seconds!")
            p.terminate()
            p.join()
            results.append({"N": N, "time": None, "timeout": True})
            break
        elif p.exitcode != 0:
            print(f"  Error or crash (exit code {p.exitcode})")
            results.append({"N": N, "time": None, "error": True})
            break
        else:
            print(f"  Finished in {elapsed:.2f}s")
            results.append({"N": N, "time": elapsed, "timeout": False})
    return results


def main():
    all_results = {}
    for name, func in METHODS:
        method_results = find_runtimes(name, func)
        all_results[name] = method_results
        print(f"{name}: completed {len(method_results)} block sizes")
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
