"""Generate elbow plots (exact vs approximated entropy) for all datasets.

This script computes diffusion entropy trajectories and knee points for each
benchmark dataset, then saves one plot per dataset and a JSON summary.
"""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from kneed import KneeLocator

from benchmarks.load import dataset_list, load_dataset
from utils.entropies import diffusion_entropy


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "figures" / "elbows_n_probes"
TABLES_DIR = BASE_DIR / "tables"


def to_numpy_array(matrix):
    """Convert dense/sparse matrix-like object to numpy array."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def largest_weak_component(A: np.ndarray, y: np.ndarray):
    """Restrict A and y to the largest weakly connected component."""
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    idx = np.array(sorted(largest_wcc))
    return A[np.ix_(idx, idx)], y[idx]


def compute_transition_matrix(A: np.ndarray) -> np.ndarray:
    """Compute row-stochastic transition matrix from adjacency."""
    diags = np.array(A.sum(axis=1)).flatten()
    diags[diags == 0] = 1e-12
    D_inv = np.diag(1 / diags)
    return D_inv @ A


def compute_elbows(P: np.ndarray, t_max: int, n_probes: int):
    """Compute exact/approx entropy curves and knee points."""
    entropy_exact = -diffusion_entropy(P, max_t=t_max, n_probes=0)
    entropy_approx = -diffusion_entropy(P, max_t=t_max, n_probes=n_probes)

    t_axis = np.arange(1, t_max + 1)
    knee_exact = KneeLocator(
        t_axis, entropy_exact, curve="concave", direction="increasing"
    ).knee
    knee_approx = KneeLocator(
        t_axis, entropy_approx, curve="concave", direction="increasing"
    ).knee
    return t_axis, entropy_exact, entropy_approx, knee_exact, knee_approx


def save_legend(ax, output_dir: Path, legend_name: str = "legend.pdf") -> Path:
    """Extract and save legend from axis as a standalone figure."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    fig_legend = plt.figure(figsize=(5, 2))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")
    fig_legend.legend(handles, labels, loc="center", frameon=False, fontsize=11)

    output_dir.mkdir(parents=True, exist_ok=True)
    legend_path = output_dir / legend_name
    fig_legend.savefig(legend_path, dpi=300, bbox_inches="tight")
    plt.close(fig_legend)
    return legend_path


def plot_elbow(
    dataset_name: str,
    t_axis: np.ndarray,
    entropy_exact: np.ndarray,
    entropy_approx: np.ndarray,
    knee_exact,
    knee_approx,
    output_path: Path,
) -> Path:
    """Save elbow plot for one dataset. Returns legend path if extracted."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t_axis, entropy_exact, label="Exact Entropy", c="#072AC8")
    ax.plot(
        t_axis,
        entropy_approx,
        linestyle="--",
        label="Approximated Entropy",
        c="#F96C39",
    )

    if knee_exact is not None:
        ax.axvline(
            x=knee_exact,
            color="#D90429",
            linestyle="-",
            label=f"Knee Exact: {knee_exact}",
        )
    if knee_approx is not None:
        ax.axvline(
            x=knee_approx,
            color="#2B9348",
            linestyle=":",
            label=f"Knee Approx: {knee_approx}",
        )

    ax.set_title(dataset_name)
    ax.set_xlabel("Time step t")
    ax.set_ylabel("Entropy")
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")

    legend_path = save_legend(ax, output_path.parent, "elbows_n_probes_legend.pdf")

    plt.close(fig)
    return legend_path


def run_dataset(dataset_cfg: dict, t_max: int, use_wcc: bool) -> tuple[dict, Path]:
    """Process one dataset config and return (summary_row, legend_path)."""
    name = dataset_cfg["name"]
    params = dataset_cfg.get("parameters", dataset_cfg.get("args", {}))

    A, y, _ = load_dataset(dataset_cfg["function"], **params)
    A = to_numpy_array(A).astype(float)
    y = np.asarray(y)

    if use_wcc:
        A, y = largest_weak_component(A, y)

    k = max(1, len(np.unique(y)))
    n_probes = int(np.sqrt(A.shape[0]))
    n_probes = max(1, min(n_probes, A.shape[0], A.shape[0] // k if k > 0 else n_probes))

    P = compute_transition_matrix(A)
    t_axis, entropy_exact, entropy_approx, knee_exact, knee_approx = compute_elbows(
        P=P,
        t_max=t_max,
        n_probes=n_probes,
    )

    suffix = "_wcc" if use_wcc else ""
    output_path = FIGURES_DIR / f"{name}{suffix}_entropy_approximation_sqrt.pdf"
    legend_path = plot_elbow(
        dataset_name=f"{name}{suffix}",
        t_axis=t_axis,
        entropy_exact=entropy_exact,
        entropy_approx=entropy_approx,
        knee_exact=knee_exact,
        knee_approx=knee_approx,
        output_path=output_path,
    )

    return (
        {
            "dataset": f"{name}{suffix}",
            "n_nodes": int(A.shape[0]),
            "n_probes": int(n_probes),
            "knee_exact": None if knee_exact is None else int(knee_exact),
            "knee_approx": None if knee_approx is None else int(knee_approx),
            "figure": str(output_path),
        },
        legend_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate elbow plots for all benchmark datasets"
    )
    parser.add_argument("--t_max", type=int, default=50, help="Maximum diffusion time")
    parser.add_argument(
        "--wcc",
        action="store_true",
        help="Restrict each dataset to its largest weakly connected component",
    )
    args = parser.parse_args()

    datasets = [d for d in dataset_list() if d["function"] in dataset_list_functions()]
    summaries = []
    legend_path = None

    for i, dataset_cfg in enumerate(datasets):
        try:
            summary, lex_path = run_dataset(
                dataset_cfg=dataset_cfg, t_max=args.t_max, use_wcc=args.wcc
            )
            summaries.append(summary)
            if i == 0:
                legend_path = lex_path
            print(
                f"[{summary['dataset']}] knees: exact={summary['knee_exact']} approx={summary['knee_approx']}"
            )
        except Exception as exc:
            print(f"[WARN] Failed for dataset '{dataset_cfg['name']}': {exc}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = TABLES_DIR / (
        "elbows_n_probes_wcc.json" if args.wcc else "elbows_n_probes.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Saved {len(summaries)} elbow summaries to {summary_file}")
    if legend_path:
        print(f"Saved legend: {legend_path}")


def dataset_list_functions():
    """Return names of dataset loader functions exposed by benchmark registry."""
    # Keep this local to avoid broad wildcard imports.
    return set(d["function"] for d in dataset_list())


if __name__ == "__main__":
    main()
