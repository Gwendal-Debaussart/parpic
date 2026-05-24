"""Plot embedding-dimension sensitivity curves for one or more datasets."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common_style import get_method_style
from benchmarks.load import load_dataset

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
EMBED_DIR = TABLES_DIR / "embedding_dim_s"


def discover_datasets(gamma: float) -> list[str]:
    """Return datasets that have AMI score files for the requested gamma."""
    gamma_str = f"{gamma:g}"
    pattern = re.compile(r"^ami_scores_(.+)_gamma=([0-9]+(?:\.[0-9]+)?)\.json$")
    datasets = []

    for file_path in EMBED_DIR.glob("ami_scores_*_gamma=*.json"):
        match = pattern.match(file_path.name)
        if not match:
            continue
        dataset_name, file_gamma = match.groups()
        if file_gamma == gamma_str:
            datasets.append(dataset_name)

    return sorted(set(datasets))


def load_curves(
    dataset_name: str, gamma: float
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Load dimensions, AMI means, and AMI stds from JSON outputs."""
    gamma_str = f"{gamma:g}"
    score_path = EMBED_DIR / f"ami_scores_{dataset_name}_gamma={gamma_str}.json"
    var_path = EMBED_DIR / f"ami_vars_{dataset_name}_gamma={gamma_str}.json"

    with score_path.open("r", encoding="utf-8") as f:
        means_map = json.load(f)
    with var_path.open("r", encoding="utf-8") as f:
        vars_map = json.load(f)

    dims = sorted(int(key) for key in means_map.keys())
    means = np.array([means_map[str(dim)] for dim in dims], dtype=float)
    stds = np.array([vars_map[str(dim)] for dim in dims], dtype=float)
    return dims, means, stds


def get_sqrt_n(dataset_name: str) -> float | None:
    """Load dataset and return sqrt(n) to mark the heuristic embedding dimension."""
    try:
        adjacency, _, _ = load_dataset(dataset_name)
        return float(np.sqrt(adjacency.shape[0]))
    except Exception:
        return None


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


def plot_dataset(
    dataset_name: str, gamma: float, output_dir: Path, show: bool
) -> tuple[Path, Path]:
    """Create and save the sensitivity plot for a single dataset.

    Returns tuple of (figure_path, legend_path).
    """
    dims, means, stds = load_curves(dataset_name, gamma)
    style = get_method_style("ParPIC")

    fig, ax = plt.subplots(figsize=(7, 4.2))

    sqrt_n = get_sqrt_n(dataset_name)
    if sqrt_n is not None:
        ymin = float(np.min(means - stds))
        ymax = float(np.max(means + stds))
        ax.vlines(
            x=sqrt_n,
            ymin=ymin,
            ymax=ymax,
            colors="#F96C39",
            linestyles="dashed",
            label=r"Selected dimension ($\sqrt{n}$)",
        )

    ax.plot(
        dims,
        means,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        label="Mean AMI",
    )
    ax.fill_between(dims, means - stds, means + stds, color=style["color"], alpha=0.2)

    ax.set_title(dataset_name)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("AMI")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"embedding_dim_sensitivity_{dataset_name}_gamma={gamma:g}.pdf"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    legend_path = save_legend(ax, output_dir, "embedding_dim_sensitivity_legend.pdf")

    if show:
        plt.show()
    plt.close(fig)
    return output_path, legend_path


def main() -> None:
    """Parse CLI args and generate plots for requested datasets."""
    parser = argparse.ArgumentParser(
        description="Plot embedding-dimension sensitivity curves for different datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to plot. If omitted, all available datasets are used.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma value used in the saved JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR / "embedding_dim_sensitivity",
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display figures interactively."
    )
    args = parser.parse_args()

    if not EMBED_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {EMBED_DIR}")

    datasets = args.datasets if args.datasets else discover_datasets(args.gamma)
    if not datasets:
        raise ValueError(
            f"No embedding-dim result files found in {EMBED_DIR} for gamma={args.gamma:g}."
        )

    legend_path = None
    for i, dataset_name in enumerate(datasets):
        output_path, legend_path = plot_dataset(
            dataset_name, args.gamma, args.output_dir, args.show
        )
        print(f"Saved: {output_path}")
        if i == 0 and legend_path:
            print(f"Saved legend: {legend_path}")


if __name__ == "__main__":
    main()
