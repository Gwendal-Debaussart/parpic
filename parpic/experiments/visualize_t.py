"""Plot time-sensitivity curves (AMI vs diffusion time t) for a given dataset and gamma."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common_style import get_method_style

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = TABLES_DIR / "gamma_t_sensitivity"


def discover_dataset_gamma_pairs(results_dir: Path) -> list[tuple[str, float]]:
    """Return list of (dataset_name, gamma) tuples available in results directory."""
    pattern = re.compile(r"^ami_scores_(.+)_gamma=([0-9]+(?:\.[0-9]+)?)\.json$")
    pairs = set()

    for file_path in results_dir.glob("ami_scores_*_gamma=*.json"):
        match = pattern.match(file_path.name)
        if not match:
            continue
        dataset_name, gamma_str = match.groups()
        pairs.add((dataset_name, float(gamma_str)))

    return sorted(pairs)


def load_time_curve(
    dataset_name: str, gamma: float, results_dir: Path
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Load time-sensitivity curve (t, means, stds) for a dataset-gamma pair."""
    gamma_str = f"{gamma:g}"
    score_path = results_dir / f"ami_scores_{dataset_name}_gamma={gamma_str}.json"
    std_path = results_dir / f"ami_stds_{dataset_name}_gamma={gamma_str}.json"

    with score_path.open("r", encoding="utf-8") as f:
        scores_map = json.load(f)
    with std_path.open("r", encoding="utf-8") as f:
        stds_map = json.load(f)

    t_values = sorted(int(k) for k in scores_map.keys())
    means = np.array([scores_map[str(t)] for t in t_values], dtype=float)
    stds = np.array([stds_map[str(t)] for t in t_values], dtype=float)
    return t_values, means, stds


def get_time_recommendation(
    dataset_name: str, gamma: float
) -> tuple[float | None, float | None]:
    """Compute time recommendations (parametrized_laplacian, diff). Returns (t_ours, t_diff) or (None, None) if not available."""
    try:
        from benchmarks.load import load_dataset
        from utils.get_time_iteration import get_time_iteration
        from competitors.parametrized_laplacians import parametrized_laplacian
        from vertex_measures import sum_deg

        A, y, _ = load_dataset(dataset_name)
        n = A.shape[0]

        diags = np.array(A.sum(axis=1)).flatten()
        diags[diags == 0] = 1e-12
        D_inv = np.diag(1.0 / diags)
        P = D_inv @ A

        nu = sum_deg(A, gamma=gamma)
        Lds = parametrized_laplacian(P, nu, normalized=True)
        t_ours = get_time_iteration(Lds, t_max=250)
        try:
            from utils.get_time_iteration import get_time_iteration_differences

            t_diff = get_time_iteration_differences(P, t_max=250)
        except (ImportError, AttributeError):
            t_diff = None

        return t_ours, t_diff
    except Exception as e:
        return None, None


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


def plot_time_curve(
    dataset_name: str,
    gamma: float,
    t_values: list[int],
    means: np.ndarray,
    stds: np.ndarray,
    output_dir: Path,
    show: bool = False,
) -> tuple[Path, Path]:
    """Plot and save time-sensitivity curve with reference lines.

    Returns tuple of (figure_path, legend_path).
    """
    style = get_method_style("ParPIC")

    fig, ax = plt.subplots(figsize=(10, 6))

    n = None
    t_ours = None
    t_diff = None
    try:
        from benchmarks.load import load_dataset

        A, _, _ = load_dataset(dataset_name)
        n = A.shape[0]
        t_ours, t_diff = get_time_recommendation(dataset_name, gamma)
    except Exception:
        pass

    ymax = float(np.max(means + stds))
    if t_ours is not None:
        ax.vlines(
            x=t_ours,
            ymin=0,
            ymax=ymax,
            colors="#F96C39",
            linestyles="solid",
            label="Selected time (ours)",
        )
    if t_diff is not None:
        ax.vlines(
            x=t_diff,
            ymin=0,
            ymax=ymax,
            colors="#28A745",
            linestyles="dashdot",
            label="Acceleration-based time",
        )
    if n is not None:
        ax.vlines(
            x=np.sqrt(n),
            ymin=0,
            ymax=ymax,
            colors="#ACACAC",
            linestyles="dashed",
            label=r"$\sqrt{N}$",
        )
        ax.vlines(
            x=np.log(n),
            ymin=0,
            ymax=ymax,
            colors="#ACACAC",
            linestyles="dotted",
            label=r"$\log{N}$",
        )

    # Main curve
    ax.plot(
        t_values,
        means,
        color=style["color"],
        marker=style["marker"],
        label="Mean AMI",
        linewidth=2,
    )
    ax.fill_between(
        t_values,
        np.maximum(means - stds, 0),
        means + stds,
        color=style["color"],
        alpha=0.2,
    )

    ax.set_xlabel(r"Diffusion time ($t$)", fontsize=12)
    ax.set_ylabel("AMI", fontsize=12)
    ax.set_title(f"{dataset_name} ($\\gamma={gamma}$)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"t_sensitivity_{dataset_name}_gamma={gamma:g}.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    legend_path = save_legend(ax, output_dir, "t_sensitivity_legend.pdf")

    if show:
        plt.show()
    plt.close(fig)
    return output_path, legend_path


def main() -> None:
    """Parse args and generate time-sensitivity plots."""
    parser = argparse.ArgumentParser(
        description="Plot time-sensitivity curves (AMI vs diffusion time) for datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to plot.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma value used in the vertex measure.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing '*_gamma=*.json' files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR / "t_sensitivity",
        help="Directory where PDF plots are saved.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively."
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {args.results_dir}")

    available_pairs = discover_dataset_gamma_pairs(args.results_dir)
    if (args.dataset, args.gamma) not in available_pairs:
        available_str = ", ".join(f"({d}, {g})" for d, g in available_pairs[:5])
        raise ValueError(
            f"Dataset-gamma pair ({args.dataset}, {args.gamma}) not found in {args.results_dir}. "
            f"Available: {available_str}..."
        )

    t_values, means, stds = load_time_curve(
        dataset_name=args.dataset,
        gamma=args.gamma,
        results_dir=args.results_dir,
    )
    output_path, legend_path = plot_time_curve(
        dataset_name=args.dataset,
        gamma=args.gamma,
        t_values=t_values,
        means=means,
        stds=stds,
        output_dir=args.output_dir,
        show=args.show,
    )
    print(f"Saved: {output_path}")
    if legend_path:
        print(f"Saved legend: {legend_path}")


if __name__ == "__main__":
    main()
