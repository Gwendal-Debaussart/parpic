"""Plot gamma-parameter sensitivity curves for one or more datasets."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common_style import get_method_style

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = TABLES_DIR / "gamma_sensitivity"


def discover_datasets(results_dir: Path) -> list[str]:
    """Return dataset names detected in the gamma-sweep results directory."""
    dataset_names = []
    for file_path in results_dir.glob("*_gamma_sweep.json"):
        dataset_names.append(file_path.name.replace("_gamma_sweep.json", ""))
    return sorted(set(dataset_names))


def load_gamma_results(dataset_name: str, results_dir: Path) -> dict:
    """Load one dataset's gamma sweep JSON result."""
    file_path = results_dir / f"{dataset_name}_gamma_sweep.json"
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_curve(result: dict) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Parse and sort gamma curve arrays from one gamma sweep result dict."""
    ami_scores = {float(k): v for k, v in result["ami_scores"].items()}
    ami_stds = {float(k): v for k, v in result["ami_stds"].items()}

    gamma_sorted = sorted(ami_scores.keys())
    ami_values = np.array([ami_scores[g] for g in gamma_sorted], dtype=float)
    ami_std_values = np.array([ami_stds[g] for g in gamma_sorted], dtype=float)
    return gamma_sorted, ami_values, ami_std_values


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


def plot_gamma_curve(result: dict, dataset_name: str, output_dir: Path, show: bool = False, save_legend_only: bool = False) -> tuple[Path, Path]:
    """Plot and save AMI vs gamma with uncertainty band for one dataset.

    Returns tuple of (figure_path, legend_path).
    """
    gamma_sorted, ami_values, ami_std_values = parse_curve(result)
    t_value = result.get("t", "unknown")
    style = get_method_style("ParPIC")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        gamma_sorted,
        ami_values,
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
        label="Mean AMI",
    )
    ax.fill_between(
        gamma_sorted,
        ami_values - ami_std_values,
        ami_values + ami_std_values,
        color=style["color"],
        alpha=0.2,
    )

    best_idx = int(np.argmax(ami_values))
    best_gamma = gamma_sorted[best_idx]
    best_ami = ami_values[best_idx]
    ax.scatter([best_gamma], [best_ami], color="#F96C39", zorder=3, label=f"Best $\\gamma$={best_gamma:.2f}")

    ax.set_title(dataset_name)
    ax.set_xlabel(r"$\\gamma$")
    ax.set_ylabel("AMI")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gamma_sensitivity_{dataset_name}_t{t_value}.pdf"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    legend_path = save_legend(ax, output_dir, "gamma_sensitivity_legend.pdf")

    if show:
        plt.show()
    plt.close(fig)
    return output_path, legend_path


def main() -> None:
    """Parse args and generate gamma-sensitivity plots for selected datasets."""
    parser = argparse.ArgumentParser(
        description="Generate gamma sensitivity plots from JSON benchmark results."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to plot. If omitted, all available datasets are used.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing '*_gamma_sweep.json' files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR / "gamma_sensitivity",
        help="Directory where PDF plots are saved.",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {args.results_dir}")

    datasets = args.datasets if args.datasets else discover_datasets(args.results_dir)
    if not datasets:
        raise ValueError(f"No gamma sweep files found in {args.results_dir}")

    legend_path = None
    for i, dataset_name in enumerate(datasets):
        result = load_gamma_results(dataset_name=dataset_name, results_dir=args.results_dir)
        output_path, legend_path = plot_gamma_curve(
            result=result,
            dataset_name=dataset_name,
            output_dir=args.output_dir,
            show=args.show,
        )
        print(f"Saved: {output_path}")
        if i == 0 and legend_path:
            print(f"Saved legend: {legend_path}")


if __name__ == "__main__":
    main()
