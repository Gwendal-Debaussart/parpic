"""Visualize transition and Laplacian-like matrices for directed SBM settings.

This script generates a synthetic directed SBM graph, computes:
- row-stochastic transition matrix P
- parametrized Laplacian-based operator P_(nu)
- symmetric random-walk operator P_sym
and saves a side-by-side matrix plot.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from benchmarks.directed_sbm import directed_sbm
from competitors.parametrized_laplacians import parametrized_laplacian
from vertex_measures.sum_deg import sum_deg


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "figures"


def build_probability_matrix(model_type: str, rho: float) -> np.ndarray:
    """Build a 3x3 chain or core-periphery probability matrix."""
    if model_type == "cp":
        return np.array(
            [
                [0.05, rho, rho],
                [0.02, 0.05, 0.02],
                [0.02, 0.02, 0.05],
            ]
        )
    if model_type == "chain":
        return np.array(
            [
                [0.05, rho, 0.0],
                [0.01, 0.05, rho],
                [0.0, 0.01, 0.05],
            ]
        )
    raise ValueError(f"Unknown model type: {model_type}. Use 'chain' or 'cp'.")


def compute_matrices(model_type: str, rho: float, block_size: int, gamma: float):
    """Generate SBM graph and compute P, P_(nu), and P_sym matrices."""
    block_sizes = [block_size, block_size, block_size]
    probs = build_probability_matrix(model_type=model_type, rho=rho)

    A, _, _ = directed_sbm(block_sizes=block_sizes, P=probs, seed=42)
    A = np.asarray(A, dtype=float)

    diags = np.array(A.sum(axis=1)).flatten()
    diags[diags == 0] = 1e-10
    D_inv = np.diag(1 / diags)
    P = D_inv @ A

    A_sym = (A + A.T) / 2.0
    d_sym = np.array(A_sym.sum(axis=1)).flatten()
    d_sym[d_sym == 0] = 1e-10
    D_inv_sym = np.diag(1 / d_sym)
    P_sym = D_inv_sym @ A_sym

    nu = sum_deg(A, gamma=gamma, degree_normalized=True)
    P_nu = parametrized_laplacian(P, nu, normalized=True)
    return P, P_nu, P_sym


def plot_matrices(
    P: np.ndarray, P_nu: np.ndarray, P_sym: np.ndarray, output_path: Path, show: bool
) -> None:
    """Plot and save the three matrices in one figure."""
    cmap = LinearSegmentedColormap.from_list("paper_bw", ["#F1F4FF", "#000000"])

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    matrices = [P, P_nu, P_sym]

    for ax, matrix in zip(axes, matrices):
        ax.imshow(matrix, cmap=cmap, interpolation="nearest")
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    print(f"Saved matrix figure to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize rho-dependent matrix operators for directed SBM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chain",
        choices=["chain", "cp"],
        help="SBM model type",
    )
    parser.add_argument(
        "--rho", type=float, default=0.3, help="Inter-cluster coupling parameter"
    )
    parser.add_argument(
        "--block_size", type=int, default=130, help="Size of each of the 3 blocks"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Gamma parameter for sum_deg"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display figure interactively"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional output PDF path"
    )
    args = parser.parse_args()

    dataset_name = f"disbm_{args.model}"
    output_path = (
        Path(args.output)
        if args.output is not None
        else FIGURES_DIR / f"matrices_{dataset_name}_{args.rho:.2f}.pdf"
    )

    P, P_nu, P_sym = compute_matrices(
        model_type=args.model,
        rho=args.rho,
        block_size=args.block_size,
        gamma=args.gamma,
    )
    plot_matrices(P=P, P_nu=P_nu, P_sym=P_sym, output_path=output_path, show=args.show)


if __name__ == "__main__":
    main()
