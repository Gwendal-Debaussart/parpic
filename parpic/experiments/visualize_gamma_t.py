import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from _common_style import BLUE_ORANGE_CMAP

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"

datasets = ["disbm_chain", "disbm_core_periphery", "seeds", "iris", "email_eu"]

gamma_values = np.arange(0.05, 1.0, 0.05)
t_values = np.arange(1, 51)

cmap_blue_orange = BLUE_ORANGE_CMAP

def load_gamma_t_results(dataset_name, results_dir=str(TABLES_DIR / "gamma_t_sensitivity")):
    """
    Load all gamma-t sensitivity results for a given dataset.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    results_dir : str
        Directory containing the JSON files

    Returns:
    --------
    ami_matrix : ndarray
        2D array of shape (len(gamma_values), len(t_values)) with mean AMI scores
    std_matrix : ndarray
        2D array of shape (len(gamma_values), len(t_values)) with std AMI scores
    """
    ami_matrix = np.zeros((len(gamma_values), len(t_values)))
    std_matrix = np.zeros((len(gamma_values), len(t_values)))

    for i, gamma in enumerate(gamma_values):
        ami_file = f"{results_dir}/ami_scores_{dataset_name}_gamma={gamma:.2f}.json"
        std_file = f"{results_dir}/ami_stds_{dataset_name}_gamma={gamma:.2f}.json"

        if os.path.exists(ami_file):
            with open(ami_file, 'r') as f:
                ami_scores = json.load(f)
            with open(std_file, 'r') as f:
                ami_stds = json.load(f)

            for j, t in enumerate(t_values):
                t_str = str(t)
                if t_str in ami_scores:
                    ami_matrix[i, j] = ami_scores[t_str]
                    std_matrix[i, j] = ami_stds[t_str]
                else:
                    ami_matrix[i, j] = np.nan
                    std_matrix[i, j] = np.nan
        else:
            print(f"Warning: File not found: {ami_file}")
            ami_matrix[i, :] = np.nan
            std_matrix[i, :] = np.nan

    return ami_matrix, std_matrix


def plot_heatmap(ami_matrix, dataset_name, figsize=(12, 8), save_path=None, show=False):
    """
    Plot a heatmap of AMI scores vs gamma and t.

    Parameters:
    -----------
    ami_matrix : ndarray
        2D array of AMI scores
    dataset_name : str
        Name of the dataset for the title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(ami_matrix, aspect='auto', cmap=cmap_blue_orange)

    gamma_tick_indices = np.arange(0, len(gamma_values), 2)
    gamma_tick_labels = [f"{gamma_values[i]:.2f}" for i in gamma_tick_indices]
    ax.set_yticks(gamma_tick_indices)
    ax.set_yticklabels(gamma_tick_labels)

    t_tick_indices = np.arange(0, len(t_values), 5)
    t_tick_labels = [str(t_values[i]) for i in t_tick_indices]
    ax.set_xticks(t_tick_indices)
    ax.set_xticklabels(t_tick_labels)

    ax.set_xlabel('Diffusion Time (t)', fontsize=12)
    ax.set_ylabel('Vertex measure parameter ($\\gamma$)', fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AMI Score', fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    if show:
        plt.show()


def plot_all_heatmaps(datasets, results_dir=str(TABLES_DIR / "gamma_t_sensitivity"),
                      save_dir=str(FIGURES_DIR / "gamma_t_sensitivity")):
    """
    Load and plot heatmaps for all datasets.

    Parameters:
    -----------
    datasets : list
        List of dataset names
    results_dir : str
        Directory containing the JSON files
    save_dir : str
        Directory to save the heatmap figures
    """
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}...")
        print(f"{'='*60}")

        ami_matrix, std_matrix = load_gamma_t_results(dataset_name, results_dir)

        if np.all(np.isnan(ami_matrix)):
            print(f"Warning: No data found for {dataset_name}")
            continue

        valid_scores = ami_matrix[~np.isnan(ami_matrix)]
        if len(valid_scores) > 0:
            print(f"AMI Score Statistics:")
            print(f"  Min:  {np.min(valid_scores):.4f}")
            print(f"  Max:  {np.max(valid_scores):.4f}")
            print(f"  Mean: {np.mean(valid_scores):.4f}")
            print(f"  Std:  {np.std(valid_scores):.4f}")

        max_idx = np.unravel_index(np.nanargmax(ami_matrix), ami_matrix.shape)
        best_gamma = gamma_values[max_idx[0]]
        best_t = t_values[max_idx[1]]
        best_ami = ami_matrix[max_idx]
        print(f"\nBest configuration:")
        print(f"  Gamma: {best_gamma:.2f}")
        print(f"  Time:  {best_t}")
        print(f"  AMI:   {best_ami:.4f}")

        save_path = f"{save_dir}/heatmap_{dataset_name}.pdf"
        plot_heatmap(ami_matrix, dataset_name, save_path=save_path)


if __name__ == "__main__":
    print("Gamma-t Sensitivity Heatmap Visualization")
    print("="*60)

    for dataset_name in ["disbm_chain", "disbm_core_periphery", "seeds", "iris", "email_eu", "polblogs", "vertebral", "glass"]:
        results_dir = str(TABLES_DIR / "gamma_t_sensitivity")
        if not os.path.exists(results_dir):
            print(f"Error: Results directory '{results_dir}' not found!")
            print("Please run gamma_t.py first to generate the results.")
            exit(1)

        print(f"\nLoading results for {dataset_name}...")
        ami_matrix, std_matrix = load_gamma_t_results(dataset_name, results_dir)

        if np.all(np.isnan(ami_matrix)):
            print(f"Error: No data found for {dataset_name}")
            print("Please run gamma_t.py first to generate the results.")
            exit(1)

        valid_scores = ami_matrix[~np.isnan(ami_matrix)]
        print(f"\nAMI Score Statistics:")
        print(f"  Min:  {np.min(valid_scores):.4f}")
        print(f"  Max:  {np.max(valid_scores):.4f}")
        print(f"  Mean: {np.mean(valid_scores):.4f}")
        print(f"  Std:  {np.std(valid_scores):.4f}")

        max_idx = np.unravel_index(np.nanargmax(ami_matrix), ami_matrix.shape)
        best_gamma = gamma_values[max_idx[0]]
        best_t = t_values[max_idx[1]]
        best_ami = ami_matrix[max_idx]
        print(f"\nBest configuration:")
        print(f"  Gamma: {best_gamma:.2f}")
        print(f"  Time:  {best_t}")
        print(f"  AMI:   {best_ami:.4f}")

        print(f"\nGenerating heatmap...")
        save_path = str(FIGURES_DIR / "gamma_t_sensitivity" / f"heatmap_{dataset_name}.pdf")
        plot_heatmap(ami_matrix, dataset_name, save_path=save_path)

        print("\n" + "="*60)
        print("Visualization complete!")
