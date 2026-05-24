"""
Visualization script for DiSBM core-periphery cluster sensitivity analysis
Creates heatmaps showing AMI scores for different rho and first cluster size combinations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from _common_style import BLUE_ORANGE_CMAP

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"

cmap_blue_orange = BLUE_ORANGE_CMAP


def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def create_heatmap_data(results_dict, method='param'):
    """Create matrix for heatmap from results dictionary.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing means and stds for different configurations.
    method : str, default='param'
        Which method's results to use ('param' or 'sym').
    """
    if method == 'param':
        means = results_dict.get('means_param', results_dict.get('means', {}))
        stds = results_dict.get('stds_param', results_dict.get('stds', {}))
    else:  # method == 'sym'
        means = results_dict.get('means_sym', {})
        stds = results_dict.get('stds_sym', {})

    params = results_dict['parameters']
    cluster_sizes = params['first_cluster_sizes']
    rho_values = params['rho_values']
    rho_values = [r for r in rho_values if r != 0]
    mean_matrix = np.zeros((len(rho_values), len(cluster_sizes)))
    std_matrix = np.zeros((len(rho_values), len(cluster_sizes)))

    for i, rho in enumerate(rho_values):
        for j, size in enumerate(cluster_sizes):
            key = f"size_{size}_rho_{rho:.3f}"
            mean_matrix[i, j] = means.get(key, np.nan)
            std_matrix[i, j] = stds.get(key, np.nan)

    return mean_matrix, std_matrix, cluster_sizes, rho_values, params


def plot_heatmap(results_dict, output_dir=str(FIGURES_DIR / 'cp_cluster_sensitivity')):
    """Create and save heatmap visualizations for both parametrized and symmetric methods."""
    os.makedirs(output_dir, exist_ok=True)

    has_both = 'means_param' in results_dict and 'means_sym' in results_dict
    methods = [('param', 'Parametrized'), ('sym', 'Symmetric (A+A.T)')] if has_both else [('param', '')]

    for method_key, method_name in methods:
        if method_key == 'sym' and 'means_sym' not in results_dict:
            continue

        mean_matrix, _, cluster_sizes, rho_values, _ = create_heatmap_data(results_dict, method=method_key)
        suffix = f'_{method_key}' if has_both else ''
        title_suffix = f' ({method_name})' if has_both else ''

        fig, ax = plt.subplots(figsize=(14, 10))
        heatmap = sns.heatmap(mean_matrix[::-1],
                    xticklabels=[str(s) for s in cluster_sizes],
                    yticklabels=[f'{r:.2f}' for r in reversed(rho_values)],
                    cmap=cmap_blue_orange,
                    vmin=0,
                    vmax=1,
                    cbar=(method_key == 'sym'),
                    cbar_kws={'label': 'Mean AMI Score'} if method_key == 'sym' else {},
                    ax=ax)

        if method_key == 'param':
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cax.set_visible(False)

        if method_key == 'sym' and heatmap.collections:
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label('Mean AMI Score', fontsize=20)

        ax.set_xlabel('First Cluster Size', fontsize=20)
        ax.set_ylabel(r'$\rho$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cp_sensitivity_heatmap_mean{suffix}.pdf'), bbox_inches='tight')
        print(f"Saved mean heatmap to {output_dir}/cp_sensitivity_heatmap_mean{suffix}.pdf")
        plt.close()

    print("\nAll visualizations saved successfully!")


def print_statistics(results_dict):
    """Print summary statistics for both methods."""
    params = results_dict['parameters']
    has_both = 'means_param' in results_dict and 'means_sym' in results_dict

    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    print(f"Parameters: gamma={params['gamma']}, t={params['t']}, n_runs={params['n_runs']}")
    print(f"Cluster sizes tested: {len(params['first_cluster_sizes'])}")
    print(f"Rho values tested: {len(params['rho_values'])}")

    if has_both:
        means_param = results_dict['means_param']
        stds_param = results_dict['stds_param']
        means_sym = results_dict['means_sym']
        stds_sym = results_dict['stds_sym']

        print(f"Total combinations: {len(means_param)}")

        all_means_param = list(means_param.values())
        all_stds_param = list(stds_param.values())
        all_means_sym = list(means_sym.values())
        all_stds_sym = list(stds_sym.values())

        print(f"\n--- Parametrized Laplacian ---")
        print(f"  Mean AMI: {np.mean(all_means_param):.4f} ± {np.std(all_means_param):.4f}")
        print(f"  Min AMI: {np.min(all_means_param):.4f}")
        print(f"  Max AMI: {np.max(all_means_param):.4f}")
        print(f"  Mean Std: {np.mean(all_stds_param):.4f}")

        best_key_param = max(means_param, key=means_param.get)
        worst_key_param = min(means_param, key=means_param.get)

        print(f"\n  Best configuration: {best_key_param}")
        print(f"    AMI: {means_param[best_key_param]:.4f} ± {stds_param[best_key_param]:.4f}")
        print(f"  Worst configuration: {worst_key_param}")
        print(f"    AMI: {means_param[worst_key_param]:.4f} ± {stds_param[worst_key_param]:.4f}")

        print(f"\n--- Symmetric Laplacian (A + A.T) ---")
        print(f"  Mean AMI: {np.mean(all_means_sym):.4f} ± {np.std(all_means_sym):.4f}")
        print(f"  Min AMI: {np.min(all_means_sym):.4f}")
        print(f"  Max AMI: {np.max(all_means_sym):.4f}")
        print(f"  Mean Std: {np.mean(all_stds_sym):.4f}")

        best_key_sym = max(means_sym, key=means_sym.get)
        worst_key_sym = min(means_sym, key=means_sym.get)

        print(f"\n  Best configuration: {best_key_sym}")
        print(f"    AMI: {means_sym[best_key_sym]:.4f} ± {stds_sym[best_key_sym]:.4f}")
        print(f"  Worst configuration: {worst_key_sym}")
        print(f"    AMI: {means_sym[worst_key_sym]:.4f} ± {stds_sym[worst_key_sym]:.4f}")

        print(f"\n--- Comparison ---")
        differences = [means_param[k] - means_sym[k] for k in means_param.keys()]
        print(f"  Mean difference (Param - Sym): {np.mean(differences):.4f}")
        print(f"  Param wins: {sum(d > 0 for d in differences)} / {len(differences)}")
        print(f"  Sym wins: {sum(d < 0 for d in differences)} / {len(differences)}")

    else:
        means = results_dict.get('means', results_dict.get('means_param', {}))
        stds = results_dict.get('stds', results_dict.get('stds_param', {}))

        print(f"Total combinations: {len(means)}")

        all_means = list(means.values())
        all_stds = list(stds.values())

        print(f"\nOverall statistics:")
        print(f"  Mean AMI: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
        print(f"  Min AMI: {np.min(all_means):.4f}")
        print(f"  Max AMI: {np.max(all_means):.4f}")
        print(f"  Mean Std: {np.mean(all_stds):.4f}")

        best_key = max(means, key=means.get)
        worst_key = min(means, key=means.get)

        print(f"\nBest configuration: {best_key}")
        print(f"  AMI: {means[best_key]:.4f} ± {stds[best_key]:.4f}")
        print(f"\nWorst configuration: {worst_key}")
        print(f"  AMI: {means[worst_key]:.4f} ± {stds[worst_key]:.4f}")

    print("="*80)


def main(args):
    """Main visualization function."""
    print("Loading results...")
    results = load_results(args.results_file)

    print("Printing statistics...")
    print_statistics(results)

    print("\nCreating visualizations...")
    plot_heatmap(results, output_dir=args.output_dir)

    print("\nVisualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize DiSBM CP sensitivity results')
    parser.add_argument('--results_file', type=str,
                        default=str(TABLES_DIR / 'disbm_cp_cluster_sensitivity' / 'results.json'),
                        help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str,
                        default=str(FIGURES_DIR / 'cp_cluster_sensitivity'),
                        help='Output directory for visualizations')

    args = parser.parse_args()
    main(args)
