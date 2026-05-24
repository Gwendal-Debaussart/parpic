import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from competitors.parametrized_laplacians import parametrized_laplacian
from utils.get_power_embedding import get_power_embedding
from vertex_measures.sum_deg import sum_deg
from utils.get_time_iteration import get_time_iteration
from benchmarks.load import dataset_list, load_dataset
from benchmarks.directed_sbm import directed_sbm
from joblib import Parallel, delayed

def run_gamma_trial(gamma_val, A, y, P, t, run_id, seed=42):
        """Run a single trial for a specific gamma value."""
        np.random.seed(seed + run_id)
        nu = sum_deg(A, gamma = gamma_val)
        Lds = parametrized_laplacian(P, nu, normalized = True)
        L_emb = get_power_embedding(Lds, t = t, projection_type="random")
        k_means = KMeans(n_clusters = len(np.unique(y)), n_init = 100, random_state=seed + run_id).fit(L_emb)
        y_pred = k_means.labels_
        return ami(y, y_pred)

# for dataset_name in ["email_eu", "polblogs", "seeds", "iris"]:
# for dataset_name in ["control_chart", "vertebral", "polblogs"]:
for dataset_name in ["disbm_chain"]:
    print(f"Dataset: {dataset_name}")
    if dataset_name in ["disbm_chain", "disbm_cp"]:
        if dataset_name == "disbm_chain":
            block_sizes = [500, 500, 500]
            probs = [
                [0.05, 0.6, 0.0],
                [0.01, 0.05, 0.6],
                [0.0, 0.01, 0.05],
            ]
        else:
            block_sizes = [1300, 1300, 1300]
            probs = np.array([
                [0.05, 0.6, 0.6],
                [0.02, 0.05, 0.02],
                [0.02, 0.02, 0.05],
            ])
        A, y, _ = directed_sbm(block_sizes=block_sizes, P=probs)
        A = A.astype(float)
    else:
        A, y , _ = load_dataset(dataset_name)


# block_sizes = [500, 500, 500]
# probs = [
#         [0.05, 0.6, 0.0],
#         [0.01, 0.05, 0.6],
#         [0.0, 0.01, 0.05],
#     ]
# dataset_name = "chain_sbm"
# A, y, _ = directed_sbm(block_sizes=block_sizes, P=probs)


    diags = np.array(A.sum(axis=1)).flatten()
    diags[diags == 0] = 1e-12
    D_inv = np.diag(1 / diags)
    P = D_inv @ A

    gamma_range = np.arange(0.0, 1.05, 0.05)
    n_runs = 50
    t = int(get_time_iteration(parametrized_laplacian(P, sum_deg(A, gamma=0.5), normalized=True), t_max=250))
    print(f"Selected time: t = {t}")
    if dataset_name == "polblogs":
        t=4
    if dataset_name == "disbm_chain":
        t=8

    print(f"Running {len(gamma_range)} gamma values with {n_runs} trials each in parallel...")
    start_time = time.time()

    all_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_gamma_trial)(gamma_val, A, y, P, t, run_id, 42)
        for gamma_val in gamma_range
        for run_id in range(n_runs)
    )

    elapsed_time = time.time() - start_time
    print(f"\nAll runs completed in {elapsed_time:.1f} seconds!")

    ami_scores = {}
    ami_stds = {}
    idx = 0
    for gamma_val in gamma_range:
        gamma_results = all_results[idx:idx + n_runs]
        ami_scores[gamma_val] = np.mean(gamma_results)
        ami_stds[gamma_val] = np.std(gamma_results)
        print(f"gamma: {gamma_val:.2f}, AMI: {ami_scores[gamma_val]:.4f} ± {ami_stds[gamma_val]:.4f}")
        idx += n_runs

    results_dict = {
        'dataset_name': dataset_name,
        't': t,
        'n_runs': n_runs,
        'gamma_values': gamma_range.tolist(),
        'ami_scores': {f"{k:.2f}": float(v) for k, v in ami_scores.items()},
        'ami_stds': {f"{k:.2f}": float(v) for k, v in ami_stds.items()}
    }

    import os
    os.makedirs('tables/gamma_sensitivity', exist_ok=True)

    json_filename = f"tables/gamma_sensitivity/{dataset_name}_gamma_sweep.json"
    with open(json_filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {json_filename}")

    json_filename = f"tables/gamma_sensitivity/{dataset_name}_gamma_sweep.json"
    with open(json_filename, 'r') as f:
        results_dict = json.load(f)

    gamma_values = results_dict['gamma_values']
    ami_scores_json = {float(k): v for k, v in results_dict['ami_scores'].items()}
    ami_stds_json = {float(k): v for k, v in results_dict['ami_stds'].items()}

    gamma_sorted = sorted(ami_scores_json.keys())
    ami_values = [ami_scores_json[g] for g in gamma_sorted]
    ami_std_values = [ami_stds_json[g] for g in gamma_sorted]

    fig, ax = plt.subplots()
    ax.plot(
        gamma_sorted,
        ami_values,
        marker='o',
        color = "#072AC8",
    )
    ax.fill_between(
        gamma_sorted,
        np.maximum(np.array(ami_values) - np.array(ami_std_values), 0),
        np.array(ami_values) + np.array(ami_std_values),
        color='#072AC8', alpha=0.2,
    )
    ax.set_xlabel("$\\gamma$", fontsize=15)
    ax.set_ylabel("AMI", fontsize=15)

    plt.tight_layout()
    plt.savefig(f"figures/gamma_sensitivity_{dataset_name}_t{t}.pdf")
    plt.show()

    print(f"Plot generated from {json_filename}")