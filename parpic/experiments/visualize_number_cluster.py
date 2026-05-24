import json
from pathlib import Path

import matplotlib.pyplot as plt

from _common_style import get_method_style

# ========== MODEL SELECTION ==========
MODEL_TYPE = "chain"  # Change to "chain" or "cp" (core-periphery)
# ======================================

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"

cluster_means_path = TABLES_DIR / f"disbm_{MODEL_TYPE}_cluster_sensitivity_means.json"
cluster_vars_path = TABLES_DIR / f"disbm_{MODEL_TYPE}_cluster_sensitivity_vars.json"
cluster_means = json.load(open(cluster_means_path, "r"))
cluster_vars = json.load(open(cluster_vars_path, "r"))
num_clusters_list = [3, 4, 5, 6, 8, 10]


def get_series(prefixes, k_values):
    """Return y-series for the first matching key prefix list."""
    for prefix in prefixes:
        keys = [f"{prefix}_{k}" for k in k_values]
        if all(key in cluster_means for key in keys):
            return [cluster_means[key] for key in keys]
    return None

fig, ax = plt.subplots()

sym_series = get_series(["sym"], num_clusters_list)
dd_series = get_series(["dd"], num_clusters_list)
pic_series = get_series(["pic", "dsc"], num_clusters_list)
par_series = get_series(["par"], num_clusters_list)

if sym_series is not None:
    ax.plot(num_clusters_list, sym_series, label="S-PIC", **get_method_style("S-PIC"))
if dd_series is not None:
    ax.plot(num_clusters_list, dd_series, label="DD-Sym", **get_method_style("DD-Sym"))
if pic_series is not None:
    ax.plot(num_clusters_list, pic_series, label="PIC", **get_method_style("PIC"))
if par_series is not None:
    ax.plot(num_clusters_list, par_series, label="ParPIC", **get_method_style("ParPIC"))

ax.set_xlabel("Number of Clusters")
ax.set_ylabel("AMI")
ax.legend(fontsize="15")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / f"num_clusters_sensitivity_disbm_{MODEL_TYPE}.pdf", bbox_inches='tight')
plt.tight_layout()