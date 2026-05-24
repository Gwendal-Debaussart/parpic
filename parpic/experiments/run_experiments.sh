#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_step() {
	local script_name="$1"
	echo ""
	echo "============================================================"
	echo "Running: ${script_name}"
	echo "============================================================"
	"${PYTHON_BIN}" "${SCRIPT_DIR}/${script_name}"
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
	echo "Error: Python executable '${PYTHON_BIN}' not found in PATH."
	echo "Tip: set PYTHON_BIN=/path/to/python when launching this script."
	exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Experiments directory: ${SCRIPT_DIR}"

# 1) Generate experiment outputs
run_step "experiments.py"
run_step "emb_dim_experiment.py"
run_step "parameter_sensitivity.py"
run_step "t_experiments.py"
run_step "cp_cluster_sensitivity.py"
run_step "test_rho.py"
run_step "num_clust_sensitivity.py"

# 2) Build visualizations from generated outputs
run_step "visualize_gamma_t.py"
run_step "visualize_cp_cluster_sensitivity.py"
run_step "visualize_embedding.py"
run_step "visualize_number_cluster.py"
run_step "visualize_elbows_n_probes.py"
run_step "runtimes.py"
run_step "visualize_rho_matrices.py"

echo ""
echo "All experiments and visualizations completed successfully."
