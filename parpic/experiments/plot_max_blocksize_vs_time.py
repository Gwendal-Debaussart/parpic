"""
Plot the maximum block size each method can handle within a 1-minute time budget.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "max_blocksize_vs_time_results.json"


with open(RESULTS_PATH, "r") as f:
    all_results = json.load(f)

plt.figure(figsize=(8, 5))
colors = {"ParPIC-random": "#072AC8", "ParPIC-full": "#2B9348", "Classical": "#F96C39"}
for method, results in all_results.items():
    Ns = [r["N"] for r in results if r["time"] is not None]
    times = [r["time"] for r in results if r["time"] is not None]
    plt.plot(Ns, times, marker="o", label=method, color=colors.get(method, None))

plt.xlabel("Number of nodes (N)")
plt.ylabel("Runtime (seconds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "max_blocksize_vs_time_plot.pdf", bbox_inches="tight")
plt.show()
