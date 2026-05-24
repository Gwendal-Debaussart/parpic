import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path

from benchmarks.load import dataset_list
from utils.method_list import method_list

BASE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = BASE_DIR / "tables"

r"""
The file contains functions to convert benchmark results into LaTeX tables. It relies on the following definitions inside the .tex file:
\newcommand{\mybest}[1]{\textbf{#1}}
\newcommand{\mysecond}[1]{\underline{#1}}
\newcommand{\mystd}[1]{\,(#1)}
"""


def alternative_method_name(method_name):
    """
    Generate an alternative name for a method by replacing underscores with spaces and capitalizing words.

    :param method_name: str
    :return: str
    """
    methods_alt = {
        "(random-proj) PR-PIC": r"\pagerankpic",
        "(random-proj) Parametrized Random Walk Laplacian gamma = 0.5": r"\parampic",
        "(random-proj) S-PIC-ADJ": r"\sympic",
        "(random-proj) N-PIC": r"\pic",
        "DD-sym": r"\ddsym",
        "DSC-plus": r"\dipla",
        "SC Parametrized Random Walk Laplacian gamma = 0.5": r"\genlap",
        "Hermitian Adjacency": r"\hermit",
        "Hermitian Random Walk Normalized": r"\hermrw",
        "SC-Sym RW": r"\scsymrw",
        "Simple-Herm": r"\simpleherm",
    }
    if method_name in methods_alt:
        return methods_alt[method_name]
    else:
        return method_name.title().replace("_", " ")


def alternative_dataset_name(dataset_name):
    """
    Generate an alternative name for a dataset by replacing underscores with spaces and capitalizing words.

    :param dataset_name: str
    :return: str
    """
    dataset_alt = {
        "Dataset name in the benchmark": "Dataset name as displayed in LaTeX tables"
    }
    if dataset_name in dataset_alt:
        return dataset_alt[dataset_name]
    else:
        return dataset_name.title().replace("_", " ")


def format_result(mean, std, bold=False, second=False, percentage=True):
    """
    Format mean ± std with macros.

    :param mean: float
    :param std: float
    :param bold: bool, whether to bold the result
    :param second: bool, whether to underline the result
    :param percentage: bool, whether to format the result as a percentage
    :return: str, formatted result

    Remarks:
    - If mean is NaN, returns "—"
    - If percentage is True, mean and std are multiplied by 100
    - If std is provided and not NaN, it is included in the output using \mystd{}
    - If bold is True, the result is wrapped in \mybest{}
    - If second is True, the result is wrapped in \mysecond{}

    Can be adapted to personal preferences for formatting if wanting to add methods parameters etc.
    """
    if pd.isna(mean):
        return "—"
    if percentage:
        mean *= 100
        if std is not None and not pd.isna(std):
            std *= 100
    base = f"{mean:.2f}"
    if std is not None and not pd.isna(std):
        base += f" \\mystd{{{std:.2f}}}"
    if bold:
        return f"\\mybest{{{base}}}"
    elif second:
        return f"\\mysecond{{{base}}}"
    return base


def benchark_to_tex(
    dataset_list, method_list, metric, percentage, results_dir, output_dir
):
    """
    Convert benchmark results to LaTeX tables.

    :param dataset_list: list of dataset names
    :param method_list: list of method names
    :param metric: str, the metric to be displayed in the table
    :param results_dir: str, directory where raw results are stored
    :param tex_dir: str, directory to save LaTeX tables
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    tab_cols = "c" * len(dataset_list)
    header = r"""\begin{table*}
        \centering
        \begin{adjustbox}{width=\linewidth,center}
        \begin{tabular}{l|c|%s}
        \Xhline{1pt}
        \textbf{Methods} & \textbf{PRB} & %s \\
        \Xhline{0.75pt}
        """ % (
        tab_cols,
        " & ".join(alternative_dataset_name(d["name"]) for d in dataset_list),
    )

    footer = r"""\Xhline{1pt}
        \end{tabular}
          \end{adjustbox}
          \caption{\textbf{%s results across multiple datasets.} The best mean per dataset is in bold (\\mybest{xxx}), the second-best is underlined (\\mysecond{xxx}). Standard deviations appear in parentheses via \\mystd{xxx}.}
          \label{tab:results_combined}
        \end{table*}
        """ % metric.title()

    all_dfs = []
    for d in dataset_list:
        df = pd.read_csv(os.path.join(results_dir, f"{d['name']}__formatted.csv"))
        df = df[df["metric"] == metric]
        all_dfs.append(
            df.rename(
                columns={"mean_val": f"{d['name']}_mean", "std_val": f"{d['name']}_std"}
            )
        )
    cols_to_drop = ["metric", "repeat_tol"]

    combined = all_dfs[0].drop(columns=cols_to_drop)
    for df in all_dfs[1:]:
        combined = pd.merge(
            combined,
            df.drop(columns=cols_to_drop),
            on="method",
            how="outer",
        )
    method_name_list = [m["name"] for m in method_list]
    combined = combined[combined["method"].isin(method_name_list)].copy()

    prb_cols = []
    for d in dataset_list:
        mean_col = f"{d['name']}_mean"
        prb_col = f"{d['name']}_prb"
        prb_cols.append(prb_col)
        if mean_col in combined.columns:
            best = combined[mean_col].max(skipna=True)
            if pd.isna(best) or best == 0:
                combined[prb_col] = np.nan
            else:
                combined[prb_col] = combined[mean_col] / best
        else:
            combined[prb_col] = np.nan

    combined["PRB_mean"] = combined[prb_cols].mean(axis=1, skipna=True)
    combined["PRB_std"] = combined[prb_cols].std(axis=1, skipna=True)

    for d in dataset_list:
        means = combined[f"{d['name']}_mean"]
        if means.notna().any():
            sorted_idx = means.sort_values(ascending=False).index
            if len(sorted_idx) > 0:
                combined.loc[sorted_idx[0], f"{d['name']}_style"] = "best"
            if len(sorted_idx) > 1:
                combined.loc[sorted_idx[1], f"{d['name']}_style"] = "second"

    rows = []
    for _, row in combined.iterrows():
        method_label = alternative_method_name(row["method"])
        row_entries = [method_label]

        prb_mean = row.get("PRB_mean", float("nan"))
        prb_std = row.get("PRB_std", float("nan"))
        prb_formatted = format_result(
            prb_mean, prb_std, bold=False, second=False, percentage=False
        )
        row_entries.append(prb_formatted)
        for d in dataset_list:
            mean = row.get(f"{d['name']}_mean", float("nan"))
            std = row.get(f"{d['name']}_std", float("nan"))
            style = row.get(f"{d['name']}_style", "")
            bold = style == "best"
            second = style == "second"
            formatted = format_result(
                mean, std, bold=bold, second=second, percentage=percentage
            )
            row_entries.append(formatted)
        rows.append(" & ".join(row_entries) + r" \\")

    body = "\n".join(rows)
    latex_code = header + body + footer
    with open(output_dir + f"benchmark_{metric}.tex", "w") as f:
        f.write(latex_code)

    print(f"LaTeX table saved to {output_dir + f'benchmark_{metric}.tex'}")


def main() -> None:
    """Parse args and generate LaTeX benchmark tables."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from benchmark clustering results."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to include. If omitted, includes default set.",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        default=[
            "cornell",
            "texas",
            "wisconsin",
            "citeseer",
            "pubmed",
            "mnist",
            "directed_sbm",
            "directed_sbm_chain",
            "karate_club",
        ],
        help="Dataset names to exclude.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method names to include. If omitted, includes default set.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="Adjusted Mutual Information",
        help="Metric to display in table.",
    )
    parser.add_argument(
        "--percentage",
        action="store_true",
        help="Whether to format results as percentages.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=TABLES_DIR / "clustering_benchmark",
        help="Directory containing formatted benchmark results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR / "benchmark_tex",
        help="Directory where LaTeX tables are saved.",
    )
    args = parser.parse_args()
    all_datasets = dataset_list()
    filtered_datasets = [
        d for d in all_datasets if d["name"] not in args.exclude_datasets
    ]
    if args.datasets:
        filtered_datasets = [d for d in filtered_datasets if d["name"] in args.datasets]
    else:
        default_datasets = [
            "polblogs",
            "disbm_baseline",
            "directed_sbm_core_periphery",
            "email_eu",
            "chain_sbm",
            "cora",
        ]
        filtered_datasets = [
            d for d in filtered_datasets if d["name"] in default_datasets
        ]

    all_methods = method_list()

    if args.methods:
        filtered_methods = [m for m in all_methods if m["name"] in args.methods]
    else:
        default_methods = [
            "(random-proj) PR-PIC",
            "(random-proj) Parametrized Random Walk Laplacian gamma = 0.5",
            "(random-proj) S-PIC-ADJ",
            "DD-sym",
            "DSC-plus",
            "(random-proj) N-PIC",
            "SC Parametrized Random Walk Laplacian gamma = 0.5",
            "Hermitian Adjacency",
            "Hermitian Random Walk Normalized",
            "SC-Sym RW",
            "Simple-Herm",
        ]
        filtered_methods = [m for m in all_methods if m["name"] in default_methods]

    print(f"Datasets: {[d['name'] for d in filtered_datasets]}")
    print(f"Methods: {[m['name'] for m in filtered_methods]}")
    print(f"Metric: {args.metric}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print()

    benchark_to_tex(
        dataset_list=filtered_datasets,
        method_list=filtered_methods,
        metric=args.metric,
        percentage=args.percentage,
        results_dir=str(args.results_dir),
        output_dir=str(args.output_dir),
    )


if __name__ == "__main__":
    main()
