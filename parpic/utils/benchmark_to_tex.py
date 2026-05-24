import os
import pandas as pd

r"""
The file contains functions to convert benchmark results into LaTeX tables. It relies on the following definitions inside the .tex file:
\newcommand{\mybest}[1]{\textbf{#1}}
\newcommand{\mysecond}[1]{\underline{#1}}
\newcommand{\mystd}[1]{\,(#1)}

If the formatting macros are different, the format_result function can be adapted to personal preferences for formatting if wanting to add methods parameters etc.
"""


def alternative_method_name(method_name):
    """
    Generate an alternative name for a method by replacing underscores with spaces and capitalizing words.

    :param method_name: str
    :return: str
    """
    methods_alt = {
        "Method name in the benchmark": "Method name as displayed in LaTeX tables"
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

    header = r"""
    \begin{table*}
    \centering
    \begin{adjustbox}{width=\linewidth,center}
    \begin{tabular}{l|%s|r}
    \Xhline{1pt}
    \multirow{2}{*}{\textbf{Methods}} & \multicolumn{%d}{c|}{\textbf{Datasets}} \\
    \cline{2-%d}
    &
    """ % (
        "c" * len(dataset_list),
        len(dataset_list),
        len(dataset_list),
    )

    dataset_header = (
        " & ".join(alternative_dataset_name(d["name"]) for d in dataset_list) + r" \\"
    )
    header += dataset_header + "\n\\Xhline{0.75pt}\n"

    footer = r"""
    \Xhline{1pt}
    \end{tabular}
      \end{adjustbox}
      \caption{\textbf{%s results across multiple datasets.} The best mean per dataset is in bold (\mybest{xxx}), the second-best is underlined (\mysecond{xxx}). Standard deviations appear in parentheses via \mystd{xxx}.}
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
    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = pd.merge(combined, df, on="method", how="outer")

    combined = combined[combined["method"].isin(method_list)].copy()
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
