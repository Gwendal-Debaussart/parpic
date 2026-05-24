"""
This file contains functions to load various datasets for benchmarking clustering algorithms. It includes both real-world datasets and synthetic datasets generated using the Stochastic Block Model (SBM). The `dataset_list` function provides a list of dataset configurations, while the `dataset_functions` function returns a mapping of dataset names to their corresponding loading functions. The `load_dataset` function allows users to load a dataset by specifying its name and any necessary parameters.

Datasets are expected to be returned in the form of a tuple:
  - Adjacency matrix
  - Labels (ground truth for clustering)
  - Features (optional, may be None for graph-based datasets)
"""

from .iris import iris
from .wine import wine
from .glass import glass
from .wdbc import wdbc
from .segmentation import segmentation
from .seeds import seeds
from .vertebral import vertebral
from .yeast import yeast
from .control_chart import control_chart
from .polblogs import polblogs
from .olivetti import olivetti
from .directed_sbm import directed_sbm
from .mnist import mnist
from .email import email_eu
import numpy as np


def dataset_list():
    """
    Outputs a list of dataset configurations for benchmarking.

    Synthetic datasets are defined with their generation parameters, while real datasets are listed with their loading function names. Each dataset configuration is a dictionary containing at least the keys "name" and "function", and optionally "parameters" for synthetic datasets.
    """
    datasets = [
        {
            "name": "chain_sbm",
            "function": "directed_sbm",
            "parameters": {
                "block_sizes": [500, 500, 500],
                "P": np.array(
                    [
                        [0.05, 0.6, 0.0],
                        [0.01, 0.05, 0.6],
                        [0, 0.01, 0.05],
                    ]
                ),
            },
        },
        {
            "name": "disbm_baseline",
            "function": "directed_sbm",
            "parameters": {
                "block_sizes": [400, 400, 400],
                "P": np.array(
                    [
                        [0.05, 0.01, 0.01],
                        [0.01, 0.05, 0.01],
                        [0.01, 0.01, 0.05],
                    ]
                ),
            },
        },
        {
            "name": "directed_sbm_core_periphery",
            "function": "directed_sbm",
            "parameters": {
                "block_sizes": [1300, 1300, 1300],
                "P": [[0.05, 0.6, 0.6], [0.02, 0.05, 0.02], [0.02, 0.02, 0.05]],
            },
        },
    ]
    for name in dataset_functions().keys():
        if name != "directed_sbm":
            datasets.extend([{"name": name, "function": name, "args": {}}])
    return datasets


def dataset_functions():
    """
    Returns a dictionary of available datasets and their loading functions.

    Returns:
    --------
    datasets : dict
        A dictionary where keys are dataset names and values are functions to load them.
    """
    return {
        "iris": iris,
        "wine": wine,
        "glass": glass,
        "wdbc": wdbc,
        "control_chart": control_chart,
        "segmentation": segmentation,
        "seeds": seeds,
        "olivetti": olivetti,
        "vertebral": vertebral,
        "yeast": yeast,
        "mnist": mnist,
        # ------ graph-based datasets ------
        "polblogs": polblogs,
        "email_eu": email_eu,
        # ------- synthetic datasets -------
        "directed_sbm": directed_sbm,
    }


def load_dataset(function_name: str, **args):
    """
    Load a dataset by name.

    Parameters:
    -----------
    name : str
        Name of the dataset to load.
    **args : dict
        Additional arguments to pass to the dataset loading function.

    Returns:
    --------
    data : tuple
        The loaded dataset, typically as (Adjacency matrix, labels, [features]). For graph datasets, features may be omitted.
    """
    datasets = dataset_functions()
    if function_name in datasets:
        return datasets[function_name](**args)
    else:
        raise ValueError(f"Function '{function_name}' is not recognized.")
