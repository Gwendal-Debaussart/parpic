"""
This file defines a list of methods to be used in the later experiments.
Each method is represented as a dictionary containing:
  - name
    o A string representing the name of the method, used for display and saving results.
  - function:
    o A callable that outputs the matrix to be decomposed, given the input graph.
  - input_type:
    o A string indicating the type of input the function expects (e.g., "adjacency", "transition").
  - decomposition:
    o A string indicating the type of decomposition to perform (Optional, e.g., "eigen", "natural").
  - params:
    o A dictionary of additional parameters to pass to the function. (Optional)
  - vertex_measure:
    o A dictionary specifying a vertex measure function and its parameters, if applicable.
  - power_iteration:
    o A boolean indicating whether to use power iteration for eigen decomposition.
  - projection_type:
    o A string indicating the type of projection to use (e.g., "random") if power iteration is True.
"""

from competitors.sym_matrix import sym_matrix
from competitors.simple_herm import simple_herm
from competitors.dd_sym import dd_sym
from competitors.dsc_plus import dsc_plus
from competitors.herm import herm
from competitors.parametrized_laplacians import parametrized_laplacian
from vertex_measures import sum_deg
from vertex_measures.harmonic_degree import harmonic_degree_measure
from competitors.utils import teleporting_rw

def method_list():
    methods = [
        {
            "name": "SC-Sym RW",
            "function": sym_matrix,
            "input_type": "transition",
            "decomposition": "eigen",
        },
        {
            "name": "Simple-Herm",
            "function": simple_herm,
            "input_type": "adjacency",
            "decomposition": "eigen",
            "params": {"d": 3},
        },
        {
            "name": "DD-sym",
            "function": dd_sym,
            "input_type": "adjacency",
            "decomposition": "eigen",
            "params": {},
        },
        {
          "name": "DSC-plus",
          "function": dsc_plus,
          "input_type": "adjacency",
          "decomposition": "eigen",
        },
        {
            "name": "Hermitian Adjacency",
            "function": herm,
            "input_type": "adjacency",
            "decomposition": "eigen",
            "params": {"normalized": False},
        },
        {
            "name": "Hermitian Random Walk Normalized",
            "function": herm,
            "input_type": "adjacency",
            "decomposition": "eigen",
            "params": {"normalized": True},
        },
        {
          "name": "PR-SC",
          "function": teleporting_rw,
          "input_type": "adjacency",
          "decomposition": "eigen",
        },
        #------- pic-like algorithms -------
        {
          'name': '(full) N-PIC',
          'function': lambda A: A,
          "input_type": "transition",
          "power_iteration": True,
        },
        {
          "name": "(random-proj) N-PIC",
          "function": lambda A: A,
          "input_type": "transition",
          "projection_type": "random",
          "power_iteration": True,
        },
        {
          "name": "(full) S-PIC",
          "function": sym_matrix,
          "input_type": "transition",
          "power_iteration": True,
        },
        {
          "name": "(full) PR-PIC",
          "function": teleporting_rw,
          "input_type": "adjacency",
          "power_iteration": True,
        },
        {
          "name": "(random-proj) S-PIC",
          "function": sym_matrix,
          "input_type": "adjacency",
          "projection_type": "random",
          "power_iteration": True,
        },
        {
          "name": "(random-proj) PR-PIC",
          "function": teleporting_rw,
          "input_type": "adjacency",
          "projection_type": "random",
          "power_iteration": True,
        },

        #------- parametrized random walk laplacians -------
        {
            "name": "(full) Parametrized Random Walk Laplacian gamma = 0.5",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.5}},
        },
        {
            "name": "(full) Parametrized Random Walk Laplacian gamma = 0.85",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.85}},
        },
        {
            "name": "(full) Parametrized Random Walk Laplacian gamma = 0.15",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.15}},
        },
        {
            "name": "(random-proj) Parametrized Random Walk Laplacian gamma = 0.5",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.5}},
        },
        {
            "name": "(random-proj) Parametrized Random Walk Laplacian gamma = 0.25",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.25}},
        },
        {
          "name": "(random-proj) S-PIC-ADJ",
          "projection_type": "random",
          "power_iteration": True,
          "function": sym_matrix,
          "input_type": "adjacency",
        },
        {
            "name": "SC Parametrized Random Walk Laplacian gamma = 0.5",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "params": {"normalized": True},
            "decomposition": "eigen",
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.5}},
        },
        {
            "name": "(random-proj) Parametrized Random Walk Laplacian gamma = 0.85",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.85}},
        },
        {
            "name": "(random-proj) Parametrized Random Walk Laplacian gamma = 0.15",
            "function": parametrized_laplacian,
            "input_type": "transition",
            "projection_type": "random",
            "power_iteration": True,
            "params": {"normalized": True},
            "vertex_measure": {"function": sum_deg, "params": {"gamma": 0.15}},
        },
    ]
    return methods
