"""Common plotting style settings for visualization scripts."""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
    }
)


METHOD_STYLES = {
    "S-PIC": {"color": "#F96C39", "linestyle": "--", "marker": "^"},
    "PIC": {"color": "#28A745", "linestyle": ":", "marker": "s"},
    "DD-Sym": {"color": "#6F42C1", "linestyle": "-.", "marker": "x"},
    "ParPIC": {"color": "#072AC8", "linestyle": "-", "marker": "o"},
}


def get_method_style(method_label: str) -> dict:
    """Return plotting kwargs for a method label."""
    return METHOD_STYLES.get(
        method_label,
        {"color": "#333333", "linestyle": "-", "marker": "o"},
    )


# BLUE_ORANGE_COLORS = ["#072AC8", "#9A44C5", "#ff459c", "#F96C39"]
BLUE_ORANGE_COLORS = ["#072AC8", "#9A44C5", "#F96C39"]
BLUE_ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "blue_orange", BLUE_ORANGE_COLORS, N=256
)
