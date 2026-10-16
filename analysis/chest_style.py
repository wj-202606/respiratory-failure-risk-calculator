# -*- coding: utf-8 -*-
"""
CHEST-journal figure style for matplotlib.
Usage:
    from chст_style import apply_style
    apply_style()
"""
import matplotlib as mpl

def apply_style():
    mpl.rcParams.update({
        # Font: Arial (CHEST requirement)
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "savefig.dpi": 1200,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "pdf.fonttype": 42,   # embed TrueType, editable
        "ps.fonttype": 42,
        "figure.dpi": 120,
    })