"""Contacts family plots — plain consumers of the contacts-derived products.

Every neighborhood / density / energetics quantity is now a Build-stage product
(its own :class:`~cellflow.aggregate_quantification.quantifier.Quantifier` persists
a tidy table); see ``quantifiers/neighbor_count.py`` etc. So these plots carry **no
computation** — each is a plain
:class:`~cellflow.napari.aggregate_quantification.plots._pool_plot.PoolPlot` that
pools its product's ``object_table`` across the in-scope positions and hands the
frame to the generic ``PlotPanel``. The heavy z-score null / Boltzmann inversion /
graph walk that used to run at plot time now runs once, at Build.

The plot-time tuning the old plugins exposed (pixel size, FOV, shuffle count) moved
to Build too: pixel size flows in via ``PositionInputs``; the FOV area and shuffle
count come from the studio's shared params bar at build time.
"""
from __future__ import annotations

from cellflow.napari.aggregate_quantification.plots._pool_plot import PoolPlot
from cellflow.napari.aggregate_quantification.plots._pooling import CLASS_COLUMN

_FAMILY = "Contacts"


# ------------------------------------------------------------- potential landscape
class ContactEnergeticsPlot(PoolPlot):
    """Effective potential / barrier of T1 junction lengths."""

    plot_id = "contact_energetics"
    display_name = "Potential landscape"
    family = _FAMILY
    render_type = "potential"
    consumes = ("contact_energetics",)
    value_columns = ("signed_length",)
    # The signed-length table carries no per-cell identity; offer catalogue
    # metadata + the contact type (normalized to "unlabelled" at build time).
    group_columns = ("condition", "date", "position_id", "contact_type")
    default_plot = "potential"
    default_adaptive_bins = True
    join_class = False


# ----------------------------------------------------------- neighborhood & density
class NeighborCountPlot(PoolPlot):
    plot_id = "neighbor_count"
    display_name = "Neighbor count"
    family = _FAMILY
    consumes = ("neighbor_count",)
    value_columns = ("n_neighbors",)
    group_columns = ("condition", "date", "position_id", CLASS_COLUMN)
    default_plot = "box"


class NeighborEnrichmentPlot(PoolPlot):
    plot_id = "neighbor_enrichment"
    display_name = "Neighbor enrichment"
    family = _FAMILY
    consumes = ("neighbor_enrichment",)
    value_columns = ("enrichment",)
    group_columns = ("condition", "focal_label", "neighbor_label")
    default_plot = "box"
    join_class = False


class ContactTypeZScorePlot(PoolPlot):
    plot_id = "contact_type_zscore"
    display_name = "Contact-type z-score"
    family = _FAMILY
    render_type = "bar"
    consumes = ("contact_type_zscore",)
    value_columns = ("z_score",)
    group_columns = ("contact_type", "condition")
    default_plot = "bar"
    join_class = False


class DensityPlot(PoolPlot):
    plot_id = "cell_density"
    display_name = "Density"
    family = _FAMILY
    render_type = "bar"
    consumes = ("cell_density",)
    value_columns = ("density",)
    group_columns = ("label", "condition")
    default_plot = "bar"
    join_class = False
