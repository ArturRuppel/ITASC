"""Shape family plots — one per shape product.

The old Shape *plugin* carried a cell / nucleus / both scope dropdown that
selected which of three quantifiers to plot. Under the producer/consumer model
that dropdown dissolves: each scope is simply a separate
:class:`~cellflow.napari.aggregate_quantification.plots._pool_plot.PoolPlot`
consuming one product, and the plot area lights each one only when its product is
built. All three list under the **Shape** family.
"""
from __future__ import annotations

from cellflow.aggregate_quantification.shape import DESCRIPTOR_COLUMNS, RELATIONAL_COLUMNS
from cellflow.napari.aggregate_quantification.plots._pool_plot import PoolPlot

_FAMILY = "Shape"
_CELL_FIELD = "cell_tracked_labels_path"
_NUCLEUS_FIELD = "nucleus_tracked_labels_path"


class CellShapePlot(PoolPlot):
    plot_id = "cell_shape"
    display_name = "Cell shape"
    family = _FAMILY
    consumes = ("cell_shape",)
    value_columns = DESCRIPTOR_COLUMNS
    label_field = _CELL_FIELD


class NucleusShapePlot(PoolPlot):
    plot_id = "nucleus_shape"
    display_name = "Nucleus shape"
    family = _FAMILY
    consumes = ("nucleus_shape",)
    value_columns = DESCRIPTOR_COLUMNS
    label_field = _NUCLEUS_FIELD


class ShapeRelationalPlot(PoolPlot):
    plot_id = "shape_relational"
    display_name = "Nucleus–cell shape"
    family = _FAMILY
    consumes = ("shape_relational",)
    value_columns = RELATIONAL_COLUMNS
    # Relational rows are keyed by the cell; picking loads the cell labels.
    label_field = _CELL_FIELD
