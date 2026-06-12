"""Contacts family plots — derived views over the ``contacts`` product.

Every plot here consumes the single ``contacts`` product (the
``contact_analysis.h5`` the contacts quantifier builds) and derives a different
table from it, then opens the generic
:class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel`:

* **Potential landscape** — Boltzmann-inverts the signed central junction length
  of T1 events into ``U(L) = −ln P(L)`` [kT].
* **Neighbor count / enrichment / contact-type z-score / density** — adjacency,
  sorting-vs-mixing, and density of the contact graph.

These read the contacts HDF5 and derive their own tables (not the standard
``object_table``), so each overrides :meth:`PoolPlot.prepare` with a bespoke pool
while reusing ``PoolPlot``'s ``PlotPanel`` construction. All list under the
**Contacts** family and are lit whenever ``contacts`` is built for the scope.

Plot-time tuning that the old plugins exposed (a manual pixel size, field-of-view
area, and shuffle count) auto-resolves here: pixel size / FOV resolve per
position and the z-score uses the default shuffle count. Those knobs can return
as panel controls if needed; the common path needs none of them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cellflow.aggregate_quantification.plotting import PositionSource, pool_object_tables
from cellflow.napari.aggregate_quantification.plots import PlotParams
from cellflow.napari.aggregate_quantification.plots._pool_plot import PoolPlot
from cellflow.napari.aggregate_quantification.plots._pooling import (
    CLASS_COLUMN,
    position_metadata,
)

_FAMILY = "Contacts"


def _is_built(record: dict) -> bool:
    path = record.get("contact_analysis_path")
    return bool(path) and Path(path).is_file()


def _nls_labels(record: dict) -> dict[int, str] | None:
    """The position's ``cell_id -> NLS label`` map, or ``None`` when unclassified."""
    from cellflow.aggregate_quantification.contacts.nls_classification import (
        nls_classification_csv_path,
        read_nls_classification_csv,
    )

    position_path = record.get("position_path")
    if not position_path:
        return None
    csv_path = nls_classification_csv_path(position_path)
    if not csv_path.is_file():
        return None
    return read_nls_classification_csv(csv_path)


# ------------------------------------------------------------- potential landscape
class ContactEnergeticsPlot(PoolPlot):
    """Effective potential / barrier of T1 junction lengths."""

    plot_id = "contact_energetics"
    display_name = "Potential landscape"
    family = _FAMILY
    render_type = "potential"
    consumes = ("contacts",)
    value_columns = ("signed_length",)
    # The signed-length table carries no per-cell identity; offer catalogue
    # metadata + the contact type (always present after prepare normalizes it).
    group_columns = ("condition", "date", "position_id", "contact_type")
    default_plot = "potential"
    default_adaptive_bins = True
    join_class = False

    def prepare(self, records: list[dict], params: PlotParams = PlotParams()) -> pd.DataFrame:
        from cellflow.aggregate_quantification.contacts.energetics import (
            signed_central_junction_lengths,
        )
        from cellflow.aggregate_quantification.contacts.reader import (
            read_position_contact_analysis,
        )
        from cellflow.aggregate_quantification.pixel_size import resolve_pixel_size_um

        sources: list[PositionSource] = []
        for record in records:
            if not _is_built(record):
                continue
            analysis = read_position_contact_analysis(record["contact_analysis_path"])
            # A shared pixel size overrides per-position auto-resolution; blank
            # (None) falls back to each position's config / label TIFF.
            pixel = params.pixel_size_um
            if pixel is None:
                pixel = resolve_pixel_size_um(
                    record.get("position_path"), analysis.cell_tracked_labels_path
                )
            table = signed_central_junction_lengths(
                analysis, pixel_size_um=pixel, labels=_nls_labels(record)
            )
            if table["signed_length"].size == 0:
                continue
            sources.append(PositionSource(metadata=position_metadata(record), table=table))
        pooled = pool_object_tables(sources)
        # Guarantee the contact_type group axis exists (blank → "unlabelled") so
        # the static group_columns above never references a missing column.
        if not pooled.empty:
            if "contact_type" not in pooled.columns:
                pooled["contact_type"] = "unlabelled"
            else:
                pooled["contact_type"] = pooled["contact_type"].replace("", "unlabelled")
        return pooled


# ----------------------------------------------------------- neighborhood & density
class _NeighborhoodPlot(PoolPlot):
    """One neighborhood/density view derived from the contact graph."""

    family = _FAMILY
    consumes = ("contacts",)
    join_class = False
    #: Views needing labelled cells skip (silently) a position lacking an NLS map
    #: or carrying more than two labels.
    typed: bool = False

    def _derive(
        self, analysis: Any, labels: dict[int, str] | None, record: dict, params: PlotParams
    ) -> dict | None:
        """Return this view's per-position table (column-major), or ``None`` to skip."""
        raise NotImplementedError

    def prepare(self, records: list[dict], params: PlotParams = PlotParams()) -> pd.DataFrame:
        from cellflow.aggregate_quantification.contacts.reader import (
            read_position_contact_analysis,
        )

        sources: list[PositionSource] = []
        for record in records:
            if not _is_built(record):
                continue
            analysis = read_position_contact_analysis(record["contact_analysis_path"])
            labels = _nls_labels(record)
            if self.typed and (not labels or len({*labels.values()}) > 2):
                continue
            source = self._derive(analysis, labels, record, params)
            if source is None:
                continue
            if next(iter(source.table.values()), np.empty(0)).size == 0:
                continue
            sources.append(source)
        return pool_object_tables(sources)


class NeighborCountPlot(_NeighborhoodPlot):
    plot_id = "neighbor_count"
    display_name = "Neighbor count"
    value_columns = ("n_neighbors",)
    group_columns = ("condition", "date", "position_id", CLASS_COLUMN)
    default_plot = "box"

    def _derive(self, analysis, labels, record, params):
        from cellflow.aggregate_quantification.contacts.neighborhood import (
            cell_neighbor_counts,
        )

        return PositionSource(
            metadata=position_metadata(record),
            table=cell_neighbor_counts(analysis),
            join_table=_class_join_table(labels),
            join_columns=(CLASS_COLUMN,),
        )


class NeighborEnrichmentPlot(_NeighborhoodPlot):
    plot_id = "neighbor_enrichment"
    display_name = "Neighbor enrichment"
    value_columns = ("enrichment",)
    group_columns = ("condition", "focal_label", "neighbor_label")
    default_plot = "box"
    typed = True

    def _derive(self, analysis, labels, record, params):
        from cellflow.aggregate_quantification.contacts.neighborhood import (
            neighbor_enrichment,
        )

        return PositionSource(
            metadata=position_metadata(record),
            table=neighbor_enrichment(analysis, labels or {}),
        )


class ContactTypeZScorePlot(_NeighborhoodPlot):
    plot_id = "contact_type_zscore"
    display_name = "Contact-type z-score"
    render_type = "bar"
    value_columns = ("z_score",)
    group_columns = ("contact_type", "condition")
    default_plot = "bar"
    typed = True

    def _derive(self, analysis, labels, record, params):
        from cellflow.aggregate_quantification.contacts.neighborhood import (
            contact_type_zscores,
        )

        return PositionSource(
            metadata=position_metadata(record),
            table=contact_type_zscores(analysis, labels or {}, n_shuffles=params.shuffles),
        )


class DensityPlot(_NeighborhoodPlot):
    plot_id = "cell_density"
    display_name = "Density"
    render_type = "bar"
    value_columns = ("density",)
    group_columns = ("label", "condition")
    default_plot = "bar"

    def _derive(self, analysis, labels, record, params):
        from cellflow.aggregate_quantification.contacts.neighborhood import cell_density

        fov = _fov_area_mm2(record, analysis, params.fov_area_mm2)
        return PositionSource(
            metadata=position_metadata(record),
            table=cell_density(analysis, labels or {}, fov_area_mm2=fov),
        )


def _class_join_table(labels: dict[int, str] | None) -> dict[str, np.ndarray]:
    """A ``{cell_id, class_label}`` join table from the NLS map (empty when none)."""
    if not labels:
        return {}
    cell_ids = sorted(labels)
    return {
        "cell_id": np.asarray(cell_ids, dtype=np.int64),
        CLASS_COLUMN: np.asarray([labels[c] for c in cell_ids], dtype=object),
    }


def _fov_area_mm2(record: dict, analysis: Any, override: float | None) -> float | None:
    """The field-of-view area in mm² for *record*.

    Uses the shared *override* when given (one value for all positions);
    otherwise the position's full image area ``H·W·pixel_size² / 1e6``. Returns
    ``None`` when the pixel size or image shape can't be resolved."""
    if override is not None:
        return override
    from cellflow.aggregate_quantification.pixel_size import resolve_pixel_size_um

    pixel = resolve_pixel_size_um(record.get("position_path"), analysis.cell_tracked_labels_path)
    if pixel is None:
        return None
    shape = _image_shape_hw(analysis.cell_tracked_labels_path)
    if shape is None:
        return None
    height, width = shape
    return height * width * pixel * pixel / 1e6


def _image_shape_hw(path: str) -> tuple[int, int] | None:
    """``(height, width)`` of a label TIFF without loading pixels; ``None`` on failure."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
    except Exception:  # pragma: no cover - unreadable/missing TIFF → no default FOV
        return None
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    return None
