"""Neighborhood & Density group plugin — adjacency, sorting/mixing, and density.

Reads each in-scope position's existing ``contact_analysis.h5`` (the contacts
quantifier already builds it — there is no Compute here), derives one of four
neighborhood tables
(:mod:`cellflow.aggregate_quantification.contacts.neighborhood`), pools it, and
opens the generic
:class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel` with
sensible value / group-by / plot-type defaults per view:

* **Neighbor count** — adjacency degree per cell, grouped by ``class_label``.
* **Neighbor enrichment** — per-cell observed/expected neighbor-type ratio.
* **Contact-type z-score** — observed vs label-shuffle null (sorting vs mixing).
* **Density** — cells per field-of-view area (mm²).

Thin Qt shell — all maths live in the headless backend / panel. Mirrors the
pool-a-snapshot / launch-a-panel path of Potential Landscape and Track Dynamics'
distribution views, but with a View selector and no Compute section.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.plotting import PositionSource, pool_object_tables
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext, AnalysisPlugin
from cellflow.napari.aggregate_quantification.plugins._plot_dock import PlotDockTabs
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

# matplotlib's Qt canvas needs a running QApplication; probe it so a headless
# environment degrades to a disabled button instead of breaking discovery.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401

    _HAS_MPL_QT = True
except Exception:  # pragma: no cover - exercised only without a Qt matplotlib
    _HAS_MPL_QT = False

#: NLS subpopulation label column, attached to the neighbor-count view by the
#: standard per-position ``cell_id`` join (the same name the shape / track
#: plugins use).
_CLASS_COLUMN = "class_label"

#: view key -> (display label, value column, group-by columns, default plot).
#: Each view pools a different neighborhood table and opens the generic panel
#: with these presets; the existing plot types cover every view (no PlotPanel
#: changes). The typed views (enrichment, z-score) carry their labels in-table;
#: the neighbor-count view joins ``class_label`` per position.
_VIEWS: dict[str, dict[str, Any]] = {
    "neighbor_count": {
        "label": "Neighbor count",
        "value": "n_neighbors",
        "group": ("condition", "date", "position_id", _CLASS_COLUMN),
        "plot": "box",
    },
    "enrichment": {
        "label": "Neighbor enrichment",
        "value": "enrichment",
        "group": ("condition", "focal_label", "neighbor_label"),
        "plot": "box",
    },
    "zscore": {
        "label": "Contact-type z-score",
        "value": "z_score",
        "group": ("contact_type", "condition"),
        "plot": "bar",
    },
    "density": {
        "label": "Density",
        "value": "density",
        "group": ("label", "condition"),
        "plot": "bar",
    },
}
#: Views that need labelled cells; a position without an NLS CSV (or with more
#: than two labels) is skipped for these and noted.
_TYPED_VIEWS = ("enrichment", "zscore")
_DEFAULT_SHUFFLES = 1000


class NeighborhoodPlugin(AnalysisPlugin):
    """Pool a neighborhood/density table and launch the generic plot panel."""

    plugin_id = "neighborhood"
    display_name = "Neighborhood & Density"
    requires = ()

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        self._records: list[dict] = []
        self._pool_worker = None
        self._plot_count = 0
        #: All plots share one dock as tabs (constant size) — see ``_plot_dock.py``.
        self._plot_tabs = PlotDockTabs(self, dock_name="Neighborhood & Density plots")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        layout.addWidget(CollapsibleSection("Plot", self._build_plot(), expanded=True))
        self._update_enabled()
        self._on_view_changed()

    # ------------------------------------------------------------------ plot UI
    def _build_plot(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        intro = QLabel(
            "Neighborhood & density of the contact graph: how many neighbors each "
            "cell has, whether cell types sort or mix (per-cell enrichment and a "
            "label-shuffle z-score), and cells per field-of-view area."
        )
        intro.setWordWrap(True)
        status_label(intro, muted=True)
        col.addWidget(intro)

        self._scope_status = QLabel("No positions in scope.")
        self._scope_status.setWordWrap(True)
        status_label(self._scope_status)
        col.addWidget(self._scope_status)

        self._view_combo = QComboBox()
        for key, cfg in _VIEWS.items():
            self._view_combo.addItem(cfg["label"], key)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        col.addWidget(self._view_combo)

        self._fov_row = QWidget()
        fov = QHBoxLayout(self._fov_row)
        fov.setContentsMargins(0, 0, 0, 0)
        fov.addWidget(QLabel("Field-of-view area (mm²):"))
        self._fov_edit = QLineEdit()
        self._fov_edit.setPlaceholderText("auto")
        self._fov_edit.setToolTip(
            "Field-of-view area in mm² for the Density view, applied to all pooled "
            "positions. Leave blank to use each position's full image area "
            "(H·W·pixel_size²); a position whose pixel size can't be resolved then "
            "reports density unavailable."
        )
        fov.addWidget(self._fov_edit, 1)
        col.addWidget(self._fov_row)

        self._shuffles_row = QWidget()
        shuf = QHBoxLayout(self._shuffles_row)
        shuf.setContentsMargins(0, 0, 0, 0)
        shuf.addWidget(QLabel("Shuffles:"))
        self._shuffles_edit = QLineEdit()
        self._shuffles_edit.setPlaceholderText(str(_DEFAULT_SHUFFLES))
        self._shuffles_edit.setToolTip(
            "Number of label permutations for the contact-type z-score null."
        )
        shuf.addWidget(self._shuffles_edit, 1)
        col.addWidget(self._shuffles_row)

        self._plot_btn = QPushButton("Plot…")
        action_button(self._plot_btn, expand=True)
        self._plot_btn.clicked.connect(self._on_plot)
        col.addWidget(self._plot_btn)

        self._plot_status = QLabel("")
        self._plot_status.setWordWrap(True)
        status_label(self._plot_status, muted=True)
        if not _HAS_MPL_QT:  # pragma: no cover - only without a Qt matplotlib
            self._plot_status.setText("Plotting unavailable (matplotlib Qt backend not usable).")
        col.addWidget(self._plot_status)
        return body

    @property
    def _view(self) -> str:
        return self._view_combo.currentData() or "neighbor_count"

    def _on_view_changed(self, *_: Any) -> None:
        """Show only the inputs the selected view uses."""
        view = self._view
        self._fov_row.setVisible(view == "density")
        self._shuffles_row.setVisible(view == "zscore")

    # -------------------------------------------------------- studio integration
    def set_context(self, ctx: AnalysisContext) -> None:
        if ctx.viewer is not None:
            self.viewer = ctx.viewer
        self._records = list(ctx.records)
        self._update_enabled()

    def _built_records(self) -> list[dict]:
        return [r for r in self._records if _is_built(r)]

    def _update_enabled(self) -> None:
        built = self._built_records()
        if not self._records:
            self._scope_status.setText("No positions in scope.")
        else:
            self._scope_status.setText(
                f"{len(built)} of {len(self._records)} in-scope position(s) have built contacts."
            )
        self._plot_btn.setEnabled(
            bool(built) and _HAS_MPL_QT and self.viewer is not None and self._pool_worker is None
        )

    def _manual_fov(self) -> float | None:
        return _positive_or_none(self._fov_edit.text())

    def _shuffles(self) -> int:
        text = self._shuffles_edit.text().strip()
        if not text:
            return _DEFAULT_SHUFFLES
        try:
            value = int(text)
        except ValueError:
            return _DEFAULT_SHUFFLES
        return value if value > 0 else _DEFAULT_SHUFFLES

    # --------------------------------------------------------- pooling + launching
    def _on_plot(self) -> None:
        records = list(self._records)
        view = self._view
        fov = self._manual_fov()
        shuffles = self._shuffles()
        self._plot_status.setText("Reading contacts…")
        self._pool_worker = object()
        self._update_enabled()

        @thread_worker(connect={"returned": self._on_pool_done, "errored": self._on_pool_error})
        def _worker():
            return (view, *_pool_neighborhood(records, view, fov, shuffles))

        self._pool_worker = _worker()

    def _on_pool_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._plot_status.setText(f"Plot error: {exc}")
        self._update_enabled()

    def _on_pool_done(self, result: tuple[str, pd.DataFrame, list[str]]) -> None:
        self._pool_worker = None
        view, pooled, notes = result
        note_suffix = f" ({'; '.join(notes)})" if notes else ""
        if pooled.empty:
            self._plot_status.setText(f"No data in scope for this view.{note_suffix}")
            self._update_enabled()
            return
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

        cfg = _VIEWS[view]
        panel = PlotPanel(
            pooled,
            value_columns=(cfg["value"],),
            group_columns=tuple(cfg["group"]),
            default_plot=cfg["plot"],
        )
        self._panel = panel
        self._plot_count += 1
        name = f"Plot {self._plot_count}"
        self._plot_tabs.add(panel, name)
        self._plot_status.setText(f"Opened {name} ({len(pooled)} rows).{note_suffix}")
        self._update_enabled()


# ------------------------------------------------------------- pooling (off-thread)
def _is_built(record: dict) -> bool:
    path = record.get("contact_analysis_path")
    return bool(path) and Path(path).is_file()


def _pool_neighborhood(
    records: list[dict], view: str, fov_override: float | None, shuffles: int
) -> tuple[pd.DataFrame, list[str]]:
    """Pool the selected neighborhood table across built positions, off-thread.

    Reads each built ``contact_analysis.h5`` and its NLS sidecar map (empty when
    the position was never classified), derives the *view*'s table, and
    concatenates. The neighbor-count view attaches ``class_label`` through the
    standard per-position join; the typed views carry their labels in-table and
    skip (with a note) a position lacking an NLS map or carrying more than two
    labels. Returns the pooled frame and a list of per-position notes.
    """
    from cellflow.aggregate_quantification.contacts.neighborhood import (
        cell_density,
        cell_neighbor_counts,
        contact_type_zscores,
        neighbor_enrichment,
    )
    from cellflow.aggregate_quantification.contacts.reader import (
        read_position_contact_analysis,
    )

    sources: list[PositionSource] = []
    notes: list[str] = []
    for record in records:
        if not _is_built(record):
            continue
        analysis = read_position_contact_analysis(record["contact_analysis_path"])
        labels = _nls_labels(record)
        pos_id = str(record.get("id", ""))

        if view in _TYPED_VIEWS:
            if not labels:
                notes.append(f"{pos_id}: no NLS classification")
                continue
            if len({*labels.values()}) > 2:
                notes.append(f"{pos_id}: >2 cell types")
                continue

        if view == "neighbor_count":
            table = cell_neighbor_counts(analysis)
            source = PositionSource(
                metadata=_metadata(record),
                table=table,
                join_table=_class_join_table(labels),
                join_columns=(_CLASS_COLUMN,),
            )
        elif view == "enrichment":
            table = neighbor_enrichment(analysis, labels or {})
            source = PositionSource(metadata=_metadata(record), table=table)
        elif view == "zscore":
            table = contact_type_zscores(analysis, labels or {}, n_shuffles=shuffles)
            source = PositionSource(metadata=_metadata(record), table=table)
        else:  # density
            fov = _fov_area_mm2(record, analysis, fov_override)
            if fov is None:
                notes.append(f"{pos_id}: density unavailable (no pixel size)")
            table = cell_density(analysis, labels or {}, fov_area_mm2=fov)
            source = PositionSource(metadata=_metadata(record), table=table)

        if next(iter(source.table.values()), np.empty(0)).size == 0:
            continue
        sources.append(source)

    return pool_object_tables(sources), notes


def _class_join_table(labels: dict[int, str] | None) -> dict[str, np.ndarray]:
    """A ``{cell_id, class_label}`` join table from the NLS map (empty when none)."""
    if not labels:
        return {}
    cell_ids = sorted(labels)
    return {
        "cell_id": np.asarray(cell_ids, dtype=np.int64),
        _CLASS_COLUMN: np.asarray([labels[c] for c in cell_ids], dtype=object),
    }


def _fov_area_mm2(record: dict, analysis: Any, override: float | None) -> float | None:
    """The field-of-view area in mm² for *record*.

    Uses the manual *override* when given (one global value for all positions);
    otherwise the position's full image area ``H·W·pixel_size² / 1e6``, resolved
    from its config / label TIFF. Returns ``None`` when the pixel size or image
    shape can't be resolved, so density reads as unavailable rather than crashing.
    """
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
    """``(height, width)`` of a label TIFF without loading pixels; None on failure."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
    except Exception:  # pragma: no cover - unreadable/missing TIFF → no default FOV
        return None
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    return None


def _nls_labels(record: dict) -> dict[int, str] | None:
    """The position's ``cell_id -> NLS label`` map, or None when not classified."""
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


def _positive_or_none(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _metadata(record: dict) -> dict[str, Any]:
    return {
        "condition": str(record.get("condition", "")),
        "date": str(record.get("date", "")),
        "position_id": str(record.get("id", "")),
    }
