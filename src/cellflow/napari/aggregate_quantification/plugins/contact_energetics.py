"""Potential landscape group plugin — the effective potential / barrier of contacts.

Reads each in-scope position's existing ``contact_analysis.h5`` (the contacts
quantifier already builds it — there is no Compute here), derives the signed
central junction-length reaction coordinate
(:func:`cellflow.aggregate_quantification.contacts.energetics.signed_central_junction_lengths`),
pools it, and opens the generic
:class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel` straight
into the ``potential`` view: ``U(L) = −ln P(L)`` [kT] with the effective barrier
``ΔE_eff`` annotated per curve.

Thin Qt shell — all maths live in the headless backend / panel. Mirrors the
pool-a-snapshot / launch-a-panel path of Track Dynamics' distribution views, but
without a Compute section or a scope selector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
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

#: Metadata axes a potential curve can be grouped by. The signed-length table
#: carries no per-cell identity, so only catalogue metadata is offered.
_GROUP_COLUMNS = ("condition", "date", "position_id")
_VALUE_COLUMNS = ("signed_length",)


class ContactEnergeticsPlugin(AnalysisPlugin):
    """Pool the signed junction-length coordinate and launch the potential landscape."""

    plugin_id = "contact_energetics"
    display_name = "Potential landscape"
    requires = ()

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        self._records: list[dict] = []
        self._pool_worker = None
        self._plot_count = 0
        #: All plots share one dock as tabs (constant size) — see ``_plot_dock.py``.
        self._plot_tabs = PlotDockTabs(self, dock_name="Potential landscape plots")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        layout.addWidget(CollapsibleSection("Plot", self._build_plot(), expanded=True))
        self._update_enabled()

    # ------------------------------------------------------------------ plot UI
    def _build_plot(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        intro = QLabel(
            "Boltzmann-inverts the signed central junction length of T1 events into "
            "an effective potential U(L) = −ln P(L) [kT]; the barrier ΔE_eff is the "
            "energy to reach the four-fold vertex (L → 0)."
        )
        intro.setWordWrap(True)
        status_label(intro, muted=True)
        col.addWidget(intro)

        self._scope_status = QLabel("No positions in scope.")
        self._scope_status.setWordWrap(True)
        status_label(self._scope_status)
        col.addWidget(self._scope_status)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Pixel size (µm/px):"))
        self._pixel_size_edit = QLineEdit()
        self._pixel_size_edit.setPlaceholderText("auto")
        self._pixel_size_edit.setToolTip(
            "µm per pixel for the signed-length axis. Leave blank to auto-resolve "
            "per position from its cellflow_config.json or label TIFF; a value here "
            "applies to all. Unresolved → the axis is in pixels."
        )
        row.addWidget(self._pixel_size_edit, 1)
        col.addLayout(row)

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

    def _manual_pixel_size(self) -> float | None:
        text = self._pixel_size_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    # --------------------------------------------------------- pooling + launching
    def _on_plot(self) -> None:
        records = list(self._records)
        pixel_override = self._manual_pixel_size()
        self._plot_status.setText("Reading contacts…")
        self._pool_worker = object()
        self._update_enabled()

        @thread_worker(connect={"returned": self._on_pool_done, "errored": self._on_pool_error})
        def _worker():
            return _pool_energetics(records, pixel_override)

        self._pool_worker = _worker()

    def _on_pool_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._plot_status.setText(f"Plot error: {exc}")
        self._update_enabled()

    def _on_pool_done(self, pooled: pd.DataFrame) -> None:
        self._pool_worker = None
        if pooled.empty:
            self._plot_status.setText("No T1 junction lengths in scope.")
            self._update_enabled()
            return
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

        # Offer "group by contact type" only when the edges actually carry more
        # than one label (the build leaves cell-cell edges unlabelled unless a
        # tagging step populated ``edge_label``); blanks read as "unlabelled".
        group_columns = _GROUP_COLUMNS
        if "contact_type" in pooled.columns:
            pooled = pooled.copy()
            pooled["contact_type"] = pooled["contact_type"].replace("", "unlabelled")
            if pooled["contact_type"].nunique() > 1:
                group_columns = (*_GROUP_COLUMNS, "contact_type")

        panel = PlotPanel(
            pooled,
            value_columns=_VALUE_COLUMNS,
            group_columns=group_columns,
            default_plot="potential",
            default_adaptive_bins=True,
        )
        self._panel = panel
        self._plot_count += 1
        name = f"Plot {self._plot_count}"
        self._plot_tabs.add(panel, name)
        self._plot_status.setText(f"Opened {name} ({len(pooled)} samples).")
        self._update_enabled()


# ------------------------------------------------------------- pooling (off-thread)
def _is_built(record: dict) -> bool:
    path = record.get("contact_analysis_path")
    return bool(path) and Path(path).is_file()


def _pool_energetics(records: list[dict], pixel_override: float | None) -> pd.DataFrame:
    """Pool the signed central junction-length table across built positions.

    Reads each built ``contact_analysis.h5``, derives its signed coordinate (µm
    when a pixel size resolves, else px), and concatenates. A position whose
    artifact is missing or yields no T1 edges contributes nothing. Runs off the
    GUI thread.
    """
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
        pixel = pixel_override
        if pixel is None:
            pixel = resolve_pixel_size_um(
                record.get("position_path"), analysis.cell_tracked_labels_path
            )
        table = signed_central_junction_lengths(analysis, pixel_size_um=pixel)
        if table["signed_length"].size == 0:
            continue
        sources.append(PositionSource(metadata=_metadata(record), table=table))
    return pool_object_tables(sources)


def _metadata(record: dict) -> dict[str, Any]:
    return {
        "condition": str(record.get("condition", "")),
        "date": str(record.get("date", "")),
        "position_id": str(record.get("id", "")),
    }
