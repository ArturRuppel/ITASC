"""The Aggregate Quantification studio's flat plugin list.

One checkbox list hosts three plugin *roles* uniformly:

* **builder** — a Build button over a :class:`Quantifier`; computes that quantity
  for the in-scope positions. One builder per registered quantifier.
* **processor** / **aggregator** — the existing :class:`AnalysisPlugin`
  widgets (per-position, e.g. NLS classification; or cohort, e.g. catalogue
  summary).

Each :class:`PluginEntry` exposes ``display_name`` + ``requires`` (input gating)
+ a ``factory`` that builds the collapsible body widget. The studio renders one
checkbox per entry; checking mounts the body as its own collapsible and feeds it
the catalogue context.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from cellflow.aggregate_quantification.quantifier import (
    PositionInputs,
    Quantifier,
    available_quantifiers,
)
from cellflow.napari.aggregate_quantification.plugins import (
    AnalysisContext,
    available_analysis_plugins,
)
from cellflow.napari.ui_style import action_button, status_label

#: Signature of the studio callback a builder plugin invokes to run a build:
#: ``(quantifier, in_scope_records, overwrite)``.
BuildCallback = Callable[[Quantifier, list[dict], bool], None]


@dataclass(frozen=True)
class PluginEntry:
    """A row in the studio's plugin list."""

    plugin_id: str
    display_name: str
    requires: tuple[str, ...]
    factory: Callable[[Any], QWidget]  # factory(viewer) -> body widget


def position_inputs_from_record(record: dict) -> PositionInputs:
    """Build :class:`PositionInputs` from a normalized catalogue record."""
    out = record.get("contact_analysis_path")
    cell = record.get("cell_tracked_labels_path")
    nucleus = record.get("nucleus_tracked_labels_path")
    position_dir = record.get("position_path") or (Path(out).parent if out else Path("."))
    return PositionInputs(
        position_dir=Path(position_dir),
        cell_labels_path=Path(cell) if cell else None,
        nucleus_labels_path=Path(nucleus) if nucleus else None,
    )


def records_satisfying(requires: Iterable[str], records: Iterable[dict]) -> list[dict]:
    """The records whose inputs supply every field in *requires*."""
    needed = tuple(requires)
    if not needed:
        return list(records)
    out = []
    for record in records:
        inputs = position_inputs_from_record(record)
        if all(getattr(inputs, name, None) is not None for name in needed):
            out.append(record)
    return out


def available_studio_plugins(*, build_callback: BuildCallback) -> list[PluginEntry]:
    """All plugin entries: builders (one per quantifier) then meta plugins."""
    entries: list[PluginEntry] = []
    for q_cls in available_quantifiers():
        quantifier = q_cls()
        entries.append(
            PluginEntry(
                plugin_id=f"build:{quantifier.quantity_id}",
                display_name=f"Build: {quantifier.display_name}",
                requires=tuple(quantifier.requires),
                factory=lambda viewer, q=quantifier: BuilderPlugin(
                    q, build_callback, viewer=viewer
                ),
            )
        )
    for p_cls in available_analysis_plugins():
        entries.append(
            PluginEntry(
                plugin_id=p_cls.plugin_id,
                display_name=p_cls.display_name,
                requires=tuple(getattr(p_cls, "requires", ())),
                factory=lambda viewer, c=p_cls: c(viewer),
            )
        )
    return entries


class BuilderPlugin(QWidget):
    """A Build button over a :class:`Quantifier`.

    Builds the in-scope positions whose inputs satisfy the quantifier. Reads the
    catalogue scope via :meth:`set_context`; delegates execution to the studio's
    *build_callback* so building stays centralized (threaded, status-refreshed).
    """

    def __init__(
        self,
        quantifier: Quantifier,
        build_callback: BuildCallback,
        *,
        viewer: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._quantifier = quantifier
        self._build_callback = build_callback
        self._records: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._overwrite_cb = QCheckBox("Recompute (overwrite existing)")
        layout.addWidget(self._overwrite_cb)

        row = QHBoxLayout()
        self._build_btn = QPushButton(f"Build {quantifier.display_name}")
        action_button(self._build_btn)
        self._build_btn.clicked.connect(self._on_build)
        row.addWidget(self._build_btn)
        layout.addLayout(row)

        self._status = QLabel("")
        status_label(self._status, muted=True)
        layout.addWidget(self._status)

    @property
    def overwrite(self) -> bool:
        return self._overwrite_cb.isChecked()

    def set_context(self, ctx: AnalysisContext) -> None:
        self._records = list(ctx.records)
        buildable = records_satisfying(self._quantifier.requires, self._records)
        self._build_btn.setEnabled(bool(buildable))
        self._status.setText(
            f"{len(buildable)} of {len(self._records)} in-scope position(s) "
            f"have the inputs for {self._quantifier.display_name}."
        )

    def _on_build(self) -> None:
        self._build_callback(self._quantifier, list(self._records), self.overwrite)
