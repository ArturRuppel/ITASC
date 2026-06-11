"""The Aggregate Quantification studio's flat plugin list.

One flat plugin list hosts three plugin *roles* uniformly:

* **builder** — a Build button over a :class:`Quantifier`; computes that quantity
  for the in-scope positions. One builder per registered quantifier.
* **processor** / **aggregator** — the existing :class:`AnalysisPlugin`
  widgets (per-position, e.g. NLS classification; or cohort, e.g. catalogue
  summary).

Each :class:`PluginEntry` exposes ``display_name`` + ``requires`` (input gating)
+ a ``factory`` that builds the collapsible body widget. The studio renders one
collapsible per entry; its header is the on/off control (expand to use it) and
the body is fed the catalogue context.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from cellflow.aggregate_quantification.frame_interval import resolve_time_interval_s
from cellflow.aggregate_quantification.pixel_size import resolve_pixel_size_um
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
    """Build :class:`PositionInputs` from a normalized catalogue record.

    ``pixel_size_um`` / ``time_interval_s`` prefer an explicit value on the
    record (a plugin may stamp a manual override there) and otherwise auto-resolve
    from the position's ``cellflow_config.json`` or the label TIFF's tags.
    """
    out = record.get("contact_analysis_path")
    cell = record.get("cell_tracked_labels_path")
    nucleus = record.get("nucleus_tracked_labels_path")
    position_dir = record.get("position_path") or (Path(out).parent if out else Path("."))
    cell_path = Path(cell) if cell else None
    nucleus_path = Path(nucleus) if nucleus else None
    return PositionInputs(
        position_dir=Path(position_dir),
        cell_labels_path=cell_path,
        nucleus_labels_path=nucleus_path,
        pixel_size_um=_resolve_pixel_size(record, position_dir, cell_path),
        time_interval_s=_resolve_time_interval(
            record, position_dir, cell_path or nucleus_path
        ),
    )


def _resolve_pixel_size(
    record: dict, position_dir: Path | str, cell_path: Path | None
) -> float | None:
    explicit = _positive_float(record.get("pixel_size_um"))
    if explicit is not None:
        return explicit
    return resolve_pixel_size_um(position_dir, cell_path)


def _resolve_time_interval(
    record: dict, position_dir: Path | str, label_path: Path | None
) -> float | None:
    explicit = _positive_float(record.get("time_interval_s"))
    if explicit is not None:
        return explicit
    return resolve_time_interval_s(position_dir, label_path)


def _positive_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def output_for_record(quantifier: Quantifier, record: dict) -> Path:
    """Where *quantifier*'s artifact lives for a catalogue *record*.

    Contacts keeps using the catalogue's explicit ``contact_analysis_path``
    column — it predates the quantifier seam and may hold a custom *nested* path
    that the per-position read/visualize paths also rely on, so building must
    target the same file. Every other quantifier derives its destination from
    :meth:`Quantifier.default_output`, so a second quantity no longer inherits
    the contacts artifact path. (Generalize the catalogue to per-quantity output
    columns and this fallback goes away.)
    """
    if quantifier.quantity_id == "contacts":
        explicit = record.get("contact_analysis_path")
        if explicit:
            return Path(explicit)
    return quantifier.default_output(position_inputs_from_record(record))


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
    """All plugin entries: builders (one per quantifier) then meta plugins.

    A quantity whose build UI a group plugin owns (``owns_quantities``) gets no
    generic auto-builder entry — the group plugin offers compute + plot together.
    """
    plugin_classes = available_analysis_plugins()
    owned = {
        quantity_id
        for p_cls in plugin_classes
        for quantity_id in getattr(p_cls, "owns_quantities", ())
    }
    entries: list[PluginEntry] = []
    for q_cls in available_quantifiers():
        quantifier = q_cls()
        if quantifier.quantity_id in owned:
            continue
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
    for p_cls in plugin_classes:
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
