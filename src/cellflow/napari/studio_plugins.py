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

from collections.abc import Callable, Iterable, Mapping
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
from cellflow.napari.ui_style import action_button, parameter_heading, status_label

#: Signature of the studio callback the Build area invokes to run a build:
#: ``(quantifiers, in_scope_records, overwrite)`` — one job per (quantifier,
#: position), so a single Run can rebuild several metrics at once.
BuildsCallback = Callable[[list[Quantifier], list[dict], bool], None]


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
        contact_analysis_path=Path(out) if out else None,
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


def built_quantity_ids(records: Iterable[dict]) -> frozenset[str]:
    """The ``quantity_id``\\s built for at least one of *records*.

    This is the *product availability* the plot area gates on: a plot whose
    ``consumes`` is a subset of this set is live for the given scope. Mirrors the
    per-quantifier ``is_built(output_for_record(...))`` check the builder uses, so
    "built" means the same thing on both sides of the producer/consumer seam.
    """
    record_list = list(records)
    built: set[str] = set()
    for q_cls in available_quantifiers():
        quantifier = q_cls()
        for record in record_list:
            if quantifier.is_built(output_for_record(quantifier, record)):
                built.add(quantifier.quantity_id)
                break
    return frozenset(built)


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


def available_tool_plugins() -> list[PluginEntry]:
    """The non-build *tools*: the surviving analysis plugins.

    Building is no longer a per-tool plugin — every registered quantifier is a
    *metric* row in the studio's compact :class:`BuildArea` (built here, plotted
    in the Plot area). The tools are the remaining analysis plugins (NLS
    classification, contacts visualizer, catalogue summary).
    """
    entries: list[PluginEntry] = []
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


#: Status-dot colours for a metric's availability across the in-scope positions.
_DOT_ALL = "#3fb950"  # green — built for every applicable position
_DOT_MISSING = "#f85149"  # red — built for some / none
_DOT_NONE = "#6e7681"  # grey — no in-scope position has the inputs


@dataclass
class _MetricRow:
    """One metric's controls in the :class:`BuildArea`."""

    quantifier: Quantifier
    checkbox: QCheckBox
    dot: QLabel


#: Friendly names for the raw ``PositionInputs`` fields a metric can require.
_INPUT_LABELS: dict[str, str] = {
    "cell_labels_path": "cell labels",
    "nucleus_labels_path": "nucleus labels",
    "pixel_size_um": "pixel size",
    "time_interval_s": "frame interval",
}


def producers_by_field(
    quantifiers: Iterable[Quantifier],
) -> dict[str, Quantifier]:
    """Map each produced ``PositionInputs`` field → the quantifier that builds it.

    A metric whose :attr:`~Quantifier.requires` names one of these fields is
    *derived from* that producer (today: the contacts-derived metrics consume the
    ``contact_analysis_path`` that the contacts quantifier produces).
    """
    return {q.produces: q for q in quantifiers if q.produces}


def input_label(field: str, producers: Mapping[str, Quantifier]) -> str:
    """Friendly label for a required input — the producer's display name when the
    input is itself a built quantity, else a static raw-input label."""
    producer = producers.get(field)
    if producer is not None:
        return producer.display_name
    return _INPUT_LABELS.get(field, field)


def metric_input_labels(
    quantifier: Quantifier, producers: Mapping[str, Quantifier]
) -> list[str]:
    """Friendly labels for every input *quantifier* needs to build."""
    return [input_label(field, producers) for field in quantifier.requires]


@dataclass(frozen=True)
class BuildGroup:
    """A header + the metrics under it in the :class:`BuildArea`.

    ``derived`` groups gather metrics computed from another metric's artifact and
    are rendered indented under their producer's group, so the build-dependency
    reads top-to-bottom.
    """

    key: str
    label: str
    derived: bool
    members: tuple[Quantifier, ...]


def _source_group(quantifier: Quantifier) -> tuple[int, str]:
    """(order, label) for a *raw* metric, keyed by which label sources it reads."""
    req = set(quantifier.requires)
    has_cell = "cell_labels_path" in req
    has_nucleus = "nucleus_labels_path" in req
    if has_cell and has_nucleus:
        return (2, "Cell + nucleus")
    if has_nucleus:
        return (1, "Nucleus")
    if has_cell:
        return (0, "Cell")
    return (3, "Other")


def group_build_metrics(quantifiers: Iterable[Quantifier]) -> list[BuildGroup]:
    """Order *quantifiers* into input-typed groups with derived metrics nested.

    Raw metrics are grouped by their source layer (Cell / Nucleus / Cell +
    nucleus); each metric that is derived from a producer is collected into a
    "Derived from …" group placed immediately after that producer's group.
    """
    quantifier_list = list(quantifiers)
    producers = producers_by_field(quantifier_list)
    raw: dict[str, list[Quantifier]] = {}
    raw_order: dict[str, int] = {}
    derived: dict[str, list[Quantifier]] = {}
    for quantifier in quantifier_list:
        producer = next(
            (
                producers[field]
                for field in quantifier.requires
                if field in producers
                and producers[field].quantity_id != quantifier.quantity_id
            ),
            None,
        )
        if producer is not None:
            derived.setdefault(producer.quantity_id, []).append(quantifier)
        else:
            order, label = _source_group(quantifier)
            raw.setdefault(label, []).append(quantifier)
            raw_order[label] = order

    def _sorted(items: list[Quantifier]) -> tuple[Quantifier, ...]:
        return tuple(sorted(items, key=lambda q: q.display_name.lower()))

    groups: list[BuildGroup] = []
    for label in sorted(raw, key=lambda lbl: raw_order[lbl]):
        members = _sorted(raw[label])
        groups.append(BuildGroup(key=label, label=label, derived=False, members=members))
        # Trail each producer's "Derived from …" group right after its own group.
        for producer in members:
            dependents = derived.pop(producer.quantity_id, None)
            if dependents:
                groups.append(
                    BuildGroup(
                        key=f"derived:{producer.quantity_id}",
                        label=f"Derived from {producer.display_name}",
                        derived=True,
                        members=_sorted(dependents),
                    )
                )
    # Defensive: surface any derived group whose producer was not itself listed.
    by_id = {q.quantity_id: q for q in quantifier_list}
    for producer_id, dependents in derived.items():
        name = by_id[producer_id].display_name if producer_id in by_id else producer_id
        groups.append(
            BuildGroup(
                key=f"derived:{producer_id}",
                label=f"Derived from {name}",
                derived=True,
                members=_sorted(dependents),
            )
        )
    return groups


class BuildArea(QWidget):
    """Compact multi-metric build panel.

    Metrics are grouped by their input type (Cell / Nucleus / Cell + nucleus),
    with metrics *derived* from another metric's artifact nested under their
    producer's group ("Derived from …"). Each row is a checkbox + a status dot
    showing whether the metric's artifact exists for *every* in-scope position
    that has the inputs (green = all built, red = some missing, grey = no in-scope
    position has the inputs) + a muted caption of the inputs it needs. A single
    Run button (re)builds every checked metric, overwriting existing artifacts;
    execution is delegated to the studio's *build_callback* so building stays
    centralized (threaded, status-refreshed).
    """

    #: Left margin (px) applied to a derived group's header + rows.
    _DERIVED_INDENT = 14

    def __init__(
        self,
        quantifiers: Iterable[Quantifier],
        build_callback: BuildsCallback,
        *,
        viewer: object | None = None,
        params_provider: Callable[[], Mapping[str, object]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._build_callback = build_callback
        #: Supplies the shared build params (e.g. density's FOV area) so a metric
        #: missing a required param is greyed out rather than failing on Run.
        self._params_provider = params_provider
        self._records: list[dict] = []
        self._rows: dict[str, _MetricRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        quantifier_list = list(quantifiers)
        producers = producers_by_field(quantifier_list)
        for group in group_build_metrics(quantifier_list):
            indent = self._DERIVED_INDENT if group.derived else 0
            header = QLabel(("↳ " if group.derived else "") + group.label)
            parameter_heading(header)
            header.setContentsMargins(indent, 4, 0, 0)
            layout.addWidget(header)
            for quantifier in group.members:
                self._add_metric_row(layout, quantifier, producers, indent)

        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        # Check-all toggles to uncheck-all once everything buildable is ticked, so
        # one button both selects and clears the whole metric list.
        self._check_all_btn = QPushButton("Check all")
        self._check_all_btn.setToolTip("Tick (or untick) every buildable metric.")
        action_button(self._check_all_btn)
        self._check_all_btn.clicked.connect(self._on_toggle_all)
        run_row.addWidget(self._check_all_btn)
        run_row.addStretch(1)
        self._run_btn = QPushButton("Run checked builds")
        self._run_btn.setToolTip("Compute (and overwrite) every checked metric.")
        action_button(self._run_btn)
        self._run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self._run_btn)
        layout.addLayout(run_row)

        self._status = QLabel("")
        status_label(self._status, muted=True)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._refresh()

    def _add_metric_row(
        self,
        layout: QVBoxLayout,
        quantifier: Quantifier,
        producers: Mapping[str, Quantifier],
        indent: int,
    ) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(indent, 0, 0, 0)
        dot = QLabel("●")
        dot.setToolTip("")
        row.addWidget(dot)
        checkbox = QCheckBox(quantifier.display_name)
        checkbox.toggled.connect(self._sync_check_all_label)
        row.addWidget(checkbox)
        row.addStretch(1)
        # Source inputs plus any required shared param (e.g. density's FOV area),
        # so a metric that needs a param is shown to need it rather than silently
        # failing at build time.
        needs = [*metric_input_labels(quantifier, producers),
                 *quantifier.required_build_params.values()]
        if needs:
            caption = QLabel("needs: " + " · ".join(needs))
            status_label(caption, muted=True)
            row.addWidget(caption)
        layout.addLayout(row)
        self._rows[quantifier.quantity_id] = _MetricRow(quantifier, checkbox, dot)

    def set_context(self, ctx: AnalysisContext) -> None:
        self._records = list(ctx.records)
        self._refresh()

    def _refresh(self) -> None:
        params = self._params_provider() if self._params_provider else {}
        any_buildable = False
        for row in self._rows.values():
            applicable = records_satisfying(row.quantifier.requires, self._records)
            total = len(applicable)
            built = sum(
                1
                for record in applicable
                if row.quantifier.is_built(output_for_record(row.quantifier, record))
            )
            missing_params = row.quantifier.missing_build_params(params)
            if total == 0:
                color, tip = _DOT_NONE, "No in-scope position has the inputs."
            elif missing_params:
                # Inputs are present, but a required shared param is not set, so the
                # build can't run — grey it out and say what's missing.
                color = _DOT_NONE
                tip = "Set " + " · ".join(missing_params) + " in Parameters to build."
            elif built == total:
                color, tip = _DOT_ALL, f"Built for all {total} in-scope position(s)."
            else:
                color, tip = (
                    _DOT_MISSING,
                    f"Built for {built} of {total} in-scope position(s).",
                )
            buildable = total > 0 and not missing_params
            row.dot.setStyleSheet(f"color: {color};")
            row.dot.setToolTip(tip)
            row.checkbox.setEnabled(buildable)
            row.checkbox.setToolTip(tip)
            any_buildable = any_buildable or buildable
        self._run_btn.setEnabled(any_buildable)
        self._check_all_btn.setEnabled(any_buildable)
        self._sync_check_all_label()
        self._sync_check_all_label()

    def _buildable_rows(self) -> list[_MetricRow]:
        return [row for row in self._rows.values() if row.checkbox.isEnabled()]

    def _sync_check_all_label(self) -> None:
        """'Check all' flips to 'Uncheck all' once every buildable metric is ticked."""
        buildable = self._buildable_rows()
        all_checked = bool(buildable) and all(r.checkbox.isChecked() for r in buildable)
        self._check_all_btn.setText("Uncheck all" if all_checked else "Check all")

    def _on_toggle_all(self) -> None:
        buildable = self._buildable_rows()
        if not buildable:
            return
        # Uncheck when everything is already on; otherwise check the rest.
        target = not all(r.checkbox.isChecked() for r in buildable)
        for row in buildable:
            row.checkbox.setChecked(target)
        self._sync_check_all_label()

    def _checked_quantifiers(self) -> list[Quantifier]:
        return [
            row.quantifier
            for row in self._rows.values()
            if row.checkbox.isChecked() and row.checkbox.isEnabled()
        ]

    def _on_run(self) -> None:
        chosen = self._checked_quantifiers()
        if not chosen:
            self._status.setText("Tick at least one metric to build.")
            return
        self._status.setText("")
        self._build_callback(chosen, list(self._records), True)
