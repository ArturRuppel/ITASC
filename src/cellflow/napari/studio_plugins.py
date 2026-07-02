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
from typing import Any

from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.quantifier import (
    Quantifier,
)
from cellflow.aggregate_quantification.records import (
    output_for_record,
    position_inputs_from_record,
    records_satisfying,
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


#: Accent colour for full coverage / a set param (the ``✓`` state). Reused for
#: the legend badges and the output-row coverage badges. No red anywhere: an
#: unmet dependency reads off *which* chip is dimmed, not a red row.
_FULL_COLOR = "#3fb950"


@dataclass(frozen=True)
class DependencySpec:
    """How a build dependency is shown in the legend and the chip strips: a short
    abbreviation, a full label, and whether it is a per-position ``file`` input or
    a global ``param``."""

    abbrev: str
    label: str
    kind: str  # "file" | "param"


#: Ordered registry of build dependencies → display metadata, the single source
#: of truth for the input legend and the per-output chip strips. ``file`` inputs
#: are per-position (a quantifier's :attr:`~Quantifier.requires`); ``param``
#: inputs are global (set in the Parameters panel — a quantifier's
#: :attr:`~Quantifier.required_build_params`). ``contact_analysis_path`` is a
#: produced intermediate — the contacts artifact — shown as the ``CA`` file chip
#: on the contacts-derived metrics; its coverage mirrors the contacts build.
DEPENDENCIES: dict[str, DependencySpec] = {
    "cell_labels_path": DependencySpec("C", "cell labels", "file"),
    "nucleus_labels_path": DependencySpec("N", "nucleus labels", "file"),
    "contact_analysis_path": DependencySpec("CA", "contacts", "file"),
    "pixel_size_um": DependencySpec("px", "pixel size", "param"),
    "time_interval_s": DependencySpec("Δt", "frame interval", "param"),
    "fov_area_mm2": DependencySpec("A", "FOV area", "param"),
}


@dataclass
class _MetricRow:
    """One metric's controls in the :class:`BuildArea`.

    ``chips`` maps each dependency field to its abbreviation label so a row can
    dim the chip whose input is fully unmet; ``badge`` shows ``built/applicable``.
    """

    quantifier: Quantifier
    checkbox: QCheckBox
    chips: dict[str, QLabel]
    badge: QLabel


def producers_by_field(
    quantifiers: Iterable[Quantifier],
) -> dict[str, Quantifier]:
    """Map each produced ``PositionInputs`` field → the quantifier that builds it.

    A metric whose :attr:`~Quantifier.requires` names one of these fields is
    *derived from* that producer (today: the contacts-derived metrics consume the
    ``contact_analysis_path`` that the contacts quantifier produces).
    """
    return {q.produces: q for q in quantifiers if q.produces}


def metric_dependencies(quantifier: Quantifier) -> list[str]:
    """Ordered (registry order) dependency fields *quantifier* consumes — its file
    :attr:`~Quantifier.requires` plus its global
    :attr:`~Quantifier.required_build_params`."""
    fields = set(quantifier.requires) | set(quantifier.required_build_params)
    return [field for field in DEPENDENCIES if field in fields]


def referenced_dependencies(quantifiers: Iterable[Quantifier]) -> list[str]:
    """The dependency fields any of *quantifiers* consume, in registry order, so
    the legend lists only what the registered metrics actually use."""
    referenced: set[str] = set()
    for quantifier in quantifiers:
        referenced.update(metric_dependencies(quantifier))
    return [field for field in DEPENDENCIES if field in referenced]


def coverage_badge(built: int, total: int) -> str:
    """``built/total``, suffixed with ``✓`` at full coverage (non-zero total)."""
    text = f"{built}/{total}"
    return f"{text} ✓" if total > 0 and built == total else text


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


def _param_is_set(value: object) -> bool:
    """True when *value* is a positive real number (a satisfied global param)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _coverage_level(built: int, total: int) -> str:
    """``full`` / ``partial`` / ``none`` for a built-of-total coverage count."""
    if total <= 0 or built == 0:
        return "none"
    return "full" if built == total else "partial"


def _style_chip(label: QLabel, *, satisfied: bool) -> None:
    """A dependency abbreviation chip: muted + struck when its input is fully
    unmet (no in-scope position has the file, or the global param is unset), plain
    bold otherwise. Dimming is how a row says *which* dependency blocks it."""
    style = "font-size: 8pt; font-weight: 600;"
    if not satisfied:
        style += " color: palette(mid); text-decoration: line-through;"
    label.setStyleSheet(style)


def _style_badge(label: QLabel, *, level: str) -> None:
    """A coverage badge: accent/green at ``full``, muted at ``none``, plain
    ``partial``. Never red — partial coverage is a normal, buildable state."""
    if level == "full":
        label.setStyleSheet(f"font-size: 8pt; color: {_FULL_COLOR};")
    elif level == "none":
        label.setStyleSheet("font-size: 8pt; color: palette(mid);")
    else:
        label.setStyleSheet("font-size: 8pt;")


class InputLegend(QWidget):
    """The deduped INPUTS legend rendered above the metric rows.

    A **Params** column (global, set/unset) and a **Files** column (per-position
    coverage ``X/Y``), listing only the dependencies the registered metrics
    reference. It is the single source of truth for input coverage; the output
    rows carry bare abbreviation chips keyed back to these entries.
    """

    #: Left margin (px) aligning the legend body under its heading.
    _INDENT = 12

    def __init__(
        self,
        param_fields: list[str],
        file_fields: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        #: field -> (abbrev label, name label, badge label)
        self._entries: dict[str, tuple[QLabel, QLabel, QLabel]] = {}

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(1)
        heading = QLabel("INPUTS")
        parameter_heading(heading)
        col.addWidget(heading)

        grid = QGridLayout()
        grid.setContentsMargins(self._INDENT, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        grid.setColumnMinimumWidth(3, 18)  # gutter between the Params/Files columns
        if param_fields:
            params_hdr = QLabel("Params")
            status_label(params_hdr, muted=True)
            grid.addWidget(params_hdr, 0, 0, 1, 3)
        if file_fields:
            files_hdr = QLabel("Files")
            status_label(files_hdr, muted=True)
            grid.addWidget(files_hdr, 0, 4, 1, 3)
        for i, field in enumerate(param_fields):
            self._add_entry(grid, i + 1, 0, field)
        for i, field in enumerate(file_fields):
            self._add_entry(grid, i + 1, 4, field)
        col.addLayout(grid)

    def _add_entry(
        self, grid: QGridLayout, row: int, base_col: int, field: str
    ) -> None:
        spec = DEPENDENCIES[field]
        abbrev = QLabel(spec.abbrev)
        abbrev.setToolTip(spec.label)
        name = QLabel(spec.label)
        badge = QLabel("")
        grid.addWidget(abbrev, row, base_col)
        grid.addWidget(name, row, base_col + 1)
        grid.addWidget(badge, row, base_col + 2)
        self._entries[field] = (abbrev, name, badge)

    def update_coverage(
        self,
        *,
        param_set: Mapping[str, bool],
        file_coverage: Mapping[str, int],
        total: int,
    ) -> None:
        """Refresh every entry's badge + dimming from the latest scope/params."""
        for field, (abbrev, name, badge) in self._entries.items():
            spec = DEPENDENCIES[field]
            if spec.kind == "param":
                satisfied = bool(param_set.get(field, False))
                badge.setText("set ✓" if satisfied else "unset")
                _style_badge(badge, level="full" if satisfied else "none")
            else:
                built = int(file_coverage.get(field, 0))
                satisfied = built > 0
                badge.setText(coverage_badge(built, total))
                _style_badge(badge, level=_coverage_level(built, total))
            _style_chip(abbrev, satisfied=satisfied)
            status_label(name, muted=not satisfied)


class BuildArea(QWidget):
    """Compact multi-metric build panel.

    An :class:`InputLegend` at the top deduplicates the dependencies the metrics
    consume — global params (set/unset) and per-position file inputs (``X/Y``
    coverage). Below it, metrics are grouped by input type (Cell / Nucleus / Cell
    + nucleus), with metrics *derived* from another metric's artifact nested under
    their producer's group ("Derived from …"). Each row is a checkbox + a strip of
    dependency abbreviation chips (a chip dims + strikes when its input is fully
    unmet) + a ``built/applicable`` coverage badge (``✓`` accent at full coverage,
    plain when partial — never red). A single Run button (re)builds every checked
    metric, overwriting existing artifacts; execution is delegated to the studio's
    *build_callback* so building stays centralized (threaded, status-refreshed).
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
        referenced = referenced_dependencies(quantifier_list)
        param_fields = [f for f in referenced if DEPENDENCIES[f].kind == "param"]
        file_fields = [f for f in referenced if DEPENDENCIES[f].kind == "file"]
        self._legend = InputLegend(param_fields, file_fields)
        layout.addWidget(self._legend)

        outputs_heading = QLabel("OUTPUTS")
        parameter_heading(outputs_heading)
        outputs_heading.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(outputs_heading)
        for group in group_build_metrics(quantifier_list):
            indent = self._DERIVED_INDENT if group.derived else 0
            header = QLabel(("↳ " if group.derived else "") + group.label)
            parameter_heading(header)
            header.setContentsMargins(indent, 4, 0, 0)
            layout.addWidget(header)
            for quantifier in group.members:
                self._add_metric_row(layout, quantifier, indent)

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
        indent: int,
    ) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(indent, 0, 0, 0)
        row.setSpacing(4)
        checkbox = QCheckBox(quantifier.display_name)
        checkbox.toggled.connect(self._sync_check_all_label)
        row.addWidget(checkbox)
        row.addStretch(1)
        # One abbreviation chip per dependency (files then params, registry order).
        # The legend carries the coverage; here the chips just name the inputs and
        # dim when fully unmet, so "why can't I build this" reads off the row.
        chips: dict[str, QLabel] = {}
        for field in metric_dependencies(quantifier):
            spec = DEPENDENCIES[field]
            chip = QLabel(spec.abbrev)
            chip.setToolTip(spec.label)
            _style_chip(chip, satisfied=True)
            row.addWidget(chip)
            chips[field] = chip
        # A producer (contacts) flags the intermediate it emits: "⟶ CA".
        produced = DEPENDENCIES.get(quantifier.produces)
        if produced is not None:
            marker = QLabel(f"⟶ {produced.abbrev}")
            marker.setToolTip(f"produces {produced.label}")
            status_label(marker, muted=True)
            row.addWidget(marker)
        badge = QLabel("")
        badge.setMinimumWidth(40)
        row.addWidget(badge)
        layout.addLayout(row)
        self._rows[quantifier.quantity_id] = _MetricRow(
            quantifier, checkbox, chips, badge
        )

    def set_context(self, ctx: AnalysisContext) -> None:
        self._records = list(ctx.records)
        self._refresh()

    def _refresh(self) -> None:
        params = self._params_provider() if self._params_provider else {}
        inputs = [position_inputs_from_record(record) for record in self._records]
        total = len(self._records)
        # Per-input coverage, computed once — the legend's single source of truth.
        file_coverage = {
            field: sum(1 for inp in inputs if getattr(inp, field, None) is not None)
            for field, spec in DEPENDENCIES.items()
            if spec.kind == "file"
        }
        param_set = {
            field: _param_is_set(params.get(field))
            for field, spec in DEPENDENCIES.items()
            if spec.kind == "param"
        }
        self._legend.update_coverage(
            param_set=param_set, file_coverage=file_coverage, total=total
        )

        any_buildable = False
        for row in self._rows.values():
            quantifier = row.quantifier
            applicable = records_satisfying(quantifier.requires, self._records)
            n_app = len(applicable)
            built = sum(
                1
                for record in applicable
                if quantifier.is_built(output_for_record(quantifier, record))
            )
            missing_params = quantifier.missing_build_params(params)
            buildable = n_app > 0 and not missing_params
            # Built/applicable coverage badge — no red; an unmet dependency shows
            # as a dimmed chip, not a red row.
            row.badge.setText(coverage_badge(built, n_app))
            _style_badge(row.badge, level=_coverage_level(built, n_app))
            for field, chip in row.chips.items():
                if DEPENDENCIES[field].kind == "param":
                    satisfied = param_set.get(field, False)
                else:
                    satisfied = file_coverage.get(field, 0) > 0
                _style_chip(chip, satisfied=satisfied)
            tip = self._row_tip(n_app, built, missing_params)
            row.badge.setToolTip(tip)
            row.checkbox.setEnabled(buildable)
            row.checkbox.setToolTip(tip)
            any_buildable = any_buildable or buildable
        self._run_btn.setEnabled(any_buildable)
        self._check_all_btn.setEnabled(any_buildable)
        self._sync_check_all_label()

    @staticmethod
    def _row_tip(applicable: int, built: int, missing_params: tuple[str, ...]) -> str:
        if applicable == 0:
            return "No in-scope position has the inputs."
        if missing_params:
            return "Set " + " · ".join(missing_params) + " in Parameters to build."
        if built == applicable:
            return f"Built for all {applicable} in-scope position(s)."
        return f"Built for {built} of {applicable} in-scope position(s)."

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
