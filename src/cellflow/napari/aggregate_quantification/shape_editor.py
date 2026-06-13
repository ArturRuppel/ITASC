"""The Shape editor — one ordered ``filter`` + ``collapse`` pipeline for the dock.

The reduce layer's two primitives
(:class:`~cellflow.aggregate_quantification.reduce.Filter` /
:class:`~cellflow.aggregate_quantification.reduce.Collapse`) made user-visible as a
single editable, reorderable list — the plot dock's whole *Shape* step. A
**filter** keeps rows matching ``column op value``; a **collapse** aggregates to one
row per ticked column-combination, fixing the independent unit a comparison
averages over. Chaining single-rung collapses is how the pseudoreplication-safe
nested reduction is expressed (``collapse by cell`` then ``collapse by position`` →
equal-weighted per-position); interleaving a filter (``n ≥ 5`` after a collapse)
drops undersampled units. The order is the knob, and it is the user's.

Two facts make the list pipeline-aware:

* **Position-aware columns.** Each step only offers the columns *present at that
  point* — a collapse drops the identity keys it folds away and **adds ``n``** (the
  group size), so ``filter n ≥ 5`` becomes selectable only *after* a collapse.
* **A row-count trail.** The panel feeds back the running row count after each
  step (:meth:`set_row_counts`); the widget itself stays headless and data-free.

The widget is **self-contained and headless-testable**: :meth:`pipeline` reads the
editor's state into a ``tuple[Filter | Collapse, …]`` with no plotting dependency,
and :attr:`changed` fires on every edit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.reduce import (
    COLLAPSE_STATS,
    FILTER_OPS,
    IDENTITY_COLUMNS,
    Collapse,
    Filter,
    Step,
)
from cellflow.napari.ui_style import action_button, status_label


class ShapePipelineEditor(QWidget):
    """An ordered list of editable filter / collapse steps → a ``tuple[Step, …]``."""

    #: Emitted on any edit (add / remove / reorder / toggle / change op / value …).
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: Base selectable columns for the active table (catalogue metadata,
        #: nesting entities, categorical axes, value columns).
        self._columns: list[str] = []
        #: column → its distinct values for a categorical filter dropdown; a column
        #: absent here is numeric (free entry). ``n`` (group size) is always numeric.
        self._categorical: dict[str, list[str]] = {}
        #: The model: an ordered list of ``filter`` / ``collapse`` step dicts.
        self._steps: list[dict] = []
        #: Per-step running row count (display-only), pushed by the panel.
        self._row_counts: list[int | None] = []
        #: Starting row count shown in the header, or None until first render.
        self._start_count: int | None = None
        #: Per-step count labels, aligned to ``_steps`` (rebuilt with the rows).
        self._count_labels: list[QLabel] = []
        #: Suppresses ``changed`` while the rows are rebuilt programmatically.
        self._building = False

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)

        self._header = QLabel("")
        status_label(self._header, muted=True)
        self._header.setWordWrap(True)
        col.addWidget(self._header)

        hint = QLabel(
            "Filter keeps matching rows; collapse aggregates to one row per ticked "
            "column-combination (adding an n group-size column). Order matters — "
            "chain collapses to climb levels, filter n to drop small units."
        )
        hint.setWordWrap(True)
        status_label(hint, muted=True)
        col.addWidget(hint)

        self._steps_box = QVBoxLayout()
        self._steps_box.setContentsMargins(0, 0, 0, 0)
        self._steps_box.setSpacing(4)
        col.addLayout(self._steps_box)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_filter = QPushButton("+ filter")
        add_filter.setToolTip("Append a filter step (keep rows matching column op value).")
        add_filter.clicked.connect(self._on_add_filter)
        add_collapse = QPushButton("+ collapse")
        add_collapse.setToolTip("Append a collapse step (aggregate to one row per group).")
        add_collapse.clicked.connect(self._on_add_collapse)
        for btn in (add_filter, add_collapse):
            action_button(btn)
            add_row.addWidget(btn)
        add_row.addStretch(1)
        col.addLayout(add_row)

        self._summary = QLabel("")
        status_label(self._summary, muted=True)
        self._summary.setWordWrap(True)
        col.addWidget(self._summary)

    # ------------------------------------------------------------------- public
    def set_columns(
        self,
        columns: Sequence[str],
        default: Sequence[Step],
        *,
        categorical: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Reset the selectable columns and seed the *default* pipeline (no signal).

        *columns* are offered for both filter columns and collapse ``by`` columns;
        *categorical* maps a column to its distinct values (a filter on it offers a
        value dropdown; a column absent from the map is numeric, free entry).
        Called when the active product switches: the editor jumps to that table's
        default pipeline, dropping any column the new table no longer carries."""
        self._columns = list(columns)
        self._categorical = {k: list(v) for k, v in (categorical or {}).items()}
        allowed = set(self._columns)
        steps: list[dict] = []
        for step in default:
            if isinstance(step, Collapse):
                steps.append(
                    {
                        "kind": "collapse",
                        "by": {c for c in step.by if c in allowed},
                        "stat": step.stat,
                    }
                )
            else:  # Filter
                steps.append(
                    {
                        "kind": "filter",
                        "column": step.column,
                        "op": step.op,
                        "value": "" if step.value is None else str(step.value),
                    }
                )
        self._steps = steps
        self._row_counts = []
        self._start_count = None
        self._rebuild()

    def pipeline(self) -> tuple[Step, ...]:
        """The current ordered pipeline as backend ``Filter`` / ``Collapse`` steps.

        A step referencing no column present at its point in the pipeline is
        skipped (a collapse with no ticked column is a no-op, not a whole-table
        collapse; a filter on a since-folded column is dropped)."""
        return tuple(self._step_for(index) for index in self._active_indices())

    def _active_indices(self) -> list[int]:
        """Model indices of the steps that contribute a backend step, in order —
        the alignment between the displayed rows and :meth:`pipeline` / the trail."""
        active: list[int] = []
        for index, step in enumerate(self._steps):
            available = self._columns_before(index)
            if step["kind"] == "collapse":
                if any(c in step["by"] for c in available):
                    active.append(index)
            elif step["column"] in available and step["value"] != "":
                active.append(index)
        return active

    def _step_for(self, index: int) -> Step:
        step = self._steps[index]
        if step["kind"] == "collapse":
            available = self._columns_before(index)
            by = tuple(c for c in available if c in step["by"])
            return Collapse(by=by, stat=step["stat"])
        return Filter(step["column"], step["op"], step["value"])

    def set_row_counts(
        self, counts: Sequence[int | None], start: int | None = None
    ) -> None:
        """Display the running row count after each step (display-only, no signal).

        *counts* aligns to the steps in order; *start* (the pre-pipeline row count)
        heads the section. The panel owns the dataframe and pushes these after each
        render so the trail matches exactly what is plotted."""
        self._row_counts = list(counts)
        if start is not None:
            self._start_count = start
        self._refresh_counts()

    # ------------------------------------------------------------------ internal
    def _columns_before(self, index: int) -> list[str]:
        """The columns present as *input* to step *index*: the base set, with each
        preceding collapse folding away the identity keys it does not group by and
        adding ``n`` (so ``n`` is filterable only after a collapse)."""
        available = list(self._columns)
        for step in self._steps[:index]:
            if step["kind"] != "collapse":
                continue
            by = [c for c in available if c in step["by"]]
            if not by:
                continue  # an empty collapse is a no-op (skipped in ``pipeline``)
            # Value columns survive an aggregation; identity columns survive only
            # when grouped by; ``n`` (the group size) appears.
            survivors = [c for c in available if c not in IDENTITY_COLUMNS and c != "n"]
            available = list(dict.fromkeys([*by, *survivors, "n"]))
        return available

    def _emit(self) -> None:
        if not self._building:
            self._update_summary()
            self.changed.emit()

    def _on_add_filter(self) -> None:
        self._steps.append({"kind": "filter", "column": None, "op": "==", "value": ""})
        self._rebuild()
        self._emit()

    def _on_add_collapse(self) -> None:
        self._steps.append({"kind": "collapse", "by": set(), "stat": "mean"})
        self._rebuild()
        self._emit()

    def _remove(self, index: int) -> None:
        del self._steps[index]
        self._rebuild()
        self._emit()

    def _move(self, index: int, delta: int) -> None:
        target = index + delta
        if 0 <= target < len(self._steps):
            self._steps[index], self._steps[target] = (
                self._steps[target],
                self._steps[index],
            )
            self._rebuild()
            self._emit()

    def _rebuild(self) -> None:
        self._building = True
        while self._steps_box.count():
            item = self._steps_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._count_labels = []
        for index in range(len(self._steps)):
            self._steps_box.addWidget(self._build_step_row(index))
        self._building = False
        self._update_summary()
        self._refresh_counts()

    def _build_step_row(self, index: int) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        row = QVBoxLayout(frame)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        kind = self._steps[index]["kind"]
        header.addWidget(QLabel(f"{index + 1}. {kind}"))
        count = QLabel("")
        status_label(count, muted=True)
        header.addWidget(count)
        self._count_labels.append(count)
        header.addStretch(1)
        for label, delta in (("↑", -1), ("↓", +1)):
            btn = QPushButton(label)
            btn.setFixedWidth(22)
            btn.clicked.connect(lambda _=False, i=index, d=delta: self._move(i, d))
            header.addWidget(btn)
        remove = QPushButton("×")
        remove.setFixedWidth(22)
        remove.setToolTip("Remove this step.")
        remove.clicked.connect(lambda _=False, i=index: self._remove(i))
        header.addWidget(remove)
        row.addLayout(header)

        body = self._build_filter_body if kind == "filter" else self._build_collapse_body
        row.addLayout(body(index))
        return frame

    def _build_filter_body(self, index: int) -> QHBoxLayout:
        step = self._steps[index]
        available = self._columns_before(index)
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(3)

        column_combo = QComboBox()
        for column in available:
            column_combo.addItem(column, column)
        if step["column"] in available:
            column_combo.setCurrentIndex(available.index(step["column"]))
        elif available:
            step["column"] = available[0]  # snap a dangling column to the first
        column_combo.currentTextChanged.connect(
            lambda text, i=index: self._set_filter_column(i, text)
        )
        line.addWidget(column_combo, 2)

        op_combo = QComboBox()
        for op in FILTER_OPS:
            op_combo.addItem(op, op)
        op_combo.setCurrentText(step["op"])
        op_combo.currentTextChanged.connect(lambda text, i=index: self._set_filter_op(i, text))
        line.addWidget(op_combo, 1)

        line.addWidget(self._build_value_control(index, step["column"]), 2)
        return line

    def _build_value_control(self, index: int, column: str | None) -> QWidget:
        """A distinct-value dropdown for a categorical column, else a free numeric
        entry. Editing it writes straight back into the step's ``value``."""
        step = self._steps[index]
        values = self._categorical.get(column) if column is not None else None
        if values:
            combo = QComboBox()
            for value in values:
                combo.addItem(value, value)
            if step["value"] in values:
                combo.setCurrentText(step["value"])
            else:
                step["value"] = combo.currentText()  # default to the first value
            combo.currentTextChanged.connect(
                lambda text, i=index: self._set_filter_value(i, text)
            )
            return combo
        edit = QLineEdit(step["value"])
        edit.setPlaceholderText("value")
        edit.textChanged.connect(lambda text, i=index: self._set_filter_value(i, text))
        return edit

    def _build_collapse_body(self, index: int) -> QGridLayout:
        step = self._steps[index]
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        grid.setColumnStretch(3, 1)

        stat_combo = QComboBox()
        for stat in COLLAPSE_STATS:
            stat_combo.addItem(stat, stat)
        stat_combo.setCurrentText(step["stat"])
        stat_combo.currentTextChanged.connect(lambda text, i=index: self._set_stat(i, text))
        grid.addWidget(QLabel("stat:"), 0, 0)
        grid.addWidget(stat_combo, 0, 1, 1, 2)

        # ``by`` candidates are the present columns minus ``n`` (you collapse by an
        # identity / group axis, not by a derived count).
        candidates = [c for c in self._columns_before(index) if c != "n"]
        for i, column in enumerate(candidates):
            cb = QCheckBox(column)
            cb.setChecked(column in step["by"])
            cb.toggled.connect(lambda checked, i=index, c=column: self._toggle(i, c, checked))
            grid.addWidget(cb, 1 + i // 3, i % 3)
        return grid

    def _set_filter_column(self, index: int, column: str) -> None:
        self._steps[index]["column"] = column
        self._rebuild()  # the value control adapts (categorical vs numeric)
        self._emit()

    def _set_filter_op(self, index: int, op: str) -> None:
        self._steps[index]["op"] = op
        self._emit()

    def _set_filter_value(self, index: int, value: str) -> None:
        self._steps[index]["value"] = value
        self._emit()

    def _set_stat(self, index: int, stat: str) -> None:
        self._steps[index]["stat"] = stat
        self._emit()

    def _toggle(self, index: int, column: str, checked: bool) -> None:
        if checked:
            self._steps[index]["by"].add(column)
        else:
            self._steps[index]["by"].discard(column)
        # A collapse's ``by`` changes which columns later steps see (n, folded keys).
        self._rebuild()
        self._emit()

    def _refresh_counts(self) -> None:
        if self._start_count is not None:
            self._header.setText(f"Shape — {self._start_count:,} rows in")
        for label in self._count_labels:
            label.setText("")
        # Counts arrive in pipeline order; map each onto its displayed step row.
        for k, index in enumerate(self._active_indices()):
            count = self._row_counts[k] if k < len(self._row_counts) else None
            if count is not None and index < len(self._count_labels):
                self._count_labels[index].setText(f"→ {count:,}")

    def _update_summary(self) -> None:
        pipeline = self.pipeline()
        if not pipeline:
            self._summary.setText("No shaping — uses the level default.")
            return
        parts = []
        for step in pipeline:
            if isinstance(step, Collapse):
                parts.append(f"collapse [{', '.join(step.by)}] ({step.stat})")
            else:
                parts.append(f"filter {step.column} {step.op} {step.value}")
        self._summary.setText("→ " + "  →  ".join(parts))
