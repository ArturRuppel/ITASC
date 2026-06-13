"""The Reduce editor — an editable, ordered ``collapse`` pipeline for the panel.

The reduce layer's :class:`~cellflow.aggregate_quantification.reduce.Collapse`
primitive made user-visible: an ordered list of collapse steps the plot panel
applies (via ``PlotSpec.collapse`` → ``reduce_to_units``) to define the
independent unit a comparison aggregates over. Chaining single-rung collapses is
how the pseudoreplication-safe nested reduction is expressed
(``collapse by=[…,cell_id]`` then ``collapse by=[…,position_id]`` →
equal-weighted per-position); a single rung is the flat pooled result. The order
is the knob, and it is the user's.

Each table opens with a sensible **default** pipeline (one collapse to the finest
unit, e.g. per cell), fully editable — add / remove / reorder steps, pick the
``by`` columns and the statistic. An empty pipeline falls back to the panel's
level convenience. The widget is **self-contained and headless-testable**:
:meth:`pipeline` reads the editor's state into a ``tuple[Collapse, …]`` with no
plotting dependency, and :attr:`changed` fires on every edit.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.reduce import COLLAPSE_STATS, Collapse
from cellflow.napari.ui_style import action_button, status_label


class CollapsePipelineEditor(QWidget):
    """An ordered list of editable collapse steps → a ``tuple[Collapse, …]``."""

    #: Emitted on any edit (add / remove / reorder / toggle a column / change stat).
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: Selectable ``by`` columns for the active table (catalogue metadata,
        #: nesting entities, categorical axes).
        self._columns: list[str] = []
        #: The model: one ``{"by": set[str], "stat": str}`` per step, in order.
        self._steps: list[dict] = []
        #: Suppresses ``changed`` while the rows are rebuilt programmatically.
        self._building = False

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)

        # No heading here: the enclosing "Reduce" collapsible section already names
        # this control, so a second "Reduce" label would only be redundant.
        hint = QLabel(
            "Each step collapses to one row per ticked column-combination. Chain "
            "steps to climb levels (cell → position → date); empty = level default."
        )
        hint.setWordWrap(True)
        status_label(hint, muted=True)
        col.addWidget(hint)

        self._steps_box = QVBoxLayout()
        self._steps_box.setContentsMargins(0, 0, 0, 0)
        self._steps_box.setSpacing(4)
        col.addLayout(self._steps_box)

        add_btn = QPushButton("+ collapse step")
        add_btn.setToolTip("Append a collapse step to the pipeline.")
        action_button(add_btn)
        add_btn.clicked.connect(self._on_add)
        col.addWidget(add_btn)

        self._summary = QLabel("")
        status_label(self._summary, muted=True)
        col.addWidget(self._summary)

    # ------------------------------------------------------------------- public
    def set_columns(self, columns: Sequence[str], default: Sequence[Collapse]) -> None:
        """Reset the selectable columns and seed the *default* pipeline (no signal).

        Called when the active product switches: the editor jumps to that table's
        default pipeline, dropping any ``by`` column the new table doesn't carry."""
        self._columns = list(columns)
        allowed = set(self._columns)
        self._steps = [
            {"by": {c for c in step.by if c in allowed}, "stat": step.stat}
            for step in default
        ]
        self._rebuild()

    def pipeline(self) -> tuple[Collapse, ...]:
        """The current pipeline as ``Collapse`` steps (steps with no ``by`` column
        ticked are skipped — an all-empty step is a no-op, not a whole-table
        collapse, which would surprise)."""
        out: list[Collapse] = []
        for step in self._steps:
            by = tuple(c for c in self._columns if c in step["by"])
            if by:
                out.append(Collapse(by=by, stat=step["stat"]))
        return tuple(out)

    # ------------------------------------------------------------------ internal
    def _emit(self) -> None:
        if not self._building:
            self._update_summary()
            self.changed.emit()

    def _on_add(self) -> None:
        self._steps.append({"by": set(), "stat": "mean"})
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
        for index in range(len(self._steps)):
            self._steps_box.addWidget(self._build_step_row(index))
        self._building = False
        self._update_summary()

    def _build_step_row(self, index: int) -> QWidget:
        step = self._steps[index]
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        row = QVBoxLayout(frame)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QLabel(f"{index + 1}. collapse by"))
        header.addStretch(1)
        stat_combo = QComboBox()
        for stat in COLLAPSE_STATS:
            stat_combo.addItem(stat, stat)
        stat_combo.setCurrentText(step["stat"])
        stat_combo.currentTextChanged.connect(lambda text, i=index: self._set_stat(i, text))
        header.addWidget(QLabel("stat:"))
        header.addWidget(stat_combo)
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

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        grid.setColumnStretch(3, 1)
        from qtpy.QtWidgets import QCheckBox  # local: keep the import surface small

        for i, column in enumerate(self._columns):
            cb = QCheckBox(column)
            cb.setChecked(column in step["by"])
            cb.toggled.connect(
                lambda checked, i=index, c=column: self._toggle(i, c, checked)
            )
            grid.addWidget(cb, i // 3, i % 3)
        row.addLayout(grid)
        return frame

    def _set_stat(self, index: int, stat: str) -> None:
        self._steps[index]["stat"] = stat
        self._emit()

    def _toggle(self, index: int, column: str, checked: bool) -> None:
        if checked:
            self._steps[index]["by"].add(column)
        else:
            self._steps[index]["by"].discard(column)
        self._emit()

    def _update_summary(self) -> None:
        pipeline = self.pipeline()
        if not pipeline:
            self._summary.setText("No collapse — uses the level default.")
            return
        parts = [f"[{', '.join(step.by)}] ({step.stat})" for step in pipeline]
        self._summary.setText("→ " + "  →  ".join(parts))
