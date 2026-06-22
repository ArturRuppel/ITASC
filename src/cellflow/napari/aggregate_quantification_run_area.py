"""The Aggregate Quantification studio's single Run section.

One section replaces the old piecemeal Compute + Aggregate areas: it gathers the
run-level choices (which quantities, the optional NLS step, plot rendering) and
hands them — as a :class:`RunChoices` — to the studio, which authors
``catalog.csv`` + ``config.toml`` and drives :func:`pipeline.run`. The shared
**Parameters** bar supplies ``[params]``; this widget owns only the run-level
knobs and the Save/Run controls. Reading state into a plain value keeps the
authoring + threading testable without Qt.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.config import NlsConfig
from cellflow.aggregate_quantification.quantifier import available_quantifiers
from cellflow.napari.ui_style import action_button, parameter_heading, status_label

#: The NLS thresholding methods (mirrors config._NLS_METHODS, in UI order).
_NLS_METHODS = ("auto", "otsu", "two_cluster", "fixed")
_DEFAULT_NLS_IMAGE = "0_input/NLS_zavg.tif"


@dataclass
class RunChoices:
    """The run-level selections the studio threads into ``author_config``."""

    quantities: tuple[str, ...]
    nls: NlsConfig | None
    render_plots: bool
    plot_formats: tuple[str, ...]


class RunArea(QWidget):
    """Quantity / NLS / plots controls + Save config… and Run buttons."""

    def __init__(
        self,
        save_callback: Callable[[RunChoices], None],
        run_callback: Callable[[RunChoices], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._save_callback = save_callback
        self._run_callback = run_callback
        self._records: list[dict] = []
        self._quantity_checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        intro = QLabel(
            "Author catalog.csv + config.toml from the whole catalogue and run the "
            "pipeline (build → aggregate → export). Save config… writes the files "
            "without running."
        )
        intro.setWordWrap(True)
        status_label(intro, muted=True)
        layout.addWidget(intro)

        self._build_quantities(layout)
        self._build_nls(layout)
        self._build_plots(layout)
        self._build_buttons(layout)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        status_label(self._status, muted=True)
        layout.addWidget(self._status)

        self._refresh_enabled()

    # ----------------------------------------------------------------- sections
    def _build_quantities(self, layout) -> None:
        heading = QLabel("QUANTITIES")
        parameter_heading(heading)
        layout.addWidget(heading)
        for q_cls in available_quantifiers():
            cb = QCheckBox(q_cls.display_name or q_cls.quantity_id)
            cb.setChecked(True)
            cb.toggled.connect(lambda *_: self._refresh_enabled())
            layout.addWidget(cb)
            self._quantity_checks[q_cls.quantity_id] = cb

    def _build_nls(self, layout) -> None:
        heading = QLabel("NLS CLASSIFICATION")
        parameter_heading(heading)
        layout.addWidget(heading)
        self._nls_enabled = QCheckBox("Classify NLS subpopulations")
        layout.addWidget(self._nls_enabled)

        grid = QGridLayout()
        grid.setContentsMargins(12, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.addWidget(QLabel("method"), 0, 0)
        self._nls_method = QComboBox()
        self._nls_method.addItems(_NLS_METHODS)
        grid.addWidget(self._nls_method, 0, 1)
        grid.addWidget(QLabel("image"), 1, 0)
        self._nls_image = QLineEdit(_DEFAULT_NLS_IMAGE)
        grid.addWidget(self._nls_image, 1, 1)
        grid.addWidget(QLabel("threshold"), 2, 0)
        self._nls_threshold = QLineEdit("0.0")
        self._nls_threshold.setToolTip("Used when method = fixed.")
        grid.addWidget(self._nls_threshold, 2, 1)
        layout.addLayout(grid)

    def _build_plots(self, layout) -> None:
        heading = QLabel("PLOTS")
        parameter_heading(heading)
        layout.addWidget(heading)
        self._render_plots = QCheckBox("Render figures from the .iris bundles")
        layout.addWidget(self._render_plots)
        row = QHBoxLayout()
        row.setContentsMargins(12, 0, 0, 0)
        row.addWidget(QLabel("formats"))
        self._formats = QLineEdit("png, svg")
        row.addWidget(self._formats, 1)
        layout.addLayout(row)

    def _build_buttons(self, layout) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._save_btn = QPushButton("Save config…")
        self._save_btn.setToolTip("Write catalog.csv + config.toml without running.")
        action_button(self._save_btn)
        self._save_btn.clicked.connect(self._on_save)
        self._run_btn = QPushButton("Run ▶")
        self._run_btn.setToolTip("Write the files, then run the whole pipeline.")
        action_button(self._run_btn, expand=True)
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._save_btn)
        row.addWidget(self._run_btn, 1)
        layout.addLayout(row)

    # -------------------------------------------------------------------- state
    def choices(self) -> RunChoices:
        quantities = tuple(
            qid for qid, cb in self._quantity_checks.items() if cb.isChecked()
        )
        nls = None
        if self._nls_enabled.isChecked():
            nls = NlsConfig(
                enabled=True,
                image=self._nls_image.text().strip() or _DEFAULT_NLS_IMAGE,
                method=self._nls_method.currentText(),
                threshold=_parse_float(self._nls_threshold.text()),
            )
        formats = tuple(
            part.strip() for part in self._formats.text().split(",") if part.strip()
        )
        return RunChoices(
            quantities=quantities,
            nls=nls,
            render_plots=self._render_plots.isChecked(),
            plot_formats=formats or ("png", "svg"),
        )

    def set_context(self, ctx: object) -> None:
        self._records = list(getattr(ctx, "records", []))
        self._refresh_enabled()

    def set_status(self, message: str) -> None:
        self._status.setText(message)

    def _refresh_enabled(self) -> None:
        ready = bool(self._records) and any(
            cb.isChecked() for cb in self._quantity_checks.values()
        )
        self._run_btn.setEnabled(ready)
        self._save_btn.setEnabled(ready)

    def _on_save(self) -> None:
        if self._save_btn.isEnabled():
            self._save_callback(self.choices())

    def _on_run(self) -> None:
        if self._run_btn.isEnabled():
            self._run_callback(self.choices())


def _parse_float(text: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        return 0.0
