"""Top-level button to (re)compute Cellpose probability z-averages.

Reads `1_cellpose/{nucleus,cell}_prob_3dt.tif` and writes the sigmoid +
z-averaged `*_prob_zavg.tif` files that the rest of the pipeline consumes.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.ui_style import action_button, status_label
from cellflow.segmentation.cellpose_probability_zavg import (
    write_cellpose_probability_zavgs_for_position,
)


class CellposeZavgVizWidget(QWidget):
    """Compute `*_prob_zavg.tif` from `*_prob_3dt.tif` for the current position."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pos_dir: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        self.compute_btn = QPushButton("Compute prob z-averages")
        action_button(self.compute_btn, expand=True)
        btn_row.addWidget(self.compute_btn)
        layout.addLayout(btn_row)

        self.status_lbl = QLabel("")
        status_label(self.status_lbl, muted=True)
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

        self.compute_btn.clicked.connect(self._on_compute)
        self._update_enabled()

    def refresh(self, pos_dir: Path | str | None) -> None:
        if pos_dir is None or str(pos_dir) == "[no project]":
            self._pos_dir = None
        else:
            self._pos_dir = Path(pos_dir)
        self._update_enabled()

    def _update_enabled(self) -> None:
        if self._pos_dir is None:
            self.compute_btn.setEnabled(False)
            self.status_lbl.setText("no project open")
            return

        cellpose_dir = self._pos_dir / "1_cellpose"
        missing = [
            name
            for name in ("nucleus_prob_3dt.tif", "cell_prob_3dt.tif")
            if not (cellpose_dir / name).is_file()
        ]
        if missing:
            self.compute_btn.setEnabled(False)
            self.status_lbl.setText(f"missing: {', '.join(missing)}")
        else:
            self.compute_btn.setEnabled(True)
            self.status_lbl.setText("")

    def _on_compute(self) -> None:
        if self._pos_dir is None:
            return
        try:
            result = write_cellpose_probability_zavgs_for_position(self._pos_dir)
        except Exception as exc:
            self.status_lbl.setText(f"error: {exc}")
            return
        self.status_lbl.setText(result.message)
