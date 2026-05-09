"""Meta source browser widget for browsing and loading CellFlow meta-study positions."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellflow.meta.catalog import (
    discover_h5_files,
    discover_study,
    load_meta_catalog,
    merge_catalog_records,
    records_from_h5_paths,
    save_meta_catalog,
)

try:  # pragma: no cover - local branch compatibility
    from cellflow.analysis.artifact_reader import read_position_artifact
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_artifact(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.analysis.artifact_reader is unavailable")

try:  # pragma: no cover - local branch compatibility
    from cellflow.napari.artifact_visualization import add_artifact_layers
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def add_artifact_layers(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.napari.artifact_visualization is unavailable")


class MetaSourceBrowserWidget(QWidget):
    """Browse and load positions from a CellFlow meta-study directory.

    Scans a root directory for ``condition/experiment/position`` trees,
    populates cascading combo boxes, and allows loading ready positions
    into the napari viewer.
    """

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._records: list[dict] = []
        self._csv_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # --- Catalog actions ---
        catalog_row = QHBoxLayout()
        self.open_catalog_btn = QPushButton("Open catalog")
        self.save_catalog_btn = QPushButton("Save catalog")
        catalog_row.addWidget(self.open_catalog_btn)
        catalog_row.addWidget(self.save_catalog_btn)
        layout.addLayout(catalog_row)

        source_row = QHBoxLayout()
        self.add_h5_btn = QPushButton("Add H5")
        self.autodiscover_folder_btn = QPushButton("Autodiscover folder")
        source_row.addWidget(self.add_h5_btn)
        source_row.addWidget(self.autodiscover_folder_btn)
        layout.addLayout(source_row)

        metadata_layout = QVBoxLayout()
        metadata_layout.setSpacing(2)
        self.condition_edit = QLineEdit()
        self.condition_edit.setPlaceholderText("Condition")
        self.experiment_edit = QLineEdit()
        self.experiment_edit.setPlaceholderText("Experiment")
        self.position_edit = QLineEdit()
        self.position_edit.setPlaceholderText("Position")
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText("Optional labels")
        for label, line_edit in (
            ("Condition:", self.condition_edit),
            ("Experiment:", self.experiment_edit),
            ("Position:", self.position_edit),
            ("Labels:", self.labels_edit),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(line_edit, 1)
            metadata_layout.addLayout(row)
        layout.addLayout(metadata_layout)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.catalog_tree = QTreeWidget()
        self.catalog_tree.setHeaderHidden(True)
        self.catalog_tree.setMinimumHeight(140)
        layout.addWidget(self.catalog_tree)

        # Internal selectors keep the previous public API and selection logic stable.
        self.condition_combo = QComboBox()
        self.experiment_combo = QComboBox()
        self.position_combo = QComboBox()

        # --- Load button ---
        self.load_source_btn = QPushButton("Load Source")
        layout.addWidget(self.load_source_btn)

        layout.addStretch()

        # --- Wire signals ---
        self.condition_combo.currentTextChanged.connect(self._on_condition_changed)
        self.experiment_combo.currentTextChanged.connect(self._on_experiment_changed)
        self.position_combo.currentTextChanged.connect(self._on_position_changed)
        self.catalog_tree.currentItemChanged.connect(self._on_tree_current_item_changed)
        self.open_catalog_btn.clicked.connect(self._on_open_catalog)
        self.save_catalog_btn.clicked.connect(self._on_save_catalog)
        self.add_h5_btn.clicked.connect(self._on_add_h5)
        self.autodiscover_folder_btn.clicked.connect(self._on_autodiscover_folder)
        self.load_source_btn.clicked.connect(self._on_load_source)

        self._update_load_button()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def refresh(self, root: Path | str | None) -> None:
        """Rescan *root* and repopulate the cascading selectors."""
        self._csv_path = None
        self._set_records([] if root is None else discover_study(Path(root)))

    def _set_records(self, records: list[dict]) -> None:
        """Replace records and repopulate the cascading selectors."""
        self._records = records

        # Block signals so repopulating doesn't fire cascading updates prematurely.
        self.condition_combo.blockSignals(True)
        self.experiment_combo.blockSignals(True)
        self.position_combo.blockSignals(True)

        self.condition_combo.clear()
        self.experiment_combo.clear()
        self.position_combo.clear()
        self.catalog_tree.clear()

        conditions = sorted({r["condition_id"] for r in self._records})
        self.condition_combo.addItems(conditions)

        self.condition_combo.blockSignals(False)
        self.experiment_combo.blockSignals(False)
        self.position_combo.blockSignals(False)

        if conditions:
            self._populate_catalog_tree()
            self._on_condition_changed(conditions[0])
        else:
            self._update_load_button()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _current_record(self) -> dict | None:
        """Return the record matching the three combo selections, if any."""
        cond = self.condition_combo.currentText()
        exp = self.experiment_combo.currentText()
        pos = self.position_combo.currentText()
        if not cond or not exp or not pos:
            return None
        for r in self._records:
            if (
                r["condition_id"] == cond
                and r["experiment_id"] == exp
                and r["position_id"] == pos
            ):
                return r
        return None

    def _update_load_button(self) -> None:
        """Enable *Load Source* only when the selected record is ready."""
        record = self._current_record()
        self.load_source_btn.setEnabled(
            record is not None and record.get("analysis_status") == "ready"
        )

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _catalog_defaults(self, *, include_position: bool) -> dict[str, str]:
        defaults = {
            "date": self.experiment_edit.text().strip() or "unknown_date",
            "condition": self.condition_edit.text().strip() or "unknown_condition",
            "labels": self.labels_edit.text().strip(),
        }
        position = self.position_edit.text().strip()
        if include_position and position:
            defaults["id"] = position
        return defaults

    def _populate_catalog_tree(self) -> None:
        self.catalog_tree.blockSignals(True)
        self.catalog_tree.clear()

        by_condition: dict[str, dict[str, list[dict]]] = {}
        for record in sorted(self._records, key=lambda r: (
            str(r["condition_id"]),
            str(r["experiment_id"]),
            str(r["position_id"]),
        )):
            by_condition.setdefault(record["condition_id"], {}).setdefault(
                record["experiment_id"], []
            ).append(record)

        first_position_item: QTreeWidgetItem | None = None
        for condition, experiments in by_condition.items():
            condition_item = QTreeWidgetItem([condition])
            condition_item.setData(0, Qt.UserRole, ("condition", condition, "", ""))
            self.catalog_tree.addTopLevelItem(condition_item)
            condition_item.setExpanded(True)

            for experiment, records in experiments.items():
                experiment_item = QTreeWidgetItem([experiment])
                experiment_item.setData(
                    0, Qt.UserRole, ("experiment", condition, experiment, "")
                )
                condition_item.addChild(experiment_item)
                experiment_item.setExpanded(True)

                for record in records:
                    position = record["position_id"]
                    position_item = QTreeWidgetItem([position])
                    position_item.setData(
                        0,
                        Qt.UserRole,
                        ("position", condition, experiment, position),
                    )
                    experiment_item.addChild(position_item)
                    if first_position_item is None:
                        first_position_item = position_item

        if first_position_item is not None:
            self.catalog_tree.setCurrentItem(first_position_item)
        self.catalog_tree.blockSignals(False)

    def _select_record_keys(self, condition: str, experiment: str, position: str) -> None:
        self.condition_combo.setCurrentText(condition)
        self.experiment_combo.setCurrentText(experiment)
        self.position_combo.setCurrentText(position)
        self._update_load_button()

    # ------------------------------------------------------------------
    # signal handlers
    # ------------------------------------------------------------------

    def _on_condition_changed(self, text: str) -> None:
        if not text:
            self.experiment_combo.clear()
            self.position_combo.clear()
            self._update_load_button()
            return

        experiments = sorted(
            {r["experiment_id"] for r in self._records if r["condition_id"] == text}
        )

        self.experiment_combo.blockSignals(True)
        self.experiment_combo.clear()
        self.experiment_combo.addItems(experiments)
        self.experiment_combo.blockSignals(False)

        if experiments:
            self._on_experiment_changed(experiments[0])
        else:
            self.position_combo.clear()
            self._update_load_button()

    def _on_experiment_changed(self, text: str) -> None:
        if not text:
            self.position_combo.clear()
            self._update_load_button()
            return

        cond = self.condition_combo.currentText()
        positions = sorted(
            {
                r["position_id"]
                for r in self._records
                if r["condition_id"] == cond and r["experiment_id"] == text
            }
        )

        self.position_combo.blockSignals(True)
        self.position_combo.clear()
        self.position_combo.addItems(positions)
        self.position_combo.blockSignals(False)

        if positions:
            self.position_combo.setCurrentIndex(0)
        self._update_load_button()

    def _on_position_changed(self, _text: str) -> None:
        self._update_load_button()

    def _on_tree_current_item_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            self._update_load_button()
            return
        data = current.data(0, Qt.UserRole)
        if not data:
            self._update_load_button()
            return
        level, condition, experiment, position = data
        if level == "condition":
            experiments = sorted(
                {
                    r["experiment_id"]
                    for r in self._records
                    if r["condition_id"] == condition
                }
            )
            experiment = experiments[0] if experiments else ""
        if level in {"condition", "experiment"}:
            positions = sorted(
                {
                    r["position_id"]
                    for r in self._records
                    if r["condition_id"] == condition and r["experiment_id"] == experiment
                }
            )
            position = positions[0] if positions else ""
        if condition and experiment and position:
            self._select_record_keys(condition, experiment, position)
        else:
            self._update_load_button()

    def _on_open_catalog(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open meta catalog",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not selected:
            return

        csv_path = Path(selected)
        try:
            records = load_meta_catalog(csv_path)
        except (OSError, ValueError) as exc:
            self._set_status(str(exc))
            return

        self._csv_path = csv_path
        self._set_records(records)
        self._set_status(f"Loaded {len(records)} catalog row(s).")

    def _on_save_catalog(self) -> None:
        csv_path = self._csv_path
        if csv_path is None:
            selected, _filter = QFileDialog.getSaveFileName(
                self,
                "Save meta catalog",
                "",
                "CSV Files (*.csv);;All Files (*)",
            )
            if not selected:
                return
            csv_path = Path(selected)

        try:
            save_meta_catalog(csv_path, self._records)
        except OSError as exc:
            self._set_status(str(exc))
            return

        self._csv_path = csv_path
        self._set_status(f"Saved {len(self._records)} catalog row(s).")

    def _on_add_h5(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Add H5 source",
            "",
            "H5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not selected:
            return

        incoming = records_from_h5_paths(
            [Path(selected)],
            defaults=self._catalog_defaults(include_position=True),
        )
        self._set_records(merge_catalog_records(self._records, incoming))
        self._set_status(f"Catalog contains {len(self._records)} source(s).")

    def _on_autodiscover_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Autodiscover H5 sources",
            "",
        )
        if not selected:
            return

        h5_paths = discover_h5_files(Path(selected), recursive=True)
        if not h5_paths:
            self._set_status("No H5 files found.")
            return

        incoming = records_from_h5_paths(
            h5_paths,
            defaults=self._catalog_defaults(include_position=False),
        )
        self._set_records(merge_catalog_records(self._records, incoming))
        self._set_status(f"Catalog contains {len(self._records)} source(s).")

    def _on_load_source(self) -> None:
        record = self._current_record()
        if record is None or record.get("analysis_status") != "ready":
            return
        if self.viewer is None:
            return

        artifact = read_position_artifact(record["artifact_path"])
        add_artifact_layers(self.viewer, artifact, prefix="[Meta] ")
