"""
Consensus Segmentation tab — two-phase workflow.

Phase 1 — Segmentation
  Load a 4D (T, Z, H, W) image layer.  Define one or more Cellpose parameter
  configs.  Run segmentation; each config produces a (T, Z, H, W) NPZ saved
  to disk with full metadata.

Phase 2 — Consensus
  Scan a directory for NPZ intermediates; select which files to include.
  Run spatial consensus (votes across all config × z-plane combinations) and
  an optional temporal consistency filter to produce a (T, H, W) Labels layer.
"""

import json
import datetime
import os
import numpy as np
import napari

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QCheckBox, QComboBox,
    QRadioButton, QButtonGroup,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar, QScrollArea, QSpinBox, QListWidget,
)
from napari.qt.threading import thread_worker

from napariTissueFlow.segtrack._segmentation_tab import _dspin, SEG_DEFAULTS


# ── Per-config parameter row ────────────────────────────────────────────────

class _ConfigRow(QWidget):
    """Compact row representing one Cellpose parameter configuration."""

    def __init__(self, index: int, on_delete, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(4)

        self._idx_lbl = QLabel(f"#{index + 1}")
        self._idx_lbl.setFixedWidth(26)

        self._flow = _dspin(-10, 10, SEG_DEFAULTS["flow_threshold"], 2, 0.05)
        self._flow.setFixedWidth(62)
        self._prob = _dspin(-10, 10, SEG_DEFAULTS["cellprob_threshold"], 1, 0.5)
        self._prob.setFixedWidth(55)

        self._minsize = QSpinBox()
        self._minsize.setRange(1, 10000)
        self._minsize.setValue(SEG_DEFAULTS["min_size"])
        self._minsize.setFixedWidth(58)

        self._auto_diam = QCheckBox("auto")
        self._auto_diam.setChecked(SEG_DEFAULTS["auto_diameter"])
        self._diam = _dspin(1.0, 2000.0, SEG_DEFAULTS["diameter"], 1, 1.0)
        self._diam.setFixedWidth(62)
        self._diam.setEnabled(not SEG_DEFAULTS["auto_diameter"])
        self._auto_diam.toggled.connect(lambda c: self._diam.setEnabled(not c))

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(24)
        del_btn.setFixedHeight(22)
        del_btn.clicked.connect(on_delete)

        for w in (
            self._idx_lbl,
            QLabel("flow:"), self._flow,
            QLabel("prob:"), self._prob,
            QLabel("minpx:"), self._minsize,
            QLabel("diam:"), self._auto_diam, self._diam,
            del_btn,
        ):
            lay.addWidget(w)
        lay.addStretch()

    def set_index(self, i: int):
        self._idx_lbl.setText(f"#{i + 1}")

    def collect(self) -> dict:
        return {
            "flow_threshold":     self._flow.value(),
            "cellprob_threshold": self._prob.value(),
            "min_size":           self._minsize.value(),
            "diameter":           None if self._auto_diam.isChecked() else self._diam.value(),
        }


# ── Main tab ────────────────────────────────────────────────────────────────

class ConsensusTab(QWidget):
    """Two-phase consensus segmentation: segment with multiple configs, then vote."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self._worker            = None
        self._is_running        = False
        self._stack_data        = None   # (T, Z, H, W)
        self._secondary_data    = None   # (T, Z, H, W) or None
        self._input_scale       = None
        self._cp_model          = None
        self._cp_model_key      = None
        self._custom_model_path = None
        self._config_rows: list[_ConfigRow] = []

        self._setup_ui()

    # ── UI ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Input ──
        input_box = QGroupBox("Input — 4D stack (T × Z × H × W, all Z-planes used)")
        input_lay = QVBoxLayout(input_box)

        mode_row = QHBoxLayout()
        self._rb_single = QRadioButton("Single Channel")
        self._rb_two_ch = QRadioButton("Two Channel")
        self._rb_single.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        for rb in (self._rb_single, self._rb_two_ch):
            mode_row.addWidget(rb)
            self._mode_grp.addButton(rb)
        self._mode_grp.buttonClicked.connect(self._on_mode_changed)
        input_lay.addLayout(mode_row)

        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load Primary")
        self._load_btn.setFixedWidth(120)
        self._load_btn.setFixedHeight(25)
        self._load_btn.clicked.connect(self._on_load)
        self._input_status = QLabel("Not loaded")
        self._input_status.setWordWrap(True)
        clr_btn = QPushButton("Clear")
        clr_btn.setFixedWidth(50)
        clr_btn.setFixedHeight(25)
        clr_btn.clicked.connect(self._on_clear)
        load_row.addWidget(self._load_btn)
        load_row.addWidget(self._input_status)
        load_row.addWidget(clr_btn)
        input_lay.addLayout(load_row)

        self._secondary_row = QWidget()
        sec_lay = QHBoxLayout(self._secondary_row)
        sec_lay.setContentsMargins(0, 0, 0, 0)
        self._load_sec_btn = QPushButton("Load Secondary")
        self._load_sec_btn.setFixedWidth(120)
        self._load_sec_btn.setFixedHeight(25)
        self._load_sec_btn.clicked.connect(self._on_load_secondary)
        self._secondary_status = QLabel("Not loaded")
        self._secondary_status.setWordWrap(True)
        clr_sec_btn = QPushButton("Clear")
        clr_sec_btn.setFixedWidth(50)
        clr_sec_btn.setFixedHeight(25)
        clr_sec_btn.clicked.connect(self._on_clear_secondary)
        sec_lay.addWidget(self._load_sec_btn)
        sec_lay.addWidget(self._secondary_status)
        sec_lay.addWidget(clr_sec_btn)
        self._secondary_row.setVisible(False)
        input_lay.addWidget(self._secondary_row)

        root.addWidget(input_box)

        # ── Phase 1: Segmentation ──
        seg_box = QGroupBox("Phase 1 — Segmentation")
        seg_lay = QVBoxLayout(seg_box)

        # Shared model settings
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["cpsam", "custom"])
        self._model_combo.setCurrentText(SEG_DEFAULTS["model_type"])
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo)
        self._gpu_chk = QCheckBox("GPU")
        self._gpu_chk.setChecked(SEG_DEFAULTS["gpu"])
        model_row.addWidget(self._gpu_chk)
        model_row.addStretch()
        seg_lay.addLayout(model_row)

        self._custom_row = QWidget()
        cust_lay = QHBoxLayout(self._custom_row)
        cust_lay.setContentsMargins(0, 0, 0, 0)
        self._custom_path_edit = QLineEdit()
        self._custom_path_edit.setPlaceholderText("path/to/model.pt")
        self._custom_browse_btn = QPushButton("Browse")
        self._custom_browse_btn.setFixedWidth(65)
        self._custom_browse_btn.clicked.connect(self._browse_custom_model)
        cust_lay.addWidget(QLabel("Custom model:"))
        cust_lay.addWidget(self._custom_path_edit)
        cust_lay.addWidget(self._custom_browse_btn)
        self._custom_row.setVisible(False)
        seg_lay.addWidget(self._custom_row)

        # Config list (scrollable)
        header = QHBoxLayout()
        header.addWidget(QLabel("Parameter configs — each row = one segmentation run:"))
        add_cfg_btn = QPushButton("+ Add Config")
        add_cfg_btn.setFixedWidth(105)
        add_cfg_btn.setFixedHeight(22)
        add_cfg_btn.clicked.connect(self._add_config_row)
        header.addWidget(add_cfg_btn)
        seg_lay.addLayout(header)

        self._configs_widget = QWidget()
        self._configs_lay    = QVBoxLayout(self._configs_widget)
        self._configs_lay.setContentsMargins(0, 0, 0, 0)
        self._configs_lay.setSpacing(2)
        configs_scroll = QScrollArea()
        configs_scroll.setWidget(self._configs_widget)
        configs_scroll.setWidgetResizable(True)
        configs_scroll.setFixedHeight(120)
        seg_lay.addWidget(configs_scroll)

        # Save directory
        savedir_row = QHBoxLayout()
        savedir_row.addWidget(QLabel("Save dir:"))
        self._savedir_edit = QLineEdit()
        self._savedir_edit.setPlaceholderText("Directory for intermediate NPZ files")
        savedir_browse = QPushButton("Browse")
        savedir_browse.setFixedWidth(65)
        savedir_browse.clicked.connect(self._browse_savedir)
        savedir_row.addWidget(self._savedir_edit)
        savedir_row.addWidget(savedir_browse)
        seg_lay.addLayout(savedir_row)

        self._run_seg_btn = QPushButton("Run Segmentation → Save NPZs")
        self._run_seg_btn.clicked.connect(self._on_run_seg)
        seg_lay.addWidget(self._run_seg_btn)

        root.addWidget(seg_box)

        # ── Intermediates ──
        inter_box = QGroupBox("Intermediate NPZ files")
        inter_lay = QVBoxLayout(inter_box)

        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("Dir:"))
        self._scan_dir_edit = QLineEdit()
        self._scan_dir_edit.setPlaceholderText("Directory to scan for NPZ files")
        scan_browse = QPushButton("Browse")
        scan_browse.setFixedWidth(65)
        scan_browse.clicked.connect(self._browse_scandir)
        scan_btn = QPushButton("Scan")
        scan_btn.setFixedWidth(50)
        scan_btn.clicked.connect(self._scan_intermediates)
        scan_row.addWidget(self._scan_dir_edit)
        scan_row.addWidget(scan_browse)
        scan_row.addWidget(scan_btn)
        inter_lay.addLayout(scan_row)

        self._inter_list = QListWidget()
        self._inter_list.setFixedHeight(100)
        self._inter_list.setSelectionMode(QListWidget.MultiSelection)
        inter_lay.addWidget(self._inter_list)
        inter_lay.addWidget(QLabel("Ctrl+click to select/deselect; all selected files are used in consensus."))

        root.addWidget(inter_box)

        # ── Phase 2: Consensus ──
        con_box  = QGroupBox("Phase 2 — Consensus")
        con_form = QFormLayout(con_box)

        self._spatial_votes_spin = QSpinBox()
        self._spatial_votes_spin.setRange(1, 50)
        self._spatial_votes_spin.setValue(2)
        self._spatial_iou_spin = _dspin(0.01, 1.0, 0.30, 2, 0.05)
        con_form.addRow("Spatial min votes (z × configs):", self._spatial_votes_spin)
        con_form.addRow("Spatial IoU threshold:",           self._spatial_iou_spin)

        self._temporal_chk = QCheckBox("Enable temporal consistency filter")
        self._temporal_chk.setChecked(False)
        self._temporal_chk.toggled.connect(self._on_temporal_toggled)
        con_form.addRow(self._temporal_chk)

        self._temporal_widget = QWidget()
        t_form = QFormLayout(self._temporal_widget)
        t_form.setContentsMargins(16, 0, 0, 0)
        t_form.setSpacing(4)
        self._temporal_window_spin = QSpinBox()
        self._temporal_window_spin.setRange(1, 50)
        self._temporal_window_spin.setValue(2)
        self._temporal_votes_spin = QSpinBox()
        self._temporal_votes_spin.setRange(1, 100)
        self._temporal_votes_spin.setValue(2)
        self._temporal_iou_spin = _dspin(0.01, 1.0, 0.30, 2, 0.05)
        t_form.addRow("Window (±frames):",    self._temporal_window_spin)
        t_form.addRow("Min matching frames:", self._temporal_votes_spin)
        t_form.addRow("IoU threshold:",       self._temporal_iou_spin)
        self._temporal_widget.setVisible(False)
        con_form.addRow(self._temporal_widget)

        self._run_con_btn = QPushButton("Run Consensus")
        self._run_con_btn.clicked.connect(self._on_run_con)
        con_form.addRow(self._run_con_btn)

        root.addWidget(con_box)

        # ── Cancel / Progress / Log ──
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        root.addWidget(self._cancel_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(130)
        self._log.setPlaceholderText("Log…")
        root.addWidget(self._log)

        root.addStretch()

        # Default: two configs with slightly different parameters
        self._add_config_row()
        self._add_config_row()
        if len(self._config_rows) >= 2:
            self._config_rows[1]._flow.setValue(0.0)
            self._config_rows[1]._prob.setValue(-1.0)

    # ── Config rows ─────────────────────────────────────────────────────

    def _add_config_row(self):
        idx = len(self._config_rows)
        row = _ConfigRow(idx, on_delete=lambda: self._remove_config_row(row))
        self._config_rows.append(row)
        self._configs_lay.addWidget(row)

    def _remove_config_row(self, row: _ConfigRow):
        if len(self._config_rows) <= 1:
            return
        self._config_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        for i, r in enumerate(self._config_rows):
            r.set_index(i)

    # ── Mode / model ─────────────────────────────────────────────────────

    def _on_mode_changed(self, _=None):
        self._secondary_row.setVisible(self._rb_two_ch.isChecked())

    def _on_model_changed(self, text: str):
        self._custom_row.setVisible(text == "custom")

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select custom Cellpose model", "",
            "Model files (*.pt *.pth);;All files (*)"
        )
        if path:
            self._custom_model_path = path
            self._custom_path_edit.setText(path)

    def _on_temporal_toggled(self, checked: bool):
        self._temporal_widget.setVisible(checked)

    # ── Load / Clear ─────────────────────────────────────────────────────

    def _on_load(self):
        active = self.viewer.layers.selection.active
        if active is None or not isinstance(active, napari.layers.Image):
            self._input_status.setText("Select an Image layer first")
            return
        data = np.asarray(active.data)
        if data.ndim != 4:
            self._input_status.setText(f"Expected 4D (T,Z,H,W), got {data.shape}")
            return
        self._stack_data  = data
        self._input_scale = tuple(active.scale)
        T, Z, H, W = data.shape
        self._input_status.setText(f"T={T}, Z={Z}, H={H}, W={W}")

    def _on_clear(self):
        self._stack_data = None
        self._input_status.setText("Not loaded")

    def _on_load_secondary(self):
        active = self.viewer.layers.selection.active
        if active is None or not isinstance(active, napari.layers.Image):
            self._secondary_status.setText("Select an Image layer first")
            return
        data = np.asarray(active.data)
        if data.ndim != 4:
            self._secondary_status.setText(f"Expected 4D (T,Z,H,W), got {data.shape}")
            return
        if self._stack_data is not None and data.shape != self._stack_data.shape:
            self._secondary_status.setText(
                f"Shape mismatch: {data.shape} vs primary {self._stack_data.shape}"
            )
            return
        self._secondary_data = data
        T, Z, H, W = data.shape
        self._secondary_status.setText(f"T={T}, Z={Z}, H={H}, W={W}")

    def _on_clear_secondary(self):
        self._secondary_data = None
        self._secondary_status.setText("Not loaded")

    # ── Save / scan directory ─────────────────────────────────────────────

    def _browse_savedir(self):
        d = QFileDialog.getExistingDirectory(self, "Select save directory")
        if d:
            self._savedir_edit.setText(d)
            if not self._scan_dir_edit.text():
                self._scan_dir_edit.setText(d)

    def _browse_scandir(self):
        d = QFileDialog.getExistingDirectory(self, "Select directory to scan")
        if d:
            self._scan_dir_edit.setText(d)

    def _scan_intermediates(self):
        scan_dir = self._scan_dir_edit.text().strip()
        if not scan_dir or not os.path.isdir(scan_dir):
            self._log_append("ERROR: Set a valid directory to scan.")
            return

        self._inter_list.clear()
        npz_files = sorted(
            f for f in os.listdir(scan_dir) if f.endswith(".npz")
        )
        if not npz_files:
            self._log_append(f"No NPZ files found in: {scan_dir}")
            return

        for fname in npz_files:
            path = os.path.join(scan_dir, fname)
            try:
                with np.load(path, allow_pickle=False) as d:
                    shape = tuple(d["masks"].shape)
                    meta  = json.loads(str(d["metadata_json"]))
                p = meta.get("cellpose_params", {})
                summary = (
                    f"{fname}  "
                    f"T={shape[0]}, Z={shape[1]}, H={shape[2]}, W={shape[3]}  |  "
                    f"flow={p.get('flow_threshold', '?'):.2f}, "
                    f"prob={p.get('cellprob_threshold', '?'):.1f}, "
                    f"minpx={p.get('min_size', '?')}, "
                    f"diam={'auto' if p.get('diameter') is None else p.get('diameter')}"
                )
            except Exception as e:
                summary = f"{fname}  [could not read metadata: {e}]"
            item = self._inter_list.addItem(summary)
            # Store path in item data
            self._inter_list.item(self._inter_list.count() - 1).setData(
                32, path  # Qt.UserRole = 32
            )

        # Auto-select all
        for i in range(self._inter_list.count()):
            self._inter_list.item(i).setSelected(True)

        self._log_append(f"Found {len(npz_files)} NPZ file(s) in {scan_dir}")

    # ── Model cache ───────────────────────────────────────────────────────

    def _ensure_model(self, model_type, custom_path, gpu):
        from napariTissueFlow.segtrack._pipeline import make_cp_model
        key = (model_type, custom_path, gpu)
        if self._cp_model is not None and self._cp_model_key == key:
            return self._cp_model
        if self._cp_model is not None:
            del self._cp_model
            self._cp_model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._cp_model     = make_cp_model(model_type, custom_model_path=custom_path, gpu=gpu)
        self._cp_model_key = key
        return self._cp_model

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_append(self, msg: str):
        self._log.append(str(msg))
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    # ── Shared worker state helpers ───────────────────────────────────────

    def _set_running(self, running: bool):
        self._is_running = running
        self._run_seg_btn.setEnabled(not running)
        self._run_con_btn.setEnabled(not running)
        self._cancel_btn.setVisible(running)
        self._progress.setVisible(running)

    # ── Phase 1: Run Segmentation ─────────────────────────────────────────

    def _on_run_seg(self):
        if self._is_running:
            self._log_append("Already processing.")
            return
        if self._stack_data is None:
            self._log_append("ERROR: Load a 4D image stack first.")
            return
        two_ch = self._rb_two_ch.isChecked()
        if two_ch and self._secondary_data is None:
            self._log_append("ERROR: Load a secondary channel stack first.")
            return

        save_dir = self._savedir_edit.text().strip()
        if not save_dir:
            self._log_append("ERROR: Set a save directory.")
            return
        os.makedirs(save_dir, exist_ok=True)

        model_type  = self._model_combo.currentText()
        gpu         = self._gpu_chk.isChecked()
        custom_path = self._custom_model_path if model_type == "custom" else None

        config_params = [row.collect() for row in self._config_rows]
        if not config_params:
            self._log_append("ERROR: Add at least one parameter config.")
            return

        stack_data     = self._stack_data
        secondary_data = self._secondary_data

        self._set_running(True)
        self._log.clear()

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_done_seg,
            "errored":  self._on_error,
        })
        def _work():
            from napariTissueFlow.segtrack._pipeline import run_cp, run_cp_two_channel

            yield f"Loading Cellpose model ({model_type}, GPU={gpu})…"
            model = self._ensure_model(model_type, custom_path, gpu)

            T, Z, H, W = stack_data.shape
            mode_str   = "two-channel" if two_ch else "single-channel"
            yield (
                f"Stack: T={T}, Z={Z}, H={H}, W={W}  [{mode_str}]  "
                f"→ {len(config_params)} config(s)"
            )

            saved_paths = []
            for ci, cp in enumerate(config_params):
                yield (
                    f"Config {ci + 1}/{len(config_params)}: "
                    f"flow={cp['flow_threshold']:.2f}, "
                    f"prob={cp['cellprob_threshold']:.1f}, "
                    f"minpx={cp['min_size']}, "
                    f"diam={'auto' if cp['diameter'] is None else cp['diameter']}"
                )
                cp_kwargs = dict(
                    model              = model,
                    diameter           = cp["diameter"],
                    flow_threshold     = cp["flow_threshold"],
                    cellprob_threshold = cp["cellprob_threshold"],
                    min_size           = cp["min_size"],
                )

                # Segment every (t, z) → collect into (T, Z, H, W)
                volume = np.zeros((T, Z, H, W), dtype=np.int32)
                for t in range(T):
                    for z in range(Z):
                        if two_ch:
                            masks = run_cp_two_channel(
                                stack_data[t, z], secondary_data[t, z], **cp_kwargs
                            )
                        else:
                            masks = run_cp(stack_data[t, z], **cp_kwargs)
                        volume[t, z] = masks.astype(np.int32)
                    n_avg = int(np.mean([
                        len(np.unique(volume[t, z][volume[t, z] > 0]))
                        for z in range(Z)
                    ]))
                    yield f"  T={t + 1}/{T}: ~{n_avg} cells/z-plane"

                # Save NPZ with metadata
                timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                fname     = f"config_{ci + 1:02d}_{timestamp}.npz"
                fpath     = os.path.join(save_dir, fname)
                metadata  = {
                    "axes":            "TZHW",
                    "created_utc":     datetime.datetime.utcnow().isoformat(),
                    "config_index":    ci,
                    "cellpose_params": {
                        **cp,
                        "model_type":        model_type,
                        "custom_model_path": custom_path,
                        "gpu":               gpu,
                    },
                    "shape":           list(volume.shape),
                    "two_channel":     two_ch,
                }
                np.savez_compressed(
                    fpath,
                    masks         = volume,
                    metadata_json = np.array(json.dumps(metadata)),
                )
                yield f"  Saved: {fname}  shape={volume.shape}"
                saved_paths.append(fpath)

            return save_dir, saved_paths

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_done_seg(self, result):
        self._set_running(False)
        save_dir, paths = result
        self._log_append(f"Segmentation complete — {len(paths)} file(s) saved to {save_dir}")
        # Auto-populate scan dir and scan
        self._scan_dir_edit.setText(save_dir)
        self._scan_intermediates()

    # ── Phase 2: Run Consensus ────────────────────────────────────────────

    def _on_run_con(self):
        if self._is_running:
            self._log_append("Already processing.")
            return

        selected = [
            self._inter_list.item(i)
            for i in range(self._inter_list.count())
            if self._inter_list.item(i).isSelected()
        ]
        if not selected:
            self._log_append("ERROR: Select at least one NPZ file in the intermediates list.")
            return

        npz_paths = [item.data(32) for item in selected]

        spatial_votes = self._spatial_votes_spin.value()
        spatial_iou   = self._spatial_iou_spin.value()
        use_temporal  = self._temporal_chk.isChecked()
        temp_window   = self._temporal_window_spin.value()
        temp_votes    = self._temporal_votes_spin.value()
        temp_iou      = self._temporal_iou_spin.value()
        input_scale   = self._input_scale

        self._set_running(True)
        self._log.clear()

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_done_con,
            "errored":  self._on_error,
        })
        def _work():
            from napariTissueFlow.segtrack._consensus import consensus_all, temporal_filter

            # Load NPZs
            stacks = []
            for path in npz_paths:
                with np.load(path, allow_pickle=False) as d:
                    masks = d["masks"].astype(np.int32)   # (T, Z, H, W)
                    meta  = json.loads(str(d["metadata_json"]))
                stacks.append(masks)
                p = meta.get("cellpose_params", {})
                yield (
                    f"Loaded {os.path.basename(path)}  "
                    f"shape={masks.shape}  "
                    f"flow={p.get('flow_threshold', '?'):.2f}, "
                    f"prob={p.get('cellprob_threshold', '?'):.1f}"
                )

            # Validate shapes
            shapes = [s.shape for s in stacks]
            T, Z = shapes[0][0], shapes[0][1]
            H, W = shapes[0][2], shapes[0][3]
            for i, sh in enumerate(shapes[1:], 1):
                if sh[0] != T or sh[2] != H or sh[3] != W:
                    raise ValueError(
                        f"Shape mismatch: file 0 = {shapes[0]}, file {i} = {sh}.  "
                        "T, H, W must match across all files."
                    )
                if sh[1] != Z:
                    yield (
                        f"WARNING: file {i} has Z={sh[1]} vs Z={Z} in file 0.  "
                        "Using all Z-planes from each file as-is."
                    )

            n_voters = sum(s.shape[1] for s in stacks)
            yield (
                f"Running spatial consensus: {len(stacks)} file(s) × Z-planes "
                f"= {n_voters} total voters per timepoint  "
                f"(min_votes={spatial_votes}, IoU>{spatial_iou:.2f})"
            )

            def seg_progress(t, T):
                if t % max(1, T // 10) == 0 or t == T:
                    pass  # yielding from nested function not supported; skip progress yields

            result = consensus_all(stacks, spatial_votes, spatial_iou)
            n_cells = int(np.max([result[t].max() for t in range(T)]))
            yield f"Spatial consensus done — max {n_cells} label(s) in any frame."

            if use_temporal:
                yield (
                    f"Running temporal filter: window=±{temp_window}, "
                    f"min_votes={temp_votes}, IoU>{temp_iou:.2f}…"
                )
                result = temporal_filter(result, temp_window, temp_votes, temp_iou)
                n_after = int(np.max([result[t].max() for t in range(T)]))
                yield f"Temporal filter done — max {n_after} label(s) remaining."

            return result, input_scale

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_done_con(self, result):
        self._set_running(False)
        output, scale = result

        name  = "Consensus Segmentation"
        layer = next(
            (ly for ly in self.viewer.layers
             if isinstance(ly, napari.layers.Labels) and ly.name == name),
            None,
        )
        if layer is None:
            kw = {} if scale is None else {"scale": scale[-3:]}
            layer = self.viewer.add_labels(
                np.zeros(output.shape, dtype=np.int32), name=name, **kw
            )
        layer.data = output
        layer.refresh()
        self._log_append(f"Output layer '{name}': shape={output.shape}")

    # ── Shared callbacks ──────────────────────────────────────────────────

    def _on_error(self, exc):
        self._set_running(False)
        self._log_append(f"ERROR: {exc}")

    def _on_aborted(self):
        self._set_running(False)
        self._log_append("Cancelled.")

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.quit()
