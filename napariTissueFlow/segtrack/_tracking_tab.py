"""
Tracking tab for napariSegTrack.

Input  : a Labels layer (nuclear masks, possibly from the Segmentation tab)
         OR an Image layer (will be segmented with Cellpose first)
Output : tracked Labels layer written back to the input layer (Labels input)
         or added as a new layer (Image input)

Tracking is performed with LapTrack (centroid-distance LAP with gap closing).
Optional temporal correction detects and merges false nuclear splits.
"""

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar, QScrollArea,
)
from napari.qt.threading import thread_worker
import napari


# ── helpers ────────────────────────────────────────────────────────────

def _dspin(lo, hi, default, decimals=2, step=0.1):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(default)
    return s


def _sep(title):
    lbl = QLabel(f"<b>{title}</b>")
    lbl.setStyleSheet("color: palette(mid); margin-top: 4px;")
    return lbl


# ── defaults ───────────────────────────────────────────────────────────

TRACK_DEFAULTS = {
    # cellpose (only used when nuclear input is an Image layer)
    "model_type":               "cyto3",
    "diameter":                  30.0,
    "auto_diameter":             False,
    "flow_threshold":            0.4,
    "cellprob_threshold":        0.0,
    "min_size":                  15,
    "gpu":                       True,
    # voronoi expansion (used internally by temporal correction only)
    "max_expand":                30,
    # laptrack
    "max_link_dist":             20,
    "max_gap_dist":              25,
    "gap_closing_max_frame_count": 3,
    # temporal correction
    "skip_temporal":             False,
    "division_confirm_frames":   3,
    "division_sep_min_px":       5.0,
    "max_split_radius_factor":   2,
}


# ── widget ─────────────────────────────────────────────────────────────

class TrackingTab(QWidget):
    """Tracking tab: LapTrack-based cell tracking from a Labels or Image layer."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer  = viewer
        self._worker = None
        self._custom_model_path = None

        self._setup_ui()

        viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        viewer.layers.events.removed.connect(self._refresh_layer_combos)

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Nuclear input ──
        nuc_box = QGroupBox("Nuclear Input")
        nuc_lay = QFormLayout(nuc_box)
        self._nuc_combo = QComboBox()
        self._nuc_combo.setToolTip(
            "Labels layer (already segmented) or Image layer (will be segmented first)."
        )
        self._nuc_combo.currentTextChanged.connect(self._on_nuc_layer_changed)
        nuc_lay.addRow("Layer:", self._nuc_combo)
        root.addWidget(nuc_box)

        # ── Cellpose params (shown only when Image layer selected) ──
        self._cp_box = QGroupBox("Segmentation Parameters (Image layer input)")
        self._cp_box.setCheckable(True)
        self._cp_box.setChecked(False)
        cp_inner = QWidget()
        self._cp_form = QFormLayout(cp_inner)
        self._cp_form.setSpacing(4)
        self._build_cp_params()
        cp_scroll = QScrollArea()
        cp_scroll.setWidget(cp_inner)
        cp_scroll.setWidgetResizable(True)
        cp_scroll.setFixedHeight(200)
        cp_vlay = QVBoxLayout(self._cp_box)
        cp_vlay.addWidget(cp_scroll)
        self._cp_box.toggled.connect(cp_scroll.setVisible)
        cp_scroll.setVisible(False)
        self._cp_box.setVisible(False)   # hidden when Labels layer selected
        root.addWidget(self._cp_box)

        # ── Tracking parameters ──
        track_box = QGroupBox("LapTrack Parameters")
        track_box.setCheckable(True)
        track_box.setChecked(False)
        track_inner = QWidget()
        self._track_form = QFormLayout(track_inner)
        self._track_form.setSpacing(4)
        self._build_track_params()
        t_scroll = QScrollArea()
        t_scroll.setWidget(track_inner)
        t_scroll.setWidgetResizable(True)
        t_scroll.setFixedHeight(200)
        t_vlay = QVBoxLayout(track_box)
        t_vlay.addWidget(t_scroll)
        track_box.toggled.connect(t_scroll.setVisible)
        t_scroll.setVisible(False)
        root.addWidget(track_box)

        # ── Save output ──
        save_box = QGroupBox("Save Output")
        save_lay = QVBoxLayout(save_box)
        self._chk_save = QCheckBox("Save results to disk")
        self._chk_save.toggled.connect(self._on_save_toggled)
        save_lay.addWidget(self._chk_save)
        out_row = QHBoxLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Output directory…")
        self._out_dir.setEnabled(False)
        self._btn_browse = QPushButton("Browse")
        self._btn_browse.setEnabled(False)
        self._btn_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self._out_dir)
        out_row.addWidget(self._btn_browse)
        save_lay.addLayout(out_row)
        root.addWidget(save_box)

        # ── Run button ──
        self._run_btn = QPushButton("Run Tracking")
        self._run_btn.clicked.connect(self._on_run)
        root.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(140)
        self._log.setPlaceholderText("Tracking log…")
        root.addWidget(self._log)

        root.addStretch()
        self._refresh_layer_combos()

    def _build_cp_params(self):
        add = self._cp_form.addRow

        self._cp_model_combo = QComboBox()
        self._cp_model_combo.addItems(["cyto3", "nuclei", "cpsam", "custom"])
        self._cp_model_combo.setCurrentText(TRACK_DEFAULTS["model_type"])
        self._cp_model_combo.currentTextChanged.connect(self._on_cp_model_changed)
        add("Model:", self._cp_model_combo)

        custom_row = QHBoxLayout()
        self._cp_custom_edit = QLineEdit()
        self._cp_custom_edit.setPlaceholderText("path/to/model.pt")
        self._cp_custom_edit.setEnabled(False)
        self._cp_custom_btn = QPushButton("Browse")
        self._cp_custom_btn.setEnabled(False)
        self._cp_custom_btn.clicked.connect(self._browse_custom_model)
        custom_row.addWidget(self._cp_custom_edit)
        custom_row.addWidget(self._cp_custom_btn)
        add("Custom model:", custom_row)

        diam_row = QHBoxLayout()
        self._cp_diam = _dspin(1.0, 2000.0, TRACK_DEFAULTS["diameter"], 1, 1.0)
        self._cp_auto_diam = QCheckBox("Auto")
        self._cp_auto_diam.setChecked(TRACK_DEFAULTS["auto_diameter"])
        self._cp_auto_diam.toggled.connect(lambda c: self._cp_diam.setEnabled(not c))
        diam_row.addWidget(self._cp_diam)
        diam_row.addWidget(self._cp_auto_diam)
        add("Diameter (px):", diam_row)

        self._cp_flow  = _dspin(-10, 10, TRACK_DEFAULTS["flow_threshold"],     2, 0.05)
        self._cp_prob  = _dspin(-10, 10, TRACK_DEFAULTS["cellprob_threshold"], 1, 0.5)
        self._cp_minsz = QSpinBox()
        self._cp_minsz.setRange(1, 10000)
        self._cp_minsz.setValue(TRACK_DEFAULTS["min_size"])
        add("Flow threshold:",     self._cp_flow)
        add("CellProb threshold:", self._cp_prob)
        add("Min size (px):",      self._cp_minsz)

        self._cp_gpu = QCheckBox()
        self._cp_gpu.setChecked(TRACK_DEFAULTS["gpu"])
        add("GPU:", self._cp_gpu)

    def _build_track_params(self):
        add = self._track_form.addRow

        add(_sep("LapTrack"))
        self._p_link = QSpinBox()
        self._p_link.setRange(1, 500)
        self._p_link.setValue(TRACK_DEFAULTS["max_link_dist"])
        add("Max link dist (px):", self._p_link)

        self._p_gap = QSpinBox()
        self._p_gap.setRange(1, 500)
        self._p_gap.setValue(TRACK_DEFAULTS["max_gap_dist"])
        add("Max gap dist (px):", self._p_gap)

        self._p_gapf = QSpinBox()
        self._p_gapf.setRange(1, 20)
        self._p_gapf.setValue(TRACK_DEFAULTS["gap_closing_max_frame_count"])
        add("Gap closing frames:", self._p_gapf)

        add(_sep("Temporal Correction"))
        self._p_notemporal = QCheckBox("Skip temporal correction")
        self._p_notemporal.setChecked(TRACK_DEFAULTS["skip_temporal"])
        add("", self._p_notemporal)

        self._p_divconf = QSpinBox()
        self._p_divconf.setRange(1, 20)
        self._p_divconf.setValue(TRACK_DEFAULTS["division_confirm_frames"])
        add("Confirm frames:", self._p_divconf)

        self._p_divsep = _dspin(0, 100, TRACK_DEFAULTS["division_sep_min_px"], 1, 1.0)
        add("Min sep (px):", self._p_divsep)

        self._p_splitr = QSpinBox()
        self._p_splitr.setRange(1, 20)
        self._p_splitr.setValue(TRACK_DEFAULTS["max_split_radius_factor"])
        add("Split radius factor:", self._p_splitr)

    # ── Layer combo refresh ────────────────────────────────────────────

    def _refresh_layer_combos(self, *_):
        all_names = [""] + [lay.name for lay in self.viewer.layers
                            if isinstance(lay, (napari.layers.Image, napari.layers.Labels))]

        prev = self._nuc_combo.currentText()
        self._nuc_combo.blockSignals(True)
        self._nuc_combo.clear()
        self._nuc_combo.addItems(all_names)
        idx = self._nuc_combo.findText(prev)
        self._nuc_combo.setCurrentIndex(max(idx, 0))
        self._nuc_combo.blockSignals(False)

    def _on_nuc_layer_changed(self, name):
        is_image = False
        if name:
            try:
                layer = self.viewer.layers[name]
            except KeyError:
                layer = None
            is_image = layer is not None and isinstance(layer, napari.layers.Image)
        self._cp_box.setVisible(is_image)

    def _on_cp_model_changed(self, text):
        is_custom = (text == "custom")
        self._cp_custom_edit.setEnabled(is_custom)
        self._cp_custom_btn.setEnabled(is_custom)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select custom Cellpose model", "", "Model files (*.pt *.pth);;All files (*)"
        )
        if path:
            self._custom_model_path = path
            self._cp_custom_edit.setText(path)

    def _on_save_toggled(self, checked):
        self._out_dir.setEnabled(checked)
        self._btn_browse.setEnabled(checked)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._out_dir.setText(d)

    # ── Parameter collection ───────────────────────────────────────────

    def _collect_cp_params(self):
        return {
            "model_type":         self._cp_model_combo.currentText(),
            "custom_model_path":  self._custom_model_path,
            "diameter":           None if self._cp_auto_diam.isChecked()
                                       else self._cp_diam.value(),
            "flow_threshold":     self._cp_flow.value(),
            "cellprob_threshold": self._cp_prob.value(),
            "min_size":           self._cp_minsz.value(),
            "gpu":                self._cp_gpu.isChecked(),
        }

    def _collect_track_params(self):
        return {
            "max_link_dist":               self._p_link.value(),
            "max_gap_dist":                self._p_gap.value(),
            "gap_closing_max_frame_count": self._p_gapf.value(),
            "skip_temporal":               self._p_notemporal.isChecked(),
            "division_confirm_frames":     self._p_divconf.value(),
            "division_sep_min_px":         self._p_divsep.value(),
            "max_split_radius_factor":     self._p_splitr.value(),
        }

    # ── Run ────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._worker is not None and self._worker.is_running:
            self._log_append("Tracking already running.")
            return

        nuc_name = self._nuc_combo.currentText()

        if not nuc_name:
            self._log_append("ERROR: Select a nuclear input layer.")
            return

        try:
            nuc_layer = self.viewer.layers[nuc_name]
        except KeyError:
            nuc_layer = None
        if nuc_layer is None:
            self._log_append(f"ERROR: Layer '{nuc_name}' not found.")
            return

        is_image     = isinstance(nuc_layer, napari.layers.Image)
        nuc_data     = np.asarray(nuc_layer.data)

        cp_params    = self._collect_cp_params() if is_image else None
        track_params = self._collect_track_params()
        save_output  = self._chk_save.isChecked()
        out_dir      = self._out_dir.text().strip() or None

        self._log.clear()
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_done,
            "errored":  self._on_error,
        })
        def _work():
            from napariTissueFlow.segtrack._pipeline import (
                make_cp_model, run_cp,
                track_nuclei_laptrack,
                correct_false_splits,
            )

            # ── Prepare nuclear frames ──
            if nuc_data.ndim == 2:
                nuc_frames = [nuc_data]
            else:
                nuc_frames = [nuc_data[t] for t in range(nuc_data.shape[0])]

            # ── Segment if Image layer ──
            if is_image:
                yield "Segmenting nuclear channel with Cellpose…"
                model = make_cp_model(
                    cp_params["model_type"],
                    custom_model_path=cp_params.get("custom_model_path"),
                    gpu=cp_params["gpu"],
                )
                nuc_raw = []
                for i, frame in enumerate(nuc_frames):
                    masks = run_cp(
                        frame, model,
                        diameter          = cp_params["diameter"],
                        flow_threshold    = cp_params["flow_threshold"],
                        cellprob_threshold= cp_params["cellprob_threshold"],
                        min_size          = cp_params["min_size"],
                    )
                    nuc_raw.append(masks)
                    yield f"  Frame {i}: {len(np.unique(masks[masks>0]))} nucleus/nuclei"
            else:
                nuc_raw = [f.astype(np.int32) for f in nuc_frames]

            # ── Track ──
            yield "Running LapTrack…"
            tracked_nuc, track_df = track_nuclei_laptrack(
                nuc_raw,
                max_link_dist              = track_params["max_link_dist"],
                max_gap_dist               = track_params["max_gap_dist"],
                gap_closing_max_frame_count= track_params["gap_closing_max_frame_count"],
            )
            n_tracks = track_df["track_id"].nunique() if len(track_df) > 0 else 0
            yield f"  {n_tracks} track(s) found"

            # ── Temporal correction (requires raw nuclear images for re-segmentation) ──
            corr_nuc = None
            if not track_params["skip_temporal"]:
                if not is_image:
                    yield ("Skipping temporal correction: raw nuclear images needed "
                           "but input is a Labels layer.")
                else:
                    yield "Detecting and correcting false splits…"
                    tc_params = {
                        "diameter":                cp_params["diameter"],
                        "flow_threshold":          cp_params["flow_threshold"],
                        "cellprob_threshold":      cp_params["cellprob_threshold"],
                        "min_size":                cp_params["min_size"],
                        "max_expand":              TRACK_DEFAULTS["max_expand"],
                        "max_split_radius_factor": track_params["max_split_radius_factor"],
                        "division_confirm_frames": track_params["division_confirm_frames"],
                        "division_sep_min_px":     track_params["division_sep_min_px"],
                    }
                    log_buf = []
                    corr_nuc = correct_false_splits(
                        tracked_nuc, nuc_frames, model, tc_params, log=log_buf.append
                    )
                    for m in log_buf:
                        yield m

            # ── Optionally save ──
            if save_output and out_dir:
                import os
                from tifffile import imwrite
                os.makedirs(out_dir, exist_ok=True)
                def _stack(lst):
                    return np.stack(lst, axis=0).astype(np.uint16)
                imwrite(os.path.join(out_dir, "nuclei_tracked.tif"), _stack(tracked_nuc))
                if corr_nuc:
                    imwrite(os.path.join(out_dir, "nuclei_corrected.tif"), _stack(corr_nuc))
                yield f"Saved to {out_dir}"

            yield "Tracking complete!"
            return {
                "tracked_nuc": tracked_nuc,
                "corr_nuc":    corr_nuc,
                "nuc_name":    nuc_name,
                "is_image":    is_image,
            }

        self._worker = _work()

    def _on_done(self, result):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)

        def _stack(lst):
            return np.stack(lst, axis=0).astype(np.uint16)

        nuc_name = result["nuc_name"]
        best_nuc = result["corr_nuc"] if result["corr_nuc"] else result["tracked_nuc"]
        stacked  = _stack(best_nuc)

        if not result["is_image"]:
            try:
                self.viewer.layers[nuc_name].data = stacked
                self._log_append("Done! Segmentation layer updated with tracked result.")
                return
            except KeyError:
                pass

        self.viewer.add_labels(stacked, name=f"{nuc_name}_tracked_nuclei")
        self._log_append("Done! Results added as napari layer.")

    def _on_error(self, exc):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log_append(f"ERROR: {exc}")
        import traceback
        self._log_append(traceback.format_exc())

    def _log_append(self, msg):
        self._log.append(str(msg))
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
