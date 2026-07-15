# Atom Extraction Widget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build stage ① of the atom-based candidate pipeline — a reusable atom-extraction core module plus an interactive napari "Atom Extraction" widget that tunes five params against a live atom preview and writes `atoms.tif` for DB-Gen (②) to consume.

**Architecture:** A pure, deterministic core module (`tracking_ultrack/atoms.py`) computes a local-mean-subtracted residual of the cellpose foreground/contour maps, derives territory + ridge, and watersheds territory into atoms. A napari widget (controls class + behavior mixin, mirroring the existing DB-browser pattern) previews atoms one frame at a time and, on an explicit "Compute atoms (full stack)" action, writes `2_nucleus/atoms.tif` with the params fingerprinted into the file. This plan is **purely additive**: it does not remove the existing higra DB-generation params or widget — that removal happens in spec ② when the consumer is swapped — so the application keeps working after ①.

**Tech Stack:** Python, numpy, scipy.ndimage, scikit-image (`threshold_local`, `watershed`), tifffile, Qt (qtpy), napari, pydantic (`TrackingConfig`), pytest. Conda env `cellflow`.

**Spec:** `docs/superpowers/specs/2026-05-31-atom-extraction-widget-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/cellflow/tracking_ultrack/atoms.py` (create) | Core: `residual`, `extract_atoms_frame`, `extract_atoms_stack`, `AtomParams`, `params_fingerprint`, `write_atoms_tif`, `read_atoms_params` |
| `tests/tracking_ultrack/test_atoms.py` (create) | Unit tests for the core |
| `src/cellflow/tracking_ultrack/config.py` (modify) | Add 5 atom params to `TrackingConfig` |
| `tests/tracking_ultrack/test_config.py` (modify) | Defaults test for new params |
| `src/cellflow/napari/_paths.py` (modify) | Add `nucleus_atoms` path property |
| `tests/napari/test_paths.py` (create) | Path test |
| `src/cellflow/napari/nucleus_atom_extraction_widget.py` (create) | `NucleusAtomExtractionWidget` controls + `NucleusAtomExtractionMixin` behavior |
| `tests/napari/test_nucleus_atom_extraction_widget.py` (create) | Widget controls + mixin behavior tests |
| `src/cellflow/napari/nucleus_workflow_widget.py` (modify) | Inherit mixin, build/alias/init the section, add to layout |
| `src/cellflow/napari/_state.py` (modify) | Persist `atom_extraction` block |

**Test command prefix (all tasks):** `conda run --no-capture-output -n cellflow pytest`

---

## Task 1: Core — `residual`

**Files:**
- Create: `src/cellflow/tracking_ultrack/atoms.py`
- Test: `tests/tracking_ultrack/test_atoms.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking_ultrack/test_atoms.py
from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.atoms import residual


def test_residual_is_zero_on_flat_input():
    frame = np.full((40, 40), 0.5, dtype=np.float32)
    out = residual(frame, window=11)
    assert out.shape == frame.shape
    assert out.dtype == np.float32
    assert np.allclose(out, 0.0, atol=1e-5)


def test_residual_is_nonnegative_and_peaks_on_local_bump():
    frame = np.zeros((40, 40), dtype=np.float32)
    frame[18:22, 18:22] = 1.0  # a bright patch on flat background
    out = residual(frame, window=11)
    assert out.min() >= 0.0
    assert out[20, 20] > 0.0


def test_residual_forces_odd_window():
    frame = np.random.default_rng(0).random((30, 30)).astype(np.float32)
    # even window must not raise and must equal the next odd window result
    assert np.allclose(residual(frame, window=10), residual(frame, window=11))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellflow.tracking_ultrack.atoms'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cellflow/tracking_ultrack/atoms.py
"""Atom extraction: residual-conditioned foreground split by contour ridges.

Stage ① of the atom-based candidate pipeline. Pure, deterministic functions
shared by the interactive preview and the full-stack ``atoms.tif`` writer.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_local


def residual(frame: np.ndarray, window: int) -> np.ndarray:
    """Local-mean-subtracted residual: ``clip(frame - localmean(frame), 0)``.

    Flattens each map's per-nucleus offset so a single global threshold works
    everywhere while staying ~0 in flat background. ``window`` is forced odd.
    """
    window = int(window) | 1
    frame = np.asarray(frame, dtype=np.float32)
    local_mean = threshold_local(frame, block_size=window, method="gaussian")
    return np.clip(frame - local_mean, 0.0, None).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/atoms.py tests/tracking_ultrack/test_atoms.py
git commit -m "feat(atoms): residual (local-mean-subtracted) map conditioning"
```

---

## Task 2: Core — `extract_atoms_frame`

**Files:**
- Modify: `src/cellflow/tracking_ultrack/atoms.py`
- Test: `tests/tracking_ultrack/test_atoms.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking_ultrack/test_atoms.py
from cellflow.tracking_ultrack.atoms import extract_atoms_frame


def _two_blob_frame():
    # territory = two square nuclei; residual_contour = a ridge between them.
    territory = np.zeros((40, 80), dtype=bool)
    territory[10:30, 8:36] = True   # left nucleus
    territory[10:30, 44:72] = True  # right nucleus
    residual_contour = np.zeros((40, 80), dtype=np.float32)
    residual_contour[10:30, 35:37] = 0.0   # (separated already by background)
    return residual_contour, territory


def test_extract_atoms_frame_labels_each_territory_island():
    rc, territory = _two_blob_frame()
    atoms = extract_atoms_frame(rc, territory, contour_floor=0.05, atom_min_area=0)
    # two disconnected islands -> exactly two atoms, background stays 0
    assert atoms[territory].min() >= 1
    assert atoms[~territory].max() == 0
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 2


def test_extract_atoms_frame_splits_one_island_on_ridge():
    territory = np.zeros((40, 60), dtype=bool)
    territory[10:30, 8:52] = True  # one connected island
    rc = np.zeros((40, 60), dtype=np.float32)
    rc[10:30, 29:31] = 1.0         # a strong ridge down the middle
    atoms = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=0)
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 2


def test_extract_atoms_frame_merges_small_atoms_and_leaves_no_holes():
    territory = np.zeros((40, 60), dtype=bool)
    territory[10:30, 8:52] = True
    rc = np.zeros((40, 60), dtype=np.float32)
    rc[10:30, 12:14] = 1.0  # ridge that carves off a tiny sliver (cols 8-12)
    atoms = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=200)
    # tiny sliver merged away -> one atom, and every territory pixel is labelled
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 1
    assert np.all(atoms[territory] > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -k extract_atoms_frame -v`
Expected: FAIL — `ImportError: cannot import name 'extract_atoms_frame'`

- [ ] **Step 3: Write minimal implementation**

```python
# add imports at top of src/cellflow/tracking_ultrack/atoms.py
from scipy import ndimage as ndi
from skimage.segmentation import watershed


# add function
def extract_atoms_frame(
    residual_contour: np.ndarray,
    territory: np.ndarray,
    contour_floor: float,
    atom_min_area: int,
) -> np.ndarray:
    """Split ``territory`` into atoms along the cleaned contour ridge.

    ``ridge`` is where the residual contour exceeds ``contour_floor`` (a noise
    cutoff). Cores (ridge-free territory) seed a watershed that floods the
    residual-contour elevation, so a broken faint ridge still meets at the crest.
    Atoms smaller than ``atom_min_area`` are merged into a neighbour by dropping
    their markers and re-flooding, leaving no holes in the territory.
    """
    residual_contour = np.asarray(residual_contour, dtype=np.float32)
    territory = np.asarray(territory, dtype=bool)
    ridge = residual_contour > contour_floor
    cores = territory & ~ridge
    markers, _ = ndi.label(cores)
    atoms = watershed(residual_contour, markers=markers, mask=territory)
    if atom_min_area > 0:
        ids, counts = np.unique(atoms, return_counts=True)
        small = set(ids[(counts < atom_min_area) & (ids != 0)].tolist())
        if small:
            keep_markers = np.where(np.isin(atoms, list(small)), 0, atoms)
            atoms = watershed(residual_contour, markers=keep_markers, mask=territory)
    return atoms.astype(np.int32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -k extract_atoms_frame -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/atoms.py tests/tracking_ultrack/test_atoms.py
git commit -m "feat(atoms): extract_atoms_frame (watershed split + small-atom merge)"
```

---

## Task 3: Core — `AtomParams` + `extract_atoms_stack`

**Files:**
- Modify: `src/cellflow/tracking_ultrack/atoms.py`
- Test: `tests/tracking_ultrack/test_atoms.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking_ultrack/test_atoms.py
from cellflow.tracking_ultrack.atoms import AtomParams, extract_atoms_stack


def test_atom_params_defaults_match_spec():
    p = AtomParams()
    assert p.fg_window == 51
    assert p.fg_cutoff == 0.002
    assert p.contour_window == 51
    assert p.contour_floor == 0.01
    assert p.atom_min_area == 100


def test_extract_atoms_stack_shape_and_determinism():
    rng = np.random.default_rng(0)
    fg = rng.random((3, 40, 40)).astype(np.float32)
    contour = rng.random((3, 40, 40)).astype(np.float32)
    params = AtomParams(fg_window=11, fg_cutoff=0.01, contour_window=11,
                        contour_floor=0.05, atom_min_area=0)
    a1 = extract_atoms_stack(fg, contour, params)
    a2 = extract_atoms_stack(fg, contour, params)
    assert a1.shape == (3, 40, 40)
    assert a1.dtype == np.int32
    assert np.array_equal(a1, a2)  # deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -k "atom_params or extract_atoms_stack" -v`
Expected: FAIL — `ImportError: cannot import name 'AtomParams'`

- [ ] **Step 3: Write minimal implementation**

```python
# add import at top of src/cellflow/tracking_ultrack/atoms.py
from dataclasses import dataclass


# add after imports
@dataclass(frozen=True)
class AtomParams:
    """The five knobs that fully determine an atom segmentation."""
    fg_window: int = 51
    fg_cutoff: float = 0.002
    contour_window: int = 51
    contour_floor: float = 0.01
    atom_min_area: int = 100


# add function
def extract_atoms_stack(
    fg: np.ndarray, contour: np.ndarray, params: AtomParams
) -> np.ndarray:
    """Atom label stack for a (T, Y, X) foreground + contour pair.

    Per frame: residual the fg map and threshold it into territory, residual the
    contour map into the watershed elevation, then ``extract_atoms_frame``.
    Returns the atom labels only — the residual contour is internal.
    """
    fg = np.asarray(fg, dtype=np.float32)
    contour = np.asarray(contour, dtype=np.float32)
    out = np.zeros(fg.shape, dtype=np.int32)
    for t in range(fg.shape[0]):
        territory = residual(fg[t], params.fg_window) > params.fg_cutoff
        residual_contour = residual(contour[t], params.contour_window)
        out[t] = extract_atoms_frame(
            residual_contour, territory, params.contour_floor, params.atom_min_area
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -v`
Expected: PASS (all atoms tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/atoms.py tests/tracking_ultrack/test_atoms.py
git commit -m "feat(atoms): AtomParams + extract_atoms_stack (full-stack labels)"
```

---

## Task 4: Core — fingerprint + `atoms.tif` I/O

**Files:**
- Modify: `src/cellflow/tracking_ultrack/atoms.py`
- Test: `tests/tracking_ultrack/test_atoms.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking_ultrack/test_atoms.py
from cellflow.tracking_ultrack.atoms import (
    params_fingerprint,
    write_atoms_tif,
    read_atoms_params,
)


def test_params_fingerprint_is_stable_and_param_sensitive():
    a = AtomParams()
    assert params_fingerprint(a) == params_fingerprint(AtomParams())
    assert params_fingerprint(a) != params_fingerprint(AtomParams(fg_cutoff=0.01))


def test_write_then_read_atoms_tif_round_trips_labels_and_params(tmp_path):
    import tifffile

    atoms = np.zeros((2, 16, 16), dtype=np.int32)
    atoms[0, 2:6, 2:6] = 1
    atoms[1, 8:12, 8:12] = 7
    params = AtomParams(fg_window=31, fg_cutoff=0.005)

    path = tmp_path / "atoms.tif"
    write_atoms_tif(path, atoms, params)

    assert np.array_equal(tifffile.imread(path), atoms)
    stored_params, stored_fp = read_atoms_params(path)
    assert stored_params["fg_window"] == 31
    assert stored_params["fg_cutoff"] == 0.005
    assert stored_fp == params_fingerprint(params)


def test_read_atoms_params_returns_none_when_absent(tmp_path):
    import tifffile

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.int32))
    params, fp = read_atoms_params(path)
    assert params is None and fp is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -k "fingerprint or atoms_tif or atoms_params" -v`
Expected: FAIL — `ImportError: cannot import name 'params_fingerprint'`

- [ ] **Step 3: Write minimal implementation**

```python
# add imports at top of src/cellflow/tracking_ultrack/atoms.py
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import tifffile

_PARAMS_KEY = "cellflow_atom_params"
_FINGERPRINT_KEY = "cellflow_atom_fingerprint"


def params_fingerprint(params: AtomParams) -> str:
    """Stable SHA-1 of the params, used to detect stale atoms.tif in DB-Gen."""
    payload = json.dumps(asdict(params), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def write_atoms_tif(path, atoms: np.ndarray, params: AtomParams) -> None:
    """Write the atom label stack with the params + fingerprint embedded in the
    TIFF ImageDescription, so DB-Gen (②) can read what produced it."""
    description = json.dumps(
        {_PARAMS_KEY: asdict(params), _FINGERPRINT_KEY: params_fingerprint(params)}
    )
    tifffile.imwrite(
        str(path), np.asarray(atoms, dtype=np.int32), description=description
    )


def read_atoms_params(path) -> tuple[dict | None, str | None]:
    """Return ``(params_dict, fingerprint)`` embedded by ``write_atoms_tif``, or
    ``(None, None)`` if the file has no atom metadata."""
    if not Path(path).exists():
        return None, None
    with tifffile.TiffFile(str(path)) as tf:
        description = tf.pages[0].description or ""
    try:
        meta = json.loads(description)
    except (json.JSONDecodeError, TypeError):
        return None, None
    return meta.get(_PARAMS_KEY), meta.get(_FINGERPRINT_KEY)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py -v`
Expected: PASS (all atoms tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/atoms.py tests/tracking_ultrack/test_atoms.py
git commit -m "feat(atoms): params fingerprint + atoms.tif write/read with metadata"
```

---

## Task 5: Config — atom params on `TrackingConfig`

**Files:**
- Modify: `src/cellflow/tracking_ultrack/config.py:9-50` (add fields in the model body)
- Test: `tests/tracking_ultrack/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking_ultrack/test_config.py
def test_tracking_config_exposes_atom_extraction_params():
    cfg = TrackingConfig()
    assert cfg.fg_window == 51
    assert cfg.fg_cutoff == 0.002
    assert cfg.contour_window == 51
    assert cfg.contour_floor == 0.01
    assert cfg.atom_min_area == 100


def test_tracking_config_atom_params_override():
    cfg = TrackingConfig(fg_cutoff=0.01, atom_min_area=250)
    assert cfg.fg_cutoff == 0.01
    assert cfg.atom_min_area == 250
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_config.py -k atom -v`
Expected: FAIL — `AttributeError: 'TrackingConfig' object has no attribute 'fg_window'`

- [ ] **Step 3: Write minimal implementation**

In `src/cellflow/tracking_ultrack/config.py`, add a new block of fields inside the `TrackingConfig` model (e.g. immediately after the `max_segments_per_time` line, before `# Linking`):

```python
    # Atom extraction (stage ①) — see atoms.AtomParams
    fg_window: int = 51
    fg_cutoff: float = 0.002
    contour_window: int = 51
    contour_floor: float = 0.01
    atom_min_area: int = 100
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_config.py -v`
Expected: PASS (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/tracking_ultrack/config.py tests/tracking_ultrack/test_config.py
git commit -m "feat(config): add atom-extraction params to TrackingConfig"
```

---

## Task 6: Paths — `nucleus_atoms`

**Files:**
- Modify: `src/cellflow/napari/_paths.py` (add property in the `2_nucleus` group, near `tracked`/`ultrack_db`)
- Test: `tests/napari/test_paths.py`

`atoms.tif` lives in `2_nucleus/` (a tracking-stage artifact, sibling of `ultrack_db` and `tracked`), not `1_cellpose/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/napari/test_paths.py
from __future__ import annotations

from pathlib import Path

from cellflow.napari._paths import NucleusArtifactPaths


def test_nucleus_atoms_path_under_2_nucleus():
    paths = NucleusArtifactPaths(pos_dir=Path("/data/pos00"))
    assert paths.nucleus_atoms == Path("/data/pos00/2_nucleus/atoms.tif")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_paths.py -v`
Expected: FAIL — `AttributeError: 'NucleusArtifactPaths' object has no attribute 'nucleus_atoms'`

- [ ] **Step 3: Write minimal implementation**

In `src/cellflow/napari/_paths.py`, in the `2_nucleus` section, add:

```python
    @property
    def nucleus_atoms(self) -> Path:
        return self.pos_dir / "2_nucleus" / "atoms.tif"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/_paths.py tests/napari/test_paths.py
git commit -m "feat(paths): nucleus_atoms -> 2_nucleus/atoms.tif"
```

---

## Task 7: Widget controls class

**Files:**
- Create: `src/cellflow/napari/nucleus_atom_extraction_widget.py`
- Test: `tests/napari/test_nucleus_atom_extraction_widget.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/napari/test_nucleus_atom_extraction_widget.py
"""Tests for the Atom Extraction widget (controls + behavior mixin)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication, QWidget

from cellflow.napari.nucleus_atom_extraction_widget import (
    NucleusAtomExtractionWidget,
)


def _app():
    return QApplication.instance() or QApplication([])


def test_controls_have_spec_defaults():
    _app()
    w = NucleusAtomExtractionWidget()
    assert w.fg_window_spin.value() == 51
    assert abs(w.fg_cutoff_spin.value() - 0.002) < 1e-9
    assert w.contour_window_spin.value() == 51
    assert abs(w.contour_floor_spin.value() - 0.01) < 1e-9
    assert w.atom_min_area_spin.value() == 100


def test_controls_have_activate_and_compute_and_overlays():
    _app()
    w = NucleusAtomExtractionWidget()
    assert w.active_btn.isCheckable()
    assert not w.active_btn.isChecked()
    assert w.compute_btn is not None
    assert w.territory_overlay_check is not None
    assert w.residual_overlay_check is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k controls -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellflow.napari.nucleus_atom_extraction_widget'`

- [ ] **Step 3: Write minimal implementation (controls class only)**

```python
# src/cellflow/napari/nucleus_atom_extraction_widget.py
"""Atom Extraction section for the nucleus workflow widget (stage ①)."""
from __future__ import annotations

import logging

import numpy as np
import tifffile
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    heading as _heading,
    islider as _islider,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_style import (
    add_section_header,
    add_section_pair_row,
    section_grid,
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.atoms import (
    AtomParams,
    extract_atoms_frame,
    extract_atoms_stack,
    residual,
    write_atoms_tif,
)

logger = logging.getLogger(__name__)

_ATOM_PREFIX = "[Atoms]"
_ATOM_PREVIEW_LAYER = f"{_ATOM_PREFIX} preview"
_ATOM_TERRITORY_LAYER = f"{_ATOM_PREFIX} territory"
_ATOM_RESIDUAL_LAYER = f"{_ATOM_PREFIX} residual_contour"


class NucleusAtomExtractionWidget(QWidget):
    """Qt controls for tuning atom extraction with a live preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.header = QWidget(parent)
        header_lay = QHBoxLayout(self.header)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(4)
        self.header_lbl = QLabel("Atom Extraction")
        _stage_header_label(self.header_lbl, "nucleus")
        self.active_btn = _tool_btn(
            "⏻", "Activate atom extraction preview.", checkable=True
        )
        self.active_btn.setChecked(False)
        _stage_header_action_button(self.active_btn, "nucleus")
        header_lay.addWidget(self.header_lbl)
        header_lay.addWidget(self.active_btn)
        header_lay.addStretch(1)

        inner = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(0, 0, 0, 0)
        inner.setLayout(grid)

        self.fg_window_spin = _islider(
            3, 301, 51, tooltip="Foreground residual window (px, forced odd)."
        )
        self.fg_cutoff_spin = _dslider(
            0, 1, 0.002, 0.001, 3, "Territory threshold on the fg residual."
        )
        self.contour_window_spin = _islider(
            3, 301, 51, tooltip="Contour residual window (px, forced odd)."
        )
        self.contour_floor_spin = _dslider(
            0, 1, 0.01, 0.001, 3, "Ridge noise floor on the contour residual."
        )
        self.atom_min_area_spin = _islider(
            0, 5000, 100, tooltip="Atoms smaller than this merge into a neighbour."
        )

        self.territory_overlay_check = QCheckBox("Territory overlay")
        self.residual_overlay_check = QCheckBox("Residual-contour overlay")
        self.compute_btn = QPushButton("Compute atoms (full stack)")
        self.compute_btn.setToolTip(
            "Run atom extraction over all frames and write atoms.tif."
        )
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setVisible(False)

        row = 0
        add_section_header(grid, row, _heading("Residual")); row += 1
        add_section_pair_row(
            grid, row,
            "FG window:", self.fg_window_spin,
            "FG cutoff:", self.fg_cutoff_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Contour window:", self.contour_window_spin,
            "Contour floor:", self.contour_floor_spin,
        ); row += 1
        add_section_header(grid, row, _heading("Atoms")); row += 1
        add_section_pair_row(grid, row, "Min area:", self.atom_min_area_spin); row += 1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(inner)
        overlay_row = QWidget(self)
        overlay_lay = QHBoxLayout(overlay_row)
        overlay_lay.setContentsMargins(0, 0, 0, 0)
        overlay_lay.addWidget(self.territory_overlay_check)
        overlay_lay.addWidget(self.residual_overlay_check)
        lay.addWidget(overlay_row)
        lay.addWidget(self.compute_btn)
        lay.addWidget(self.status_lbl)

        self.section: CollapsibleSection | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k controls -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/nucleus_atom_extraction_widget.py tests/napari/test_nucleus_atom_extraction_widget.py
git commit -m "feat(atom-widget): controls class with spec defaults"
```

---

## Task 8: Behavior mixin — params + preview lifecycle

**Files:**
- Modify: `src/cellflow/napari/nucleus_atom_extraction_widget.py`
- Test: `tests/napari/test_nucleus_atom_extraction_widget.py`

The mixin is hosted by a class that provides `self.viewer`, `self._current_t()`, and three path hooks (`_atom_fg_path`, `_atom_contour_path`, `_atom_output_path`). The test uses a tiny host so the behavior is exercised without the whole workflow widget.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/napari/test_nucleus_atom_extraction_widget.py
import napari

from cellflow.napari.nucleus_atom_extraction_widget import (
    NucleusAtomExtractionMixin,
    _ATOM_PREVIEW_LAYER,
)
from cellflow.tracking_ultrack.atoms import AtomParams


class _Host(NucleusAtomExtractionMixin, QWidget):
    """Minimal host exposing the surface the mixin needs.

    Subclasses QWidget so the mixin's ``QTimer(self)`` has a valid QObject
    parent — exactly as the real workflow widget (a QWidget) provides.
    """

    def __init__(self, viewer, fg, contour, out_path):
        super().__init__()
        self.viewer = viewer
        self._fg = fg
        self._contour = contour
        self._out_path = out_path
        self._init_atom_extraction_state()
        self.atom_extraction_widget = NucleusAtomExtractionWidget()
        self._alias_atom_extraction_controls()

    def _current_t(self):
        dims = self.viewer.dims
        return int(dims.current_step[0]) if dims.ndim else 0

    def _atom_fg_path(self):
        return self._fg

    def _atom_contour_path(self):
        return self._contour

    def _atom_output_path(self):
        return self._out_path


def _host(tmp_path):
    _app()
    fg = tmp_path / "fg.tif"
    contour = tmp_path / "contour.tif"
    rng = np.random.default_rng(0)
    tifffile.imwrite(fg, rng.random((3, 40, 40)).astype(np.float32))
    tifffile.imwrite(contour, rng.random((3, 40, 40)).astype(np.float32))
    viewer = napari.Viewer(show=False)
    viewer.add_image(np.asarray(tifffile.imread(fg)), name="fg")
    return _Host(viewer, fg, contour, tmp_path / "atoms.tif"), viewer


def test_atom_params_reads_controls():
    _app()
    h = _Host(napari.Viewer(show=False),
              fg=None, contour=None, out_path=None)  # noqa: viewer unused here
    h.atom_extraction_widget.fg_cutoff_spin.setValue(0.01)
    h.atom_extraction_widget.atom_min_area_spin.setValue(250)
    p = h._atom_params()
    assert isinstance(p, AtomParams)
    assert p.fg_cutoff == 0.01
    assert p.atom_min_area == 250
    napari.Viewer.close_all()


def test_activate_adds_preview_layer_then_deactivate_removes(tmp_path):
    h, viewer = _host(tmp_path)
    h._on_atom_activate(True)
    assert _ATOM_PREVIEW_LAYER in viewer.layers
    h._on_atom_activate(False)
    assert _ATOM_PREVIEW_LAYER not in viewer.layers
    napari.Viewer.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k "atom_params or activate" -v`
Expected: FAIL — `ImportError: cannot import name 'NucleusAtomExtractionMixin'`

- [ ] **Step 3: Write minimal implementation (append the mixin)**

```python
# append to src/cellflow/napari/nucleus_atom_extraction_widget.py
class NucleusAtomExtractionMixin:
    """Behavior for the Atom Extraction section.

    Host must provide: ``self.viewer``, ``self._current_t()``,
    ``self._atom_fg_path()``, ``self._atom_contour_path()``,
    ``self._atom_output_path()``.
    """

    def _init_atom_extraction_state(self) -> None:
        self._atom_preview_active = False
        self._atom_refresh_timer = QTimer(self)
        self._atom_refresh_timer.setSingleShot(True)
        self._atom_refresh_timer.setInterval(150)
        self._atom_refresh_timer.timeout.connect(self._refresh_atom_preview)

    def _alias_atom_extraction_controls(self) -> None:
        w = self.atom_extraction_widget
        for spin in (w.fg_window_spin, w.fg_cutoff_spin, w.contour_window_spin,
                     w.contour_floor_spin, w.atom_min_area_spin):
            spin.valueChanged.connect(self._on_atom_param_changed)
        w.territory_overlay_check.toggled.connect(self._on_atom_param_changed)
        w.residual_overlay_check.toggled.connect(self._on_atom_param_changed)
        w.active_btn.toggled.connect(self._on_atom_activate)
        # compute_btn is wired in Task 9 (when _compute_atoms_full_stack exists)

    def _atom_params(self) -> AtomParams:
        w = self.atom_extraction_widget
        return AtomParams(
            fg_window=int(w.fg_window_spin.value()),
            fg_cutoff=float(w.fg_cutoff_spin.value()),
            contour_window=int(w.contour_window_spin.value()),
            contour_floor=float(w.contour_floor_spin.value()),
            atom_min_area=int(w.atom_min_area_spin.value()),
        )

    def _set_atom_status(self, msg: str) -> None:
        lbl = self.atom_extraction_widget.status_lbl
        lbl.setText(msg)
        lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _on_atom_param_changed(self, *_args) -> None:
        if self._atom_preview_active:
            self._atom_refresh_timer.start()

    def _on_atom_activate(self, checked: bool) -> None:
        self._atom_preview_active = bool(checked)
        if checked:
            self._refresh_atom_preview()
        else:
            for name in (_ATOM_PREVIEW_LAYER, _ATOM_TERRITORY_LAYER,
                         _ATOM_RESIDUAL_LAYER):
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            self._set_atom_status("")

    def _read_frame(self, path, t: int) -> np.ndarray:
        return np.asarray(tifffile.imread(str(path), key=t), dtype=np.float32)

    def _refresh_atom_preview(self) -> None:
        if not self._atom_preview_active:
            return
        fg_path = self._atom_fg_path()
        contour_path = self._atom_contour_path()
        if fg_path is None or contour_path is None:
            self._set_atom_status("Foreground/contour maps not found.")
            return
        params = self._atom_params()
        t = self._current_t()
        try:
            fg = self._read_frame(fg_path, t)
            contour = self._read_frame(contour_path, t)
        except (FileNotFoundError, IndexError) as exc:
            self._set_atom_status(f"Cannot read maps for frame {t}: {exc}")
            return
        territory = residual(fg, params.fg_window) > params.fg_cutoff
        residual_contour = residual(contour, params.contour_window)
        atoms = extract_atoms_frame(
            residual_contour, territory, params.contour_floor, params.atom_min_area
        )
        self._update_atom_labels_layer(_ATOM_PREVIEW_LAYER, atoms)
        w = self.atom_extraction_widget
        self._toggle_atom_overlay(
            _ATOM_TERRITORY_LAYER, territory.astype(np.uint8),
            w.territory_overlay_check.isChecked(), is_labels=True,
        )
        self._toggle_atom_overlay(
            _ATOM_RESIDUAL_LAYER, residual_contour,
            w.residual_overlay_check.isChecked(), is_labels=False,
        )
        n_atoms = int(atoms.max())
        self._set_atom_status(f"Frame {t}: {n_atoms} atoms.")

    def _update_atom_labels_layer(self, name: str, data: np.ndarray) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            self.viewer.layers[name].data = data.astype(np.int32)
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data.astype(np.int32), name=name, opacity=0.55)

    def _toggle_atom_overlay(self, name, data, visible, *, is_labels) -> None:
        if not visible:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            return
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
            self.viewer.layers[name].visible = True
            return
        if is_labels:
            self.viewer.add_labels(data.astype(np.uint8), name=name, opacity=0.3)
        else:
            self.viewer.add_image(data, name=name, colormap="magma",
                                  blending="additive")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k "atom_params or activate" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/nucleus_atom_extraction_widget.py tests/napari/test_nucleus_atom_extraction_widget.py
git commit -m "feat(atom-widget): behavior mixin — params + preview lifecycle"
```

---

## Task 9: Compute-atoms-full-stack action

**Files:**
- Modify: `src/cellflow/napari/nucleus_atom_extraction_widget.py` (add `_compute_atoms_full_stack`)
- Test: `tests/napari/test_nucleus_atom_extraction_widget.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/napari/test_nucleus_atom_extraction_widget.py
from cellflow.tracking_ultrack.atoms import read_atoms_params, params_fingerprint


def test_compute_atoms_full_stack_writes_tif_with_fingerprint(tmp_path):
    h, viewer = _host(tmp_path)
    h.atom_extraction_widget.fg_window_spin.setValue(11)
    h.atom_extraction_widget.contour_window_spin.setValue(11)
    h._compute_atoms_full_stack()
    out = tmp_path / "atoms.tif"
    assert out.exists()
    atoms = tifffile.imread(out)
    assert atoms.shape == (3, 40, 40)
    stored_params, stored_fp = read_atoms_params(out)
    assert stored_fp == params_fingerprint(h._atom_params())
    napari.Viewer.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k compute -v`
Expected: FAIL — `AttributeError: '_Host' object has no attribute '_compute_atoms_full_stack'`

- [ ] **Step 3: Write minimal implementation (append to the mixin + wire the button)**

First, add the compute-button connection to the end of
`_alias_atom_extraction_controls` (the line deferred in Task 8):

```python
        self.atom_extraction_widget.compute_btn.clicked.connect(
            self._compute_atoms_full_stack
        )
```

Then append the method:

```python
# append inside NucleusAtomExtractionMixin in nucleus_atom_extraction_widget.py
    def _compute_atoms_full_stack(self) -> None:
        fg_path = self._atom_fg_path()
        contour_path = self._atom_contour_path()
        out_path = self._atom_output_path()
        if fg_path is None or contour_path is None or out_path is None:
            self._set_atom_status("Foreground/contour maps not found.")
            return
        params = self._atom_params()
        self._set_atom_status("Computing atoms over all frames…")
        fg = np.asarray(tifffile.imread(str(fg_path)), dtype=np.float32)
        contour = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
        atoms = extract_atoms_stack(fg, contour, params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_atoms_tif(out_path, atoms, params)
        self._set_atom_status(
            f"Wrote {atoms.shape[0]} frames to {out_path.name} "
            f"({int(atoms.max())} atoms in last frame)."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -v`
Expected: PASS (all widget tests)

- [ ] **Step 5: Commit**

```bash
git add src/cellflow/napari/nucleus_atom_extraction_widget.py tests/napari/test_nucleus_atom_extraction_widget.py
git commit -m "feat(atom-widget): compute atoms full stack -> atoms.tif"
```

---

## Task 10: Wire into the nucleus workflow widget

**Files:**
- Modify: `src/cellflow/napari/nucleus_workflow_widget.py` (class bases ~line 73; `__init__` state init ~line 83; `_build_*` calls ~line 133-137; add section method + path hooks)
- Test: `tests/napari/test_nucleus_atom_extraction_widget.py`

The workflow widget supplies the path hooks the mixin needs, using its existing `_paths`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/napari/test_nucleus_atom_extraction_widget.py
def test_workflow_widget_builds_atom_extraction_section():
    _app()
    from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
    w = NucleusWorkflowWidget(napari.Viewer(show=False))
    assert hasattr(w, "atom_extraction_widget")
    assert w.atom_extraction_section is not None
    # path hooks resolve to None without a loaded position (no crash)
    assert w._atom_output_path() is None or str(w._atom_output_path()).endswith("atoms.tif")
    napari.Viewer.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k workflow -v`
Expected: FAIL — `AttributeError: 'NucleusWorkflowWidget' object has no attribute 'atom_extraction_widget'`

- [ ] **Step 3: Write minimal implementation**

In `src/cellflow/napari/nucleus_workflow_widget.py`:

(a) Add the import near the other widget imports (~line 37):

```python
from cellflow.napari.nucleus_atom_extraction_widget import (
    NucleusAtomExtractionMixin,
)
```

(b) Add the mixin to the class bases (line 73):

```python
class NucleusWorkflowWidget(
    NucleusUltrackDbBrowserMixin, NucleusAtomExtractionMixin, QWidget
):
```

(c) In `__init__`, alongside `self._init_ultrack_db_browser_state()` (~line 83):

```python
        self._init_atom_extraction_state()
```

(d) In the section-building block (after `self._build_tracking_ultrack_section(root)`, ~line 134):

```python
        self._build_atom_extraction_section(root)
```

(e) Add the section builder + path hooks as methods on the class (place near `_build_db_browser_section`):

```python
    def _build_atom_extraction_section(self, root) -> None:
        from cellflow.napari.nucleus_atom_extraction_widget import (
            NucleusAtomExtractionWidget,
        )
        from cellflow.napari.widgets import CollapsibleSection

        self.atom_extraction_widget = NucleusAtomExtractionWidget(self)
        self.atom_extraction_section = CollapsibleSection(
            "Atom Extraction", self.atom_extraction_widget, expanded=False,
        )
        self._alias_atom_extraction_controls()
        root.addWidget(self.atom_extraction_widget.header)
        root.addWidget(self.atom_extraction_section)

    def _atom_fg_path(self):
        return self._paths.nucleus_foreground if self._paths else None

    def _atom_contour_path(self):
        return self._paths.nucleus_contours if self._paths else None

    def _atom_output_path(self):
        return self._paths.nucleus_atoms if self._paths else None
```

`_paths` (property, ~line 635), `_current_t` (~line 706), and `self.viewer`
already exist on `NucleusWorkflowWidget` — reuse them; do not redefine.

(f) Wire frame-follow into the existing dims handler. In `_on_dims_step_changed`
(~line 756), alongside the browser-refresh line, add:

```python
        if getattr(self, "_atom_preview_active", False):
            QTimer.singleShot(0, self._refresh_atom_preview)
```

This re-renders the atom preview for the new frame when the section is active,
matching how the DB browser refreshes on frame change.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k workflow -v`
Expected: PASS

- [ ] **Step 5: Run the full widget + workflow test files (regression)**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py tests/napari/test_nucleus_pipeline_widget.py -v`
Expected: PASS (no regressions in the workflow)

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/nucleus_workflow_widget.py tests/napari/test_nucleus_atom_extraction_widget.py
git commit -m "feat(atom-widget): wire Atom Extraction section into nucleus workflow"
```

---

## Task 11: Persist atom params in workflow state

**Files:**
- Modify: `src/cellflow/napari/_state.py` (`dump_state` ~line 19, `load_state` ~line 64)
- Test: `tests/napari/test_nucleus_atom_extraction_widget.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/napari/test_nucleus_atom_extraction_widget.py
def test_state_round_trip_for_atom_params():
    _app()
    from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
    from cellflow.napari._state import dump_state, load_state

    w = NucleusWorkflowWidget(napari.Viewer(show=False))
    w.atom_extraction_widget.fg_cutoff_spin.setValue(0.01)
    w.atom_extraction_widget.atom_min_area_spin.setValue(300)
    state = dump_state(w)
    assert state["atom_extraction"]["fg_cutoff"] == 0.01
    assert state["atom_extraction"]["atom_min_area"] == 300

    w2 = NucleusWorkflowWidget(napari.Viewer(show=False))
    load_state(w2, state)
    assert w2.atom_extraction_widget.fg_cutoff_spin.value() == 0.01
    assert w2.atom_extraction_widget.atom_min_area_spin.value() == 300
    napari.Viewer.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k state -v`
Expected: FAIL — `KeyError: 'atom_extraction'`

- [ ] **Step 3: Write minimal implementation**

In `src/cellflow/napari/_state.py`, add a block to the `dump_state` return dict:

```python
        "atom_extraction": {
            "fg_window": w.atom_extraction_widget.fg_window_spin.value(),
            "fg_cutoff": w.atom_extraction_widget.fg_cutoff_spin.value(),
            "contour_window": w.atom_extraction_widget.contour_window_spin.value(),
            "contour_floor": w.atom_extraction_widget.contour_floor_spin.value(),
            "atom_min_area": w.atom_extraction_widget.atom_min_area_spin.value(),
        },
```

And in `load_state`, after the existing blocks:

```python
    if "atom_extraction" in state:
        ae = state["atom_extraction"]
        aw = w.atom_extraction_widget
        if "fg_window" in ae: aw.fg_window_spin.setValue(ae["fg_window"])
        if "fg_cutoff" in ae: aw.fg_cutoff_spin.setValue(ae["fg_cutoff"])
        if "contour_window" in ae: aw.contour_window_spin.setValue(ae["contour_window"])
        if "contour_floor" in ae: aw.contour_floor_spin.setValue(ae["contour_floor"])
        if "atom_min_area" in ae: aw.atom_min_area_spin.setValue(ae["atom_min_area"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n cellflow pytest tests/napari/test_nucleus_atom_extraction_widget.py -k state -v`
Expected: PASS

- [ ] **Step 5: Run the whole new-feature test surface**

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack/test_atoms.py tests/tracking_ultrack/test_config.py tests/napari/test_paths.py tests/napari/test_nucleus_atom_extraction_widget.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add src/cellflow/napari/_state.py tests/napari/test_nucleus_atom_extraction_widget.py
git commit -m "feat(atom-widget): persist atom params in workflow state"
```

---

## Final verification

- [ ] Run the full nucleus napari + tracking test suites to confirm no regressions:

Run: `conda run --no-capture-output -n cellflow pytest tests/tracking_ultrack tests/napari -q`
Expected: PASS (pre-existing passing tests still pass; new tests pass)

- [ ] Manual smoke (optional): launch napari, load a position with `nucleus_foreground.tif` + `nucleus_contours.tif`, open the **Atom Extraction** section, hit `⏻`, scrub sliders and the frame slider, toggle the territory/residual overlays, then click **Compute atoms (full stack)** and confirm `2_nucleus/atoms.tif` is written.

## Notes for the implementer

- **Scope:** This is additive only. Do **not** remove the existing higra DB-generation controls, `seg_*` config fields, or the source-threshold machinery — spec ② removes those when it swaps the DB-gen consumer to read `atoms.tif`.
- **`threshold_local` window:** keep unit-test images comfortably larger than the window (the tests use window 11 on 40×40). The function forces the window odd.
- **napari layer churn:** the preview reuses a single `[Atoms] preview` Labels layer by name (updates `.data` in place) to avoid flicker; overlays are added/removed on toggle.
- **Patterns to mirror:** `NucleusUltrackDbBrowserWidget` / `...Mixin` (controls-class-plus-behavior-mixin split, `⏻` activate button, debounced `QTimer` refresh) and `nucleus_tracking_inputs_widget` (`_dslider`/`_islider`/`section_grid` layout helpers).
