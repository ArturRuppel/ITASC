# Bug Log — napariTissueFlow

---

## BUG-001 · Two-channel mode: unclear what the expected output should be

**Tab:** Segmentation → Two Channel mode
**Symptom:** Running segmentation in Two Channel mode outputs a single "Segmentation" label layer.
**Context:** The two input channels are **cells (cytoplasm)** and **nuclei**. In Cellpose's native two-channel mode, the nuclear channel is used as a *guide* to improve cell-body segmentation — not segmented independently. The result is therefore a single mask of cell bodies, which is the correct Cellpose output for `channels=[1, 2]`.
**Open design question:** We need to decide what the UI should actually produce:
- **Option A (current behaviour):** One "Segmentation" layer containing cell-body masks, with the nuclear channel assisting Cellpose (standard two-channel workflow). This is probably the right default for the downstream tissue-graph pipeline which needs cell bodies.
- **Option B (two independent layers):** Run Cellpose separately on each channel and produce two label layers — "Segmentation – cells" and "Segmentation – nuclei". This lets the user inspect/correct both segmentations independently, but the nuclear masks would not feed the current pipeline without additional plumbing.
**TODO:** Clarify intended behaviour with user. If Option B is desired, `_run_segmentation` needs to return a tuple of two masks, `_on_frame_done` / `_on_stack_done` need to write to two named layers, and `_get_or_create_labels_layer` needs a name argument.
**Files:** `segtrack/_segmentation_tab.py` (`_on_frame_done`, `_on_stack_done`, `_get_or_create_labels_layer`), `segtrack/_pipeline.py` (`run_cp_two_channel`)

---

## BUG-002 · GPU checkbox has no effect / GPU not used ✓ FIXED

**Tab:** Segmentation → Cellpose Parameters → GPU checkbox
**Symptom:** Enabling the GPU checkbox does not result in GPU inference. Cellpose runs on CPU regardless.
**Root cause:** `make_cp_model` passed `gpu=True` without first verifying that `torch.cuda.is_available()`. If CUDA is absent, cellpose silently fell back to CPU without informing the user.
**Fix (committed):** `make_cp_model` now calls `torch.cuda.is_available()` before constructing the model. If CUDA is not available and the user requested GPU, a `WARNING` is logged: *"GPU requested but CUDA is not available; using CPU."* This makes the fallback visible rather than silent.
**Residual (env issue):** The `napariTissueFlow` conda environment ships with CPU-only torch. GPU will not work until a CUDA-enabled torch is installed in that env: `pip install torch --index-url https://download.pytorch.org/whl/cu128` (match CUDA version to system).
**Files:** `segtrack/_pipeline.py` (`make_cp_model`)

---

## BUG-003 · `cpsam` model not found; `channels` parameter deprecated in cellpose ≥ 4.0.1 ✓ FIXED

**Tab:** Segmentation
**Symptoms (from log):**
```
WARNING : pretrained model /home/aruppel/.cellpose/models/cpsam not found, using default model
WARNING : channels deprecated in v4.0.1+. If data contain more than 3 channels, only the first 3 channels will be used
```
**Root causes:**
- `cpsam` is not downloaded automatically; `CellposeModel(pretrained_model="cpsam")` silently fell back to the default model.
- `channels=[0, 0]` (in `run_cp`) and `channels=[1, 2]` (in `run_cp_two_channel`) are deprecated since cellpose 4.0.1. In v4, image channels are inferred from array shape.
**Fix (committed):**
- `make_cp_model` now resolves the model path via `cellpose.models.model_dir` (with fallback to `~/.cellpose/models/`) and raises a `FileNotFoundError` with a download hint if the file is absent.
- Additionally, after constructing `CellposeModel`, the loaded `pretrained_model` attribute is checked. If cellpose silently substituted a different model (e.g. because the file at `~/.cellpose/models/cpsam` is a stale cellpose 3.x download incompatible with 4.x), a `RuntimeError` is raised with the exact path to delete, turning the silent fallback into an actionable error.
- `cpsam` is kept in the dropdown — it is the highest-performing cellpose 4.x model. If the stale 3.x file is present, the error message now tells the user to delete it so cellpose 4.x can re-download.
- Removed `channels=[0, 0]` from `run_cp` and `channels=[1, 2]` from `run_cp_two_channel`. Cellpose v4 infers channel layout from array shape: `(H, W)` → grayscale; `(H, W, 2)` → two-channel.
- `run_pipeline` now calls `make_cp_model` instead of constructing `CellposeModel` directly, so it gets the same GPU and model-validation logic.
**Files:** `segtrack/_pipeline.py` (`make_cp_model`, `run_cp`, `run_cp_two_channel`, `run_pipeline`)

---

## BUG-004 · Plugin freezes when segmenting the same frame twice with different models ✓ FIXED

**Tab:** Segmentation
**Reproduction steps:** Segment a frame with model A → change model to B → segment the same frame again → UI freezes.
**Symptom:** Widget becomes unresponsive; progress bar stays visible; buttons stay disabled.
**Root cause:** `self._worker.is_running` briefly reads `False` during worker teardown (between the `returned` signal and the Qt thread fully quitting), creating a window where a second worker could be launched before the first has cleaned up. Two workers contending on the same shared state (and potentially the same GPU) then deadlock.
**Fix (committed):** Replaced the racy `self._worker.is_running` guard with an explicit `self._is_running: bool` flag. The flag is set to `True` immediately before `_work()` is created and reset to `False` as the very first statement in `_on_frame_done`, `_on_stack_done`, and `_on_error`. This is set before any other state changes, making it impossible for a second run to slip in during teardown.
**Files:** `segtrack/_segmentation_tab.py` (`__init__`, `_on_segment_frame`, `_on_segment_stack`, `_on_frame_done`, `_on_stack_done`, `_on_error`)
