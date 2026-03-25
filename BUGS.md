# Bug Log — napariTissueFlow

---

## BUG-001 · Two-channel mode produces only one segmentation layer

**Tab:** Segmentation → Two Channel mode
**Symptom:** Running segmentation in Two Channel mode outputs a single "Segmentation" label layer. The user expects two independent label layers — one per channel (e.g. "Segmentation – primary" and "Segmentation – secondary").
**Root cause:** `_get_or_create_labels_layer` always targets one layer named "Segmentation". In two-channel mode, Cellpose fuses both channels into a single mask, so only one output exists. If the intent is to segment each channel independently and produce separate layers, the current pipeline does not support that — it uses the two channels as a *joint input* to one Cellpose call (`channels=[1, 2]`).
**Files:** `segtrack/_segmentation_tab.py` (`_on_frame_done`, `_on_stack_done`, `_get_or_create_labels_layer`), `segtrack/_pipeline.py` (`run_cp_two_channel`)

---

## BUG-002 · GPU checkbox has no effect / GPU not used

**Tab:** Segmentation → Cellpose Parameters → GPU checkbox
**Symptom:** Enabling the GPU checkbox does not result in GPU inference. Cellpose runs on CPU regardless.
**Root cause:** Not confirmed yet. `make_cp_model` passes `gpu=params["gpu"]` to `CellposeModel(gpu=..., ...)`. In cellpose ≥ 4.0, GPU availability is auto-detected and the `gpu` kwarg may have changed semantics. Need to verify that the `gpu` parameter is still the correct kwarg in the installed cellpose version, and that CUDA is actually available in the environment.
**Files:** `segtrack/_pipeline.py` (`make_cp_model`)

---

## BUG-003 · `cpsam` model not found; `channels` parameter deprecated in cellpose ≥ 4.0.1

**Tab:** Segmentation
**Symptoms (from log):**
```
WARNING : pretrained model /home/aruppel/.cellpose/models/cpsam not found, using default model
WARNING : channels deprecated in v4.0.1+. If data contain more than 3 channels, only the first 3 channels will be used
```
**Root causes:**
- `cpsam` is not downloaded automatically; `CellposeModel(pretrained_model="cpsam")` silently falls back to the default model instead of raising a clear error to the user.
- `channels=[0, 0]` (in `run_cp`) and `channels=[1, 2]` (in `run_cp_two_channel`) are deprecated since cellpose 4.0.1. In v4, image channels are inferred from array shape; passing the `channels` kwarg triggers a warning and may produce unexpected behaviour. Need to update the `model.eval()` calls to use the new v4 API.
**Files:** `segtrack/_pipeline.py` (`make_cp_model`, `run_cp`, `run_cp_two_channel`)

---

## BUG-004 · Plugin freezes when segmenting the same frame twice with different models

**Tab:** Segmentation
**Reproduction steps:** Segment a frame with model A → change model to B → segment the same frame again → UI freezes.
**Symptom:** Widget becomes unresponsive; progress bar stays visible; buttons stay disabled.
**Root cause:** Not confirmed. Likely candidates:
- CellposeModel holds GPU memory from the first run; instantiating a second model before the first is garbage-collected causes a CUDA OOM or deadlock.
- napari's `thread_worker` does not enforce mutual exclusion between consecutive runs: `self._worker.is_running` may briefly read `False` during the teardown of the first worker while the Qt event loop has not fully cleared the thread, leading to two workers running simultaneously and contending on shared state.
- The freeze may be reproducible on CPU too (in which case GPU is not the cause).
**Files:** `segtrack/_segmentation_tab.py` (`_on_segment_frame`, `_on_segment_stack`, `_on_frame_done`)
