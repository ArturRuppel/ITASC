## packaging (do before adding more stages)

goal: `pip install cellflow[correction]`, `cellflow[edge-analysis]`, `cellflow[all]`

- correction+tracking only (no cellpose/ultrack): for users who segmented elsewhere
- edge-analysis: correction + graph/topology/forces, for users with tracked labels
- all: the full nuclear-guided pipeline (cellpose + ultrack + everything)

steps:
1. move `cellpose>=3.1.0` and `laptrack>=0.14.0` out of `cellflow-napari` hard deps
   - `cellflow-analysis` stays as a hard dep (its deps are appropriate for edge-analysis)
   - add extras: `correction = ["laptrack"]`, `pipeline = ["cellflow-cellpose", "cellflow-ultrack", "cellpose", "laptrack"]`, `all = ["cellflow-napari[correction,pipeline]"]`
2. guard laptrack imports in `tracking_widget.py` — missing package should disable the tab, not crash
3. guard cellpose imports in `segmentation_widget.py` and `ultrack_widgets/` — same
4. any new pipeline stage widgets go into their stage package (`cellflow-cellpose`, `cellflow-ultrack`), not into `cellflow-napari` directly

there is grey text in some places which is impossible to read on the dark grey background. make white please.

manifest and actual folder names mismatch sometimes. to be verified.

the ordering of the steps and the logic of the folders is messed up. it's 2_contours, then 3_ultrack (instead of just tracking) 4_nucleus_anchored_cell_segmentation (that name makes more sense, should be changed in the UI and maybe code as well)