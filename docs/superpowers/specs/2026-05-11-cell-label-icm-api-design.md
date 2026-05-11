# Cell Label ICM API Design

**Date:** 2026-05-11
**Status:** Approved

## Overview

Extract the ICM + geodesic-unary cell labelling pipeline from `scripts/experiment_cell_2d_t_multilabel_graphcut.py` into a reusable Python API that can be called from napari widgets (arrays in memory) and pipeline scripts (pos_dir on disk). Only the ICM solver with geodesic unary mode and unary argmin initialisation is included; alpha-expansion, all other unary modes, and the lambda-area cost term are dropped.

## Module location

`src/cellflow/segmentation/cell_label_icm.py` — a single new file.

`CellLabelICMParams` and `segment_cells_icm` are exported from `src/cellflow/segmentation/__init__.py`. `run_cell_icm_from_pos_dir` is not re-exported at the package level; callers import it directly from the module.

## Public surface

```python
@dataclass(frozen=True, slots=True)
class CellLabelICMParams:
    alpha_unary: float = 200.0   # contour weight in geodesic cost field
    lambda_s: float = 1.0        # spatial pairwise weight
    beta_s: float = 5.0          # contour sensitivity in spatial pairwise
    lambda_t: float = 0.1        # temporal pairwise weight
    n_iters: int = 25            # ICM rounds
    min_round_flips: int = 0     # early-stop if a round flips fewer than this

def segment_cells_icm(
    nuc_tracks: np.ndarray,    # (T, Y, X) uint32 nucleus tracked labels
    fg_mask: np.ndarray,       # (T, Y, X) bool
    contours: np.ndarray,      # (T, Y, X) float32 contour maps
    params: CellLabelICMParams,
) -> np.ndarray:               # (T, Y, X) uint32 predicted cell labels

def run_cell_icm_from_pos_dir(
    pos_dir: Path,
    params: CellLabelICMParams,
    *,
    crop: tuple[int, int, int, int, int, int] | None = None,  # T0,T1,Y0,Y1,X0,X1
) -> np.ndarray:               # (T, Y, X) uint32 predicted cell labels
```

`lambda_area` is hardwired to `0.0` internally and not exposed. Init mode is always unary argmin. The pos_dir wrapper returns the label array; saving to disk is the caller's responsibility.

## Internal pipeline

`segment_cells_icm` runs the following steps, all via private functions:

1. **Validate** — assert `nuc_tracks`, `fg_mask`, `contours` share the same `(T, Y, X)` shape; raise `ValueError` if not.

2. **Compute pairwise weights** (`_compute_pairwise_weights`) — boundary signal is always the raw contour map. Returns the nine edge-weight arrays `h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l`.

3. **Compute geodesic unaries** (`_compute_geodesic_unaries`) — MCP_Geometric distance from each nucleus label in each frame, weighted by `1 + alpha_unary * contour`. Depends on Numba helpers `_build_nucleus_pixels`, `_nb_integrate_flow`, `_nb_flow_unary_raw`, `_normalize_flow_unary`. Returns `dict[(frame, label_id), np.ndarray]`.

4. **Apply nucleus anchors** (`_apply_nucleus_anchors`) — pins nucleus-occupied pixels to their track ID (zero cost for the correct label, INF for all others).

5. **Densify** (`_dict_to_dense_unary`) — reshapes the dict into a `(T, Y, X, K)` float32 array.

6. **Run ICM** (`_run_icm`) — `lambda_area=0.0` hardwired, `init_labels=None` (unary argmin init). Calls the Numba kernel `_nb_icm_round`. Returns `(pred_labels, energy_log)`; `energy_log` is discarded by the public function.

All graph-cut code (`_alpha_cut`, maxflow, alpha-expansion) and multi-solver dispatch logic are not carried over.

## I/O — pos_dir wrapper

`run_cell_icm_from_pos_dir` loads:

| File | Array |
|---|---|
| `2_nucleus/tracked_labels.tif` | `nuc_tracks` uint32 |
| `3_cell/foreground_masks.tif` | `fg_mask` uint8 → bool |
| `3_cell/contour_maps.tif` | `contours` float32 |

Applies crop `(T0,T1,Y0,Y1,X0,X1)` if provided. Unions `fg_mask |= nuc_tracks > 0` (nucleus pixels are always foreground). Delegates to `segment_cells_icm` and returns the result.

## Error handling

- Shape mismatch on the three input arrays → `ValueError` before any computation.
- Missing TIFF in `run_cell_icm_from_pos_dir` → `FileNotFoundError` with the full path.
- Params are trusted; no defensive validation beyond shapes.

## Testing

`tests/segmentation/test_cell_label_icm.py`:

- Synthetic `(T=3, Y=32, X=32)` fixture with two nucleus seeds exercises the full `segment_cells_icm` call (output shape and dtype, all foreground pixels labelled).
- `run_cell_icm_from_pos_dir` tested with dummy TIFFs written to `tmp_path`; verifies correct file loading and crop application.
- Shape-mismatch test verifies `ValueError` is raised.
