# """Cell foreground mask generation via Cellpose dynamics."""
# from __future__ import annotations

# from collections.abc import Callable

# import numpy as np


# def compute_cellpose_foreground_masks(
#     prob_tzyx: np.ndarray,
#     filtered_dp_tcyx: np.ndarray,
#     *,
#     cellprob_threshold: float = 0.5,
#     flow_threshold: float = 0.0,
#     min_size: int = 15,
#     niter: int = 200,
#     progress_cb: Callable[[int, int], None] | None = None,
# ) -> np.ndarray:
#     """Generate binary cell foreground masks with Cellpose dynamics.

#     prob_tzyx is Cellpose probability logits shaped (T, Z, Y, X), or a single
#     volume shaped (Z, Y, X). filtered_dp_tcyx must be the filtered flow stack
#     produced by the cell workflow, shaped (T, 2, Y, X).

#     Probabilities are obtained via sigmoid, then averaged across Z before
#     being passed to Cellpose's ``compute_masks``.
#     """
#     prob = np.asarray(prob_tzyx, dtype=np.float32)
#     if prob.ndim == 3:
#         prob = prob[np.newaxis]
#     if prob.ndim != 4:
#         raise ValueError(
#             f"Expected probability shape (T, Z, Y, X) or (Z, Y, X), got {prob.shape}"
#         )

#     filtered_dp = np.asarray(filtered_dp_tcyx, dtype=np.float32)
#     if filtered_dp.ndim != 4 or filtered_dp.shape[1] != 2:
#         raise ValueError(
#             f"Expected filtered flow shape (T, 2, Y, X), got {filtered_dp.shape}"
#         )
#     if prob.shape[0] != filtered_dp.shape[0] or prob.shape[2:] != filtered_dp.shape[2:]:
#         raise ValueError(
#             "Cellpose probability and filtered flow shapes do not match: "
#             f"probability {prob.shape}, filtered flow {filtered_dp.shape}"
#         )

#     try:
#         import torch
#         from cellpose.dynamics import compute_masks
#     except ImportError as exc:
#         raise ImportError(
#             "cellpose and torch must be installed to generate Cellpose foreground masks"
#         ) from exc

#     # Sigmoid then z-average (average probabilities, not logits)
#     sigmoid_prob = 1.0 / (1.0 + np.exp(-prob))
#     prob_tyx = sigmoid_prob.mean(axis=1).astype(np.float32, copy=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     out = np.zeros(prob_tyx.shape, dtype=np.uint8)

#     for t in range(prob_tyx.shape[0]):
#         result = compute_masks(
#             filtered_dp[t],
#             prob_tyx[t],
#             cellprob_threshold=float(cellprob_threshold),
#             flow_threshold=float(flow_threshold),
#             min_size=int(min_size),
#             niter=int(niter),
#             do_3D=False,
#             device=device,
#         )
#         masks = result[0] if isinstance(result, tuple) else result
#         out[t] = (np.asarray(masks) > 0).astype(np.uint8)
#         if progress_cb is not None:
#             progress_cb(t + 1, prob_tyx.shape[0])

#     return out
"""Cell foreground mask generation via probability thresholding."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def compute_cellpose_foreground_masks(
    prob_tzyx: np.ndarray,
    filtered_dp_tcyx: np.ndarray,
    *,
    cellprob_threshold: float = 0.5,
    flow_threshold: float = 0.0,
    min_size: int = 15,
    niter: int = 200,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Generate binary cell foreground masks by thresholding probability.

    prob_tzyx is Cellpose probability logits shaped (T, Z, Y, X), or a single
    volume shaped (Z, Y, X). filtered_dp_tcyx is accepted for API
    compatibility but is not used.

    Probabilities are obtained via sigmoid, then averaged across Z.  The
    resulting (T, Y, X) map is thresholded at *cellprob_threshold* to produce
    a binary mask.
    """
    prob = np.asarray(prob_tzyx, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob[np.newaxis]
    if prob.ndim != 4:
        raise ValueError(
            f"Expected probability shape (T, Z, Y, X) or (Z, Y, X), got {prob.shape}"
        )

    # Sigmoid then z-average (average probabilities, not logits)
    sigmoid_prob = 1.0 / (1.0 + np.exp(-prob))
    prob_tyx = sigmoid_prob.mean(axis=1)

    out = np.zeros(prob_tyx.shape, dtype=np.uint8)

    for t in range(prob_tyx.shape[0]):
        out[t] = (prob_tyx[t] >= cellprob_threshold).astype(np.uint8)
        if progress_cb is not None:
            progress_cb(t + 1, prob_tyx.shape[0])

    return out