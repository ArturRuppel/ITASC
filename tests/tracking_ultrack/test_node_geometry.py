from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from cellflow.tracking_ultrack._node_geometry import (
    centroid_gate,
    node_bbox_and_mask,
    node_pickle_ndim,
)


def test_centroid_gate_admits_overlap_with_centroid_outside_bbox():
    """An elongated source: a node overlapping one end but whose centroid lies
    outside the source bbox must pass the distance gate (the old bbox-containment
    prefilter would have dropped it)."""
    src = np.zeros((24, 24), dtype=bool)
    src[2:4, 2:18] = True  # wide, short bar — bbox rows [2,4), cols [2,18)
    gate = centroid_gate(src)
    assert gate is not None
    cy, cx, radius = gate
    assert radius == float(np.hypot(1, 15))  # bbox diagonal (max-min spans)

    # A tall node overlapping the left end; its centroid (row 10) is below the
    # source bbox (rows 2..4) → bbox containment rejects, distance gate admits.
    node_y, node_x = 10.0, 3.0
    assert not (2 <= node_y < 4)  # outside source bbox rows
    assert np.hypot(node_y - cy, node_x - cx) <= radius

    # A genuinely distant node stays pruned.
    assert np.hypot(23.0 - cy, 23.0 - cx) > radius


def test_centroid_gate_empty_mask_is_none():
    assert centroid_gate(np.zeros((5, 5), dtype=bool)) is None


def test_node_geometry_decodes_3d_pickled_node_as_2d_bbox_and_mask():
    mask = np.ones((4, 5), dtype=bool)
    node_pickle = pickle.dumps(
        SimpleNamespace(
            bbox=np.array([0, 7, 11, 1, 11, 16], dtype=np.int64),
            mask=mask[np.newaxis],
        )
    )

    assert node_pickle_ndim(node_pickle) == 3
    bbox, parsed_mask = node_bbox_and_mask(123, node_pickle)
    assert bbox == (7, 11, 11, 16)
    assert parsed_mask.shape == (4, 5)
    assert parsed_mask.dtype == bool
    assert parsed_mask.all()
