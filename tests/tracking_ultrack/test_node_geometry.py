from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from cellflow.tracking_ultrack._node_geometry import (
    node_bbox_and_mask,
    node_pickle_ndim,
)


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
