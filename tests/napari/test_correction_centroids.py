from __future__ import annotations

import numpy as np

from cellflow.napari._correction_centroids import (
    correction_label_color_map,
    ensure_label_colormap_entries,
)


def test_correction_label_color_map_gives_high_new_labels_non_black_colors() -> None:
    labels = np.array([[[0, 1, 4096]]], dtype=np.uint32)

    color_map = correction_label_color_map(labels)

    assert 4096 in color_map
    assert not np.allclose(np.asarray(color_map[4096])[:3], [0.0, 0.0, 0.0])
    assert color_map[None] == "transparent"
    assert color_map[0] == "transparent"


def test_ensure_label_colormap_entries_extends_existing_dict() -> None:
    class _Layer:
        colormap = None

    layer = _Layer()
    color_map = ensure_label_colormap_entries(layer, [3, 8])

    assert {3, 8} <= set(color_map)
    for label_id in (3, 8):
        assert not np.allclose(np.asarray(color_map[label_id])[:3], [0.0, 0.0, 0.0])
