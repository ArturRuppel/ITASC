import numpy as np

from itasc import segmentation


def test_fill_and_close_labels_fills_per_label_bounding_boxes(monkeypatch):
    labels = np.zeros((20, 20), dtype=np.uint32)
    labels[2:5, 3:6] = 1
    labels[3, 4] = 0
    labels[12:14, 15:17] = 2
    labels[12, 16] = 0
    seen_shapes = []

    def fake_fill(mask):
        seen_shapes.append(mask.shape)
        return np.ones_like(mask, dtype=bool)

    monkeypatch.setattr("scipy.ndimage.binary_fill_holes", fake_fill)

    result = segmentation._fill_and_close_labels(labels)

    assert result[2:5, 3:6].tolist() == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert result[12:14, 15:17].tolist() == [[2, 2], [2, 2]]
    assert seen_shapes == [(3, 3), (2, 2)]
