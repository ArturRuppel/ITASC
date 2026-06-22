"""The optional NLS classification pipeline step.

``classify`` is the headless, config-gated step that writes each position's NLS
sidecar CSV before aggregation. The classification *engine* is tested separately
(test_nls_classification.py); these cover the orchestration: per-position path
resolution, method→threshold selection, graceful skipping, and the disabled
no-op.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from cellflow.aggregate_quantification import pipeline
from cellflow.aggregate_quantification.config import NlsConfig
from cellflow.aggregate_quantification.contacts.nls_classification import (
    nls_classification_csv_path,
)


def _write_position(root: Path, *, with_nls: bool = True) -> dict:
    """A position dir with a 2-track nucleus-label stack and (optionally) a marker
    image where track 1 is bright and track 2 is dim — so any splitter separates
    them. Returns the catalogue record."""
    position = root / "pos1"
    inputs = position / "0_input"
    inputs.mkdir(parents=True)
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[:, :2, :] = 1
    labels[:, 2:, :] = 2
    labels_path = inputs / "nucleus_tracked_labels.tif"
    tifffile.imwrite(labels_path, labels)

    record = {
        "id": "pos1",
        "experiment_id": "EXP1",
        "position_path": str(position),
        "nucleus_tracked_labels_path": str(labels_path),
    }
    if with_nls:
        nls = np.zeros((2, 4, 4), dtype=np.float32)
        nls[:, :2, :] = 100.0
        nls[:, 2:, :] = 1.0
        tifffile.imwrite(inputs / "NLS_zavg.tif", nls)
    return record


def test_classify_disabled_is_noop(tmp_path):
    record = _write_position(tmp_path)
    assert pipeline.classify([record], config=None) == []
    assert pipeline.classify([record], config=NlsConfig(enabled=False)) == []
    assert not nls_classification_csv_path(record["position_path"]).exists()


def test_classify_auto_writes_sidecar(tmp_path):
    record = _write_position(tmp_path)
    written = pipeline.classify(
        [record], config=NlsConfig(enabled=True, image="0_input/NLS_zavg.tif", method="auto")
    )
    csv_path = nls_classification_csv_path(record["position_path"])
    assert written == [csv_path]
    assert csv_path.is_file()
    table = pd.read_csv(csv_path)
    assert set(table.columns) == {"id", "label"}
    by_id = dict(zip(table["id"], table["label"]))
    assert by_id[1] == "positive"
    assert by_id[2] == "negative"


def test_classify_fixed_threshold(tmp_path):
    record = _write_position(tmp_path)
    written = pipeline.classify(
        [record],
        config=NlsConfig(enabled=True, method="fixed", threshold=50.0),
    )
    table = pd.read_csv(written[0])
    by_id = dict(zip(table["id"], table["label"]))
    assert by_id[1] == "positive"
    assert by_id[2] == "negative"


def test_classify_otsu_writes_sidecar(tmp_path):
    record = _write_position(tmp_path)
    written = pipeline.classify(
        [record], config=NlsConfig(enabled=True, method="otsu")
    )
    assert written and written[0].is_file()


def test_classify_skips_position_without_marker_image(tmp_path):
    record = _write_position(tmp_path, with_nls=False)
    written = pipeline.classify([record], config=NlsConfig(enabled=True))
    assert written == []
    assert not nls_classification_csv_path(record["position_path"]).exists()


def test_classify_absolute_image_path(tmp_path):
    record = _write_position(tmp_path)
    abs_image = str(Path(record["position_path"]) / "0_input" / "NLS_zavg.tif")
    written = pipeline.classify(
        [record], config=NlsConfig(enabled=True, image=abs_image, method="auto")
    )
    assert written and written[0].is_file()


def test_classify_reports_progress(tmp_path):
    record = _write_position(tmp_path)
    calls: list[tuple[int, int, str]] = []
    pipeline.classify(
        [record],
        config=NlsConfig(enabled=True),
        progress_cb=lambda done, total, name: calls.append((done, total, name)),
    )
    assert calls == [(1, 1, "pos1")]
