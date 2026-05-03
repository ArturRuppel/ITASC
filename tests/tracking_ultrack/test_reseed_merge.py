from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _merge_validated_into_export():
    ingest_stub = types.ModuleType("cellflow.tracking_ultrack.ingest")
    ingest_stub.ingest_hypotheses_to_db = lambda *args, **kwargs: None
    ingest_stub._build_ultrack_config = lambda *args, **kwargs: object()
    sys.modules.setdefault("cellflow.tracking_ultrack.ingest", ingest_stub)
    return importlib.import_module(
        "cellflow.tracking_ultrack.reseed"
    ).merge_validated_into_export


def test_merge_preserves_validated_ids_and_returns_empty_id_map():
    merge_validated_into_export = _merge_validated_into_export()
    exported = np.zeros((2, 12, 12), dtype=np.uint32)
    exported[0, 2:5, 2:5] = 99
    tracked = np.zeros_like(exported)
    tracked[0, 2:5, 2:5] = 7
    tracked[1, 3:6, 2:5] = 7

    result, id_map = merge_validated_into_export(exported, {7: {0, 1}}, tracked)

    assert id_map == {}
    assert np.all(result[0, 2:5, 2:5] == 7)
    assert np.all(result[1, 3:6, 2:5] == 7)


def test_merge_moves_solver_collisions_off_reserved_validated_ids():
    merge_validated_into_export = _merge_validated_into_export()
    exported = np.zeros((2, 12, 12), dtype=np.uint32)
    exported[1, 8:10, 8:10] = 7
    tracked = np.zeros_like(exported)
    tracked[0, 2:5, 2:5] = 7

    result, id_map = merge_validated_into_export(exported, {7: {0}}, tracked)

    assert id_map == {}
    assert np.all(result[0, 2:5, 2:5] == 7)
    assert np.all(result[1, 8:10, 8:10] == 8)
