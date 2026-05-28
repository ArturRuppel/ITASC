from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_napari_import_stubs_and_close_viewers():
    tracked_roots = {
        "cellflow.napari",
        "cellflow.tracking_ultrack",
        "cellflow.segmentation",
    }
    tracked_prefixes = tuple(f"{name}." for name in tracked_roots)
    originals = {
        name: module
        for name, module in sys.modules.items()
        if name in tracked_roots or name.startswith(tracked_prefixes)
    }
    yield
    try:
        import napari

        napari.Viewer.close_all()
    except Exception:
        pass
    for name in list(sys.modules):
        if (name in tracked_roots or name.startswith(tracked_prefixes)) and name not in originals:
            sys.modules.pop(name, None)
    for name, module in originals.items():
        sys.modules[name] = module
