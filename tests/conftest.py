from __future__ import annotations

import pytest
import tifffile


@pytest.fixture(autouse=True)
def _write_test_tiffs_as_grayscale(monkeypatch):
    """Make synthetic test TIFFs explicit grayscale stacks by default."""
    original_imwrite = tifffile.imwrite

    def imwrite_grayscale_default(file, data=None, *args, **kwargs):
        kwargs.setdefault("photometric", "minisblack")
        return original_imwrite(file, data, *args, **kwargs)

    monkeypatch.setattr(tifffile, "imwrite", imwrite_grayscale_default)
