from __future__ import annotations

from pathlib import Path

from cellflow.napari._paths import NucleusArtifactPaths


def test_nucleus_atoms_path_under_2_nucleus():
    paths = NucleusArtifactPaths(pos_dir=Path("/data/pos00"))
    assert paths.nucleus_atoms == Path("/data/pos00/2_nucleus/atoms.tif")
