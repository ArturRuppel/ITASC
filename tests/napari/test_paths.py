from __future__ import annotations

from pathlib import Path

from itasc.napari._paths import NucleusArtifactPaths


def test_nucleus_atoms_path_under_2_nucleus():
    paths = NucleusArtifactPaths(pos_dir=Path("/data/pos00"))
    assert paths.nucleus_atoms == Path("/data/pos00/2_nucleus/atoms.tif")


def test_final_output_label_paths_live_in_position_base_folder(tmp_path):
    from itasc.napari._paths import NucleusArtifactPaths

    p = NucleusArtifactPaths(tmp_path)
    assert p.nucleus_labels == tmp_path / "nucleus_labels.tif"
    assert p.cell_labels == tmp_path / "cell_labels.tif"
    # Final outputs sit beside the numbered stage dirs, not inside them.
    assert p.nucleus_labels.parent == tmp_path
    assert p.tracked == tmp_path / "2_nucleus" / "tracked_labels.tif"
