import numpy as np
import tifffile

from cellflow.analysis.artifact_reader import PositionArtifactData, read_position_artifact
from cellflow.analysis.position_artifact import build_position_analysis_artifact


def _write_position(tmp_path, cell_stack, nucleus_stack):
    pos_dir = tmp_path / "position_0001"
    cell_dir = pos_dir / "cell"
    nucleus_dir = pos_dir / "nucleus"
    cell_dir.mkdir(parents=True)
    nucleus_dir.mkdir(parents=True)
    tifffile.imwrite(cell_dir / "tracked_labels.tif", cell_stack)
    tifffile.imwrite(nucleus_dir / "tracked_labels.tif", nucleus_stack)
    return pos_dir


def test_read_position_artifact_reconstructs_edges_and_centroids(tmp_path):
    frame = np.zeros((5, 6), dtype=np.uint16)
    frame[:, :3] = 1
    frame[:, 3:] = 2
    cell_stack = np.stack([frame, frame])
    pos_dir = _write_position(tmp_path, cell_stack, cell_stack.copy())
    output_path = build_position_analysis_artifact(pos_dir, tmp_path / "analysis.h5")

    artifact = read_position_artifact(output_path)

    assert isinstance(artifact, PositionArtifactData)
    assert set(artifact.cells) >= {"frame", "cell_id", "centroid_y", "centroid_x"}
    assert set(artifact.edges) >= {"frame", "coord_offset", "coord_count", "kind"}
    assert set(artifact.t1_events) >= {"t1_event_id", "frame", "edge_id"}

    centroids = artifact.centroid_points()
    assert centroids.shape == (4, 3)
    np.testing.assert_allclose(centroids[:, 0], [0, 0, 1, 1])

    lines = artifact.edge_lines()
    assert len(lines) == len(artifact.edges["frame"])
    assert all(line.shape[1] == 3 for line in lines)
    np.testing.assert_array_equal(lines[0][:, 0], np.full(len(lines[0]), artifact.edges["frame"][0]))
