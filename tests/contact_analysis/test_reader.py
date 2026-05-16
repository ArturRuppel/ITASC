import numpy as np
import tifffile

from cellflow.contact_analysis.reader import PositionContactAnalysis, read_position_contact_analysis
from cellflow.contact_analysis.build import build_position_contact_analysis


def _write_position(tmp_path, cell_stack, nucleus_stack):
    pos_dir = tmp_path / "position_0001"
    cell_dir = pos_dir / "cell"
    nucleus_dir = pos_dir / "nucleus"
    cell_dir.mkdir(parents=True)
    nucleus_dir.mkdir(parents=True)
    tifffile.imwrite(cell_dir / "tracked_labels.tif", cell_stack)
    tifffile.imwrite(nucleus_dir / "tracked_labels.tif", nucleus_stack)
    return pos_dir


def test_read_position_contact_analysis_reconstructs_edges_and_centroids(tmp_path):
    frame = np.zeros((5, 6), dtype=np.uint16)
    frame[:, :3] = 1
    frame[:, 3:] = 2
    cell_stack = np.stack([frame, frame])
    pos_dir = _write_position(tmp_path, cell_stack, cell_stack.copy())
    output_path = build_position_contact_analysis(pos_dir, tmp_path / "contact_analysis.h5")

    contact_analysis = read_position_contact_analysis(output_path)

    assert isinstance(contact_analysis, PositionContactAnalysis)
    assert set(contact_analysis.cells) >= {"frame", "cell_id", "centroid_y", "centroid_x"}
    assert set(contact_analysis.edges) >= {"frame", "coord_offset", "coord_count", "kind"}
    assert set(contact_analysis.t1_events) >= {"t1_event_id", "frame", "edge_id"}

    centroids = contact_analysis.centroid_points()
    assert centroids.shape == (4, 3)
    np.testing.assert_allclose(centroids[:, 0], [0, 0, 1, 1])
    assert contact_analysis.cell_tracked_labels_path == str(pos_dir / "cell" / "tracked_labels.tif")
    assert contact_analysis.nucleus_tracked_labels_path == str(pos_dir / "nucleus" / "tracked_labels.tif")

    lines = contact_analysis.edge_lines()
    assert len(lines) == len(contact_analysis.edges["frame"])
    assert all(line.shape[1] == 3 for line in lines)
    np.testing.assert_array_equal(lines[0][:, 0], np.full(len(lines[0]), contact_analysis.edges["frame"][0]))
