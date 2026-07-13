import numpy as np
import tifffile

from cellflow.contact_analysis.contacts.reader import PositionContactAnalysis, read_position_contacts
from cellflow.contact_analysis.contacts.build import build_position_contacts


def _write_position(tmp_path, cell_stack, nucleus_stack):
    # The staged fallback consumes the committed base-folder labels.
    pos_dir = tmp_path / "position_0001"
    pos_dir.mkdir(parents=True)
    tifffile.imwrite(pos_dir / "cell_labels.tif", cell_stack)
    tifffile.imwrite(pos_dir / "nucleus_labels.tif", nucleus_stack)
    return pos_dir


def test_read_position_contacts_reconstructs_edges_and_centroids(tmp_path):
    frame = np.zeros((5, 6), dtype=np.uint16)
    frame[:, :3] = 1
    frame[:, 3:] = 2
    cell_stack = np.stack([frame, frame])
    pos_dir = _write_position(tmp_path, cell_stack, cell_stack.copy())
    output_path = build_position_contacts(pos_dir, tmp_path / "contact_analysis.h5")

    contact_analysis = read_position_contacts(output_path)

    assert isinstance(contact_analysis, PositionContactAnalysis)
    assert set(contact_analysis.cells) >= {"frame", "cell_id", "centroid_y", "centroid_x"}
    assert set(contact_analysis.edges) >= {"frame", "coord_offset", "coord_count", "kind"}
    assert set(contact_analysis.t1_events) >= {"t1_event_id", "frame", "edge_id"}

    assert contact_analysis.cell_tracked_labels_path == str(pos_dir / "cell_labels.tif")
    assert contact_analysis.nucleus_tracked_labels_path == str(pos_dir / "nucleus_labels.tif")
