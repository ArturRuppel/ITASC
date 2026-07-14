import h5py
import numpy as np
import pytest
import tifffile

from itasc.contact_analysis.contacts.build import (
    build_contacts,
    build_position_contacts,
    ensure_contacts,
    assign_persistent_edge_ids,
    _coordinate_segments,
    _extract_frame_cell_edges,
    _order_coordinates,
)


def _write_cell_labels(path):
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, :, :2] = 1
    labels[0, :, 2:] = 2
    tifffile.imwrite(path, labels)


def test_ensure_contacts_builds_when_missing_and_skips_when_present(tmp_path):
    cell_path = tmp_path / "cells.tif"
    _write_cell_labels(cell_path)
    out = tmp_path / "contact.h5"

    path, built = ensure_contacts(cell_labels_path=cell_path, output_path=out)
    assert built is True
    assert path == out and out.exists()
    mtime = out.stat().st_mtime_ns

    # Second call must not rebuild (missing-only policy).
    path2, built2 = ensure_contacts(cell_labels_path=cell_path, output_path=out)
    assert built2 is False
    assert out.stat().st_mtime_ns == mtime


def test_ensure_contacts_overwrite_rebuilds(tmp_path):
    cell_path = tmp_path / "cells.tif"
    _write_cell_labels(cell_path)
    out = tmp_path / "contact.h5"
    out.write_bytes(b"stale")

    path, built = ensure_contacts(
        cell_labels_path=cell_path, output_path=out, overwrite=True
    )
    assert built is True
    # A real HDF5 now, not the stale sentinel.
    with h5py.File(out, "r") as h5:
        assert h5["cells/table/cell_id"][:].tolist() == [1, 2]


def test_build_contacts_is_position_agnostic_with_explicit_paths(tmp_path):
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, :, :2] = 1
    labels[0, :, 2:] = 2
    cell_path = tmp_path / "my_cells.tif"
    tifffile.imwrite(cell_path, labels)
    output_path = tmp_path / "out" / "contact.h5"

    result = build_contacts(cell_labels_path=cell_path, output_path=output_path)

    assert result == output_path
    with h5py.File(output_path, "r") as h5:
        # No nucleus supplied: provenance records empty nucleus/source paths.
        assert h5["provenance"].attrs["cell_tracked_labels_path"] == str(cell_path)
        assert h5["provenance"].attrs["nucleus_tracked_labels_path"] == ""
        assert h5["provenance"].attrs["source_position_path"] == ""
        assert h5["cells/table/cell_id"][:].tolist() == [1, 2]


def test_build_contacts_optional_nucleus_skips_identity_check(tmp_path):
    cell = np.zeros((1, 4, 4), dtype=np.uint16)
    cell[0, :, :2] = 1
    cell[0, :, 2:] = 2
    # A nucleus stack that would FAIL the identity invariant is irrelevant when
    # nucleus_labels_path is omitted.
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, cell)

    # Should not raise despite no nucleus provided.
    build_contacts(
        cell_labels_path=cell_path,
        output_path=tmp_path / "ok.h5",
        source_path=tmp_path / "some_position",
    )


def _write_position(tmp_path, cell_stack, nucleus_stack):
    # The staged fallback consumes the *committed* labels in the position base
    # folder (cell_labels.tif / nucleus_labels.tif), not the pre-commit working
    # 3_cell / 2_nucleus tracked labels.
    pos_dir = tmp_path / "position_0001"
    pos_dir.mkdir(parents=True)
    tifffile.imwrite(pos_dir / "cell_labels.tif", cell_stack)
    tifffile.imwrite(pos_dir / "nucleus_labels.tif", nucleus_stack)
    return pos_dir


def test_build_position_contacts_writes_schema_and_references_label_paths(tmp_path):
    frame = np.zeros((5, 6), dtype=np.uint16)
    frame[:, :3] = 1
    frame[:, 3:] = 2
    cell_stack = np.stack([frame, frame])
    pos_dir = _write_position(tmp_path, cell_stack, cell_stack.copy())
    output_path = tmp_path / "contact_analysis.h5"

    build_position_contacts(pos_dir, output_path)

    with h5py.File(output_path, "r") as h5:
        assert h5["provenance"].attrs["source_position_path"] == str(pos_dir)
        assert h5["provenance"].attrs["cell_tracked_labels_path"] == str(
            pos_dir / "cell_labels.tif"
        )
        assert h5["provenance"].attrs["nucleus_tracked_labels_path"] == str(
            pos_dir / "nucleus_labels.tif"
        )

        cells = h5["cells/table"]
        assert list(cells.keys()) == [
            "frame",
            "cell_id",
            "area",
            "centroid_y",
            "centroid_x",
            "perimeter",
            "bbox_min_y",
            "bbox_min_x",
            "bbox_max_y",
            "bbox_max_x",
        ]
        assert cells["frame"][:].tolist() == [0, 0, 1, 1]
        assert cells["cell_id"][:].tolist() == [1, 2, 1, 2]
        # The subpopulation label is no longer carried in the H5 (sidecar CSV now).
        assert "class_label" not in cells

        edges = h5["edges/table"]
        assert set(edges.keys()) == {
            "frame",
            "edge_id",
            "cell_a",
            "cell_b",
            "kind",
            "edge_label",
            "is_t1_frame",
            "t1_event_id",
            "length",
            "midpoint_y",
            "midpoint_x",
            "coord_offset",
            "coord_count",
        }
        assert "cell_cell" in edges["kind"].asstr()[:].tolist()
        assert "border" in edges["kind"].asstr()[:].tolist()
        assert len(h5["edges/coordinates/y"]) == len(h5["edges/coordinates/x"])
        assert int(edges["coord_count"][:].sum()) == len(h5["edges/coordinates/y"])
        assert h5["edges/table/t1_event_id"].attrs["null_sentinel"] == -1

        assert "cells/measurements" in h5
        assert "edges/measurements" in h5
        assert len(h5["t1_events/table/t1_event_id"]) == 0


def test_build_position_contacts_accepts_itasc_v2_label_paths(tmp_path):
    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, :, :2] = 1
    labels[0, :, 2:] = 2
    cell_path = pos_dir / "3_cell" / "tracked_labels.tif"
    nucleus_path = pos_dir / "2_nucleus" / "tracked_labels.tif"
    tifffile.imwrite(cell_path, labels)
    tifffile.imwrite(nucleus_path, labels)

    output_path = build_position_contacts(
        pos_dir,
        tmp_path / "analysis.h5",
        cell_tracked_labels_path=cell_path,
        nucleus_tracked_labels_path=nucleus_path,
    )

    with h5py.File(output_path, "r") as h5:
        assert h5["provenance"].attrs["cell_tracked_labels_path"] == str(cell_path)
        assert h5["provenance"].attrs["nucleus_tracked_labels_path"] == str(nucleus_path)


def test_build_position_contacts_reports_progress_in_order(tmp_path):
    frame = np.zeros((4, 5), dtype=np.uint16)
    frame[:, :2] = 1
    frame[:, 2:] = 2
    cell_stack = np.stack([frame, frame])
    pos_dir = _write_position(tmp_path, cell_stack, cell_stack.copy())
    output_path = tmp_path / "analysis.h5"
    progress = []

    def record(done, total, message):
        progress.append((done, total, message))

    build_position_contacts(pos_dir, output_path, progress_cb=record)

    assert progress == [
        (1, 6, "read labels"),
        (2, 6, "validate IDs"),
        (3, 6, "extract cells"),
        (4, 6, "extract edges"),
        (5, 6, "assign edge IDs/T1"),
        (6, 6, "write HDF5"),
    ]


def test_build_position_contacts_rejects_cell_nucleus_identity_mismatch(tmp_path):
    cell = np.zeros((1, 4, 4), dtype=np.uint16)
    cell[0, :, :2] = 1
    nucleus = cell.copy()
    nucleus[nucleus == 1] = 7
    pos_dir = _write_position(tmp_path, cell, nucleus)

    with pytest.raises(ValueError) as exc_info:
        build_position_contacts(pos_dir, tmp_path / "bad.h5")

    message = str(exc_info.value)
    assert "frame 0" in message
    assert "cell labels present only in cell stack: [1]" in message
    assert "nucleus labels present only in nucleus stack: [7]" in message


def test_extract_frame_cell_edges_splits_discontinuous_segments_for_same_cell_pair():
    frame = np.asarray(
        [
            [1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0],
            [1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2],
        ],
        dtype=np.uint16,
    )

    records = _extract_frame_cell_edges(frame, frame_idx=0)

    assert [record.pair for record in records] == [(1, 2), (1, 2), (1, 2), (1, 2)]
    assert [len(record.coordinates) for record in records] == [2, 2, 2, 2]
    for record in records:
        jumps = np.linalg.norm(np.diff(record.coordinates, axis=0), axis=1)
        assert np.all(jumps <= 1.0)


def test_extract_frame_cell_edges_keeps_non_adjacent_crossings():
    # Cells 1 and 2 touch only at rows 0 and 4 (non-adjacent single points);
    # cells 1 and 3 form a connected run in between. The (1, 2) contact must
    # not be silently dropped just because its two crossings are not adjacent.
    frame = np.asarray(
        [
            [1, 2],
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 2],
        ],
        dtype=np.uint16,
    )

    records = _extract_frame_cell_edges(frame, frame_idx=0)
    pairs = [record.pair for record in records]

    assert (1, 2) in pairs
    assert pairs.count((1, 2)) == 2  # two distinct, isolated crossings
    assert (1, 3) in pairs


def test_coordinate_segments_emits_isolated_points_as_own_segments():
    # Two points too far apart to be neighbors: previously the edge-walk loop
    # never ran and every point was dropped. Each isolated point must survive
    # as its own single-point segment.
    coords = np.asarray([[0.0, 0.5], [4.0, 0.5]], dtype=float)

    segments = _coordinate_segments(coords)

    assert len(segments) == 2
    assert all(len(segment) == 1 for segment in segments)
    surviving = {tuple(segment[0]) for segment in segments}
    assert surviving == {(0.0, 0.5), (4.0, 0.5)}


def test_order_coordinates_starts_open_segments_at_true_endpoint():
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=float,
    )

    ordered = _order_coordinates(coords)

    jumps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    assert np.all(jumps <= 1.0)
    assert ordered[0].tolist() in ([2.0, 0.0], [0.0, 2.0])
    assert ordered[-1].tolist() in ([2.0, 0.0], [0.0, 2.0])


def test_coordinate_segments_split_branched_components_into_jump_free_paths():
    coords = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

    segments = _coordinate_segments(coords)

    assert len(segments) > 1
    for segment in segments:
        jumps = np.linalg.norm(np.diff(segment, axis=0), axis=1)
        assert np.all(jumps <= 1.0)


def test_assign_persistent_edge_ids_links_losing_and_gaining_pairs_through_t1():
    frames = [
        {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)},
        {(3, 4), (1, 3), (1, 4), (2, 3), (2, 4)},
    ]

    assignments, events = assign_persistent_edge_ids(frames)

    assert len(events) == 1
    event = events[0]
    assert event.losing_pair == (1, 2)
    assert event.gaining_pair == (3, 4)
    assert assignments[0][(1, 2)] == assignments[1][(3, 4)]
