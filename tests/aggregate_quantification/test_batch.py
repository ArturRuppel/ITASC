"""Name-based batch discovery + headless contact-analysis runs."""
from __future__ import annotations

import numpy as np
import tifffile

from cellflow.aggregate_quantification import (
    discover_contact_batch_jobs,
    run_contact_batch,
)


def _write_labels(path):
    labels = np.zeros((1, 4, 4), dtype=np.uint16)
    labels[0, :, :2] = 1
    labels[0, :, 2:] = 2
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, labels)


def test_discovery_groups_by_folder_and_pairs_colocated_nucleus(tmp_path):
    # posA has cell + nucleus; posB has cell only; a stray nucleus with no cell.
    _write_labels(tmp_path / "posA" / "cells.tif")
    _write_labels(tmp_path / "posA" / "nuc.tif")
    _write_labels(tmp_path / "posB" / "cells.tif")
    _write_labels(tmp_path / "orphan" / "nuc.tif")

    jobs = discover_contact_batch_jobs(
        tmp_path, cell_name="cells.tif", h5_name="contact.h5", nucleus_name="nuc.tif"
    )

    # One job per folder containing a cell-labels file; orphan nucleus ignored.
    assert [j.group_dir.name for j in jobs] == ["posA", "posB"]
    a, b = jobs
    assert a.nucleus_labels == tmp_path / "posA" / "nuc.tif"
    assert a.output == tmp_path / "posA" / "contact.h5"
    # No co-located nucleus -> cell-only.
    assert b.nucleus_labels is None


def test_discovery_associates_nucleus_across_subfolders_within_a_position(tmp_path):
    # Staged-style: cell and nucleus sit in different subfolders of one position.
    _write_labels(tmp_path / "pos01" / "3_cell" / "cell.tif")
    _write_labels(tmp_path / "pos01" / "2_nucleus" / "nuc.tif")
    _write_labels(tmp_path / "pos02" / "3_cell" / "cell.tif")  # no nucleus

    jobs = discover_contact_batch_jobs(
        tmp_path, cell_name="cell.tif", h5_name="contact.h5", nucleus_name="nuc.tif"
    )
    by = {j.group_dir.name: j for j in jobs}

    # pos01: nucleus paired across the sibling subfolder; output at the position.
    assert by["pos01"].nucleus_labels == tmp_path / "pos01" / "2_nucleus" / "nuc.tif"
    assert by["pos01"].output == tmp_path / "pos01" / "contact.h5"
    # pos02: nucleus search is bounded to its own position -> cell-only.
    assert by["pos02"].nucleus_labels is None
    assert by["pos02"].output == tmp_path / "pos02" / "contact.h5"


def test_discovery_ambiguous_nucleus_in_position_is_not_assigned(tmp_path):
    # Two nucleus-named files in one position -> can't associate -> cell-only.
    _write_labels(tmp_path / "pos01" / "3_cell" / "cell.tif")
    _write_labels(tmp_path / "pos01" / "2_nucleus" / "nuc.tif")
    _write_labels(tmp_path / "pos01" / "extra" / "nuc.tif")

    jobs = discover_contact_batch_jobs(
        tmp_path, cell_name="cell.tif", h5_name="contact.h5", nucleus_name="nuc.tif"
    )
    assert len(jobs) == 1
    assert jobs[0].nucleus_labels is None


def test_discovery_without_nucleus_name_is_cell_only(tmp_path):
    _write_labels(tmp_path / "posA" / "cells.tif")
    _write_labels(tmp_path / "posA" / "nuc.tif")

    jobs = discover_contact_batch_jobs(
        tmp_path, cell_name="cells.tif", h5_name="contact.h5"
    )
    assert len(jobs) == 1
    assert jobs[0].nucleus_labels is None


def test_run_builds_missing_skips_existing_and_overwrites(tmp_path):
    _write_labels(tmp_path / "posA" / "cells.tif")
    _write_labels(tmp_path / "posA" / "nuc.tif")
    _write_labels(tmp_path / "posB" / "cells.tif")

    jobs = discover_contact_batch_jobs(
        tmp_path, cell_name="cells.tif", h5_name="contact.h5", nucleus_name="nuc.tif"
    )

    seen: list[tuple[int, int, str]] = []
    results = run_contact_batch(jobs, progress_cb=lambda d, t, label: seen.append((d, t, label)))
    assert [r.status for r in results] == ["built", "built"]
    assert all(j.output.exists() for j in jobs)
    assert seen[-1][:2] == (2, 2)

    # Second run with everything present -> all skipped.
    again = run_contact_batch(jobs)
    assert [r.status for r in again] == ["skipped", "skipped"]

    # Overwrite forces a rebuild.
    forced = run_contact_batch(jobs, overwrite=True)
    assert [r.status for r in forced] == ["built", "built"]


def test_run_records_failure_without_aborting(tmp_path):
    _write_labels(tmp_path / "good" / "cells.tif")
    # A cell file that isn't a valid TIFF -> build raises, recorded as failed.
    bad = tmp_path / "bad" / "cells.tif"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a tiff")

    jobs = discover_contact_batch_jobs(
        tmp_path, cell_name="cells.tif", h5_name="contact.h5"
    )
    results = run_contact_batch(jobs)
    by_name = {r.job.group_dir.name: r for r in results}

    assert by_name["good"].status == "built"
    assert by_name["bad"].status == "failed"
    assert by_name["bad"].error
    # The good position still produced its output despite the bad one failing.
    assert (tmp_path / "good" / "contact.h5").exists()


def test_run_honors_cancel(tmp_path):
    for name in ("posA", "posB", "posC"):
        _write_labels(tmp_path / name / "cells.tif")
    jobs = discover_contact_batch_jobs(
        tmp_path, cell_name="cells.tif", h5_name="contact.h5"
    )
    # Cancel before any job runs -> no results, nothing built.
    results = run_contact_batch(jobs, cancel=lambda: True)
    assert results == []
    assert not any(j.output.exists() for j in jobs)
