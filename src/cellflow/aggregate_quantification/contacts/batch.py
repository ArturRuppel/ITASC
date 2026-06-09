"""Name-based batch discovery and headless contact-analysis builds.

The standalone contact piece is visualizer-first; the ``.h5`` it shows is a pure
derived artifact of the label inputs. This module lets users pre-compute that
artifact for many positions at once: name the three files (cell labels, optional
nucleus labels, output ``.h5``), point at one top-level folder, and every folder
that contains a cell-labels file becomes one job. A nucleus file is associated
only when it sits in that same folder; otherwise the job runs cell-only. Outputs
that already exist are skipped unless ``overwrite`` is set.

Position-agnostic and napari-free: shipped by the ``cellflow-aggregate`` wheel.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .build import ensure_contact_analysis

__all__ = [
    "ContactBatchJob",
    "ContactBatchResult",
    "discover_contact_batch_jobs",
    "run_contact_batch",
]


@dataclass(frozen=True)
class ContactBatchJob:
    """One discovered contact-analysis build: a cell-labels file and where its
    output goes, with an optional co-located nucleus-labels file."""

    group_dir: Path
    cell_labels: Path
    output: Path
    nucleus_labels: Path | None = None


@dataclass(frozen=True)
class ContactBatchResult:
    """Outcome of a single :class:`ContactBatchJob`.

    ``status`` is ``"built"`` (computed), ``"skipped"`` (output already present),
    or ``"failed"`` (``error`` holds the message).
    """

    job: ContactBatchJob
    status: str
    error: str | None = None


def _top_level_dir(path: Path, root: Path) -> Path:
    """The position folder a discovered file belongs to: the first directory
    component under ``root`` (or ``root`` itself for files directly in it)."""
    rel = path.relative_to(root)
    return root / rel.parts[0] if len(rel.parts) > 1 else root


def discover_contact_batch_jobs(
    root: str | Path,
    *,
    cell_name: str,
    h5_name: str,
    nucleus_name: str | None = None,
) -> list[ContactBatchJob]:
    """Find contact-analysis jobs under ``root`` by file name.

    Files are discovered recursively, then grouped by **position folder** — the
    first directory under ``root`` on the path to each match (``root`` itself for
    files lying directly in it). The named files of one position may live in
    different subfolders (e.g. ``pos01/3_cell/`` and ``pos01/2_nucleus/``); they
    are still grouped together.

    For a position with exactly one cell-labels file, that file is one job whose
    output is ``<position>/<h5_name>``; the nucleus is associated only when
    exactly one ``nucleus_name`` file exists anywhere in that position (zero or
    several → cell-only, "can't associate → don't assign"). A position holding
    several cell-labels files is ambiguous to group, so each falls back to a
    cell-only job written beside its own cell file. Jobs are sorted by cell path.
    """
    root = Path(root)
    cells = [c for c in root.rglob(cell_name) if c.is_file()]

    groups: dict[Path, list[Path]] = {}
    for cell in cells:
        groups.setdefault(_top_level_dir(cell, root), []).append(cell)

    jobs: list[ContactBatchJob] = []
    for position, position_cells in groups.items():
        if len(position_cells) == 1:
            cell = position_cells[0]
            nucleus: Path | None = None
            if nucleus_name:
                matches = [
                    n for n in position.rglob(nucleus_name) if n.is_file()
                ]
                if len(matches) == 1:
                    nucleus = matches[0]
            jobs.append(
                ContactBatchJob(
                    group_dir=position,
                    cell_labels=cell,
                    output=position / h5_name,
                    nucleus_labels=nucleus,
                )
            )
        else:
            # Several cell files share a position folder: too ambiguous to pair a
            # single nucleus or a single output, so keep them independent.
            for cell in position_cells:
                jobs.append(
                    ContactBatchJob(
                        group_dir=cell.parent,
                        cell_labels=cell,
                        output=cell.parent / h5_name,
                        nucleus_labels=None,
                    )
                )
    return sorted(jobs, key=lambda job: job.cell_labels)


def run_contact_batch(
    jobs: Iterable[ContactBatchJob],
    *,
    overwrite: bool = False,
    edge_extraction_params: dict | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> list[ContactBatchResult]:
    """Build the contact-analysis HDF5 for each job, collecting per-job outcomes.

    A single job's failure never aborts the run — it is recorded as a ``failed``
    result and the batch continues. ``cancel`` is polled before each job;
    ``progress_cb(done, total, label)`` is reported after each.
    """
    jobs = list(jobs)
    total = len(jobs)
    results: list[ContactBatchResult] = []
    for index, job in enumerate(jobs):
        if cancel is not None and cancel():
            break
        try:
            _, built = ensure_contact_analysis(
                cell_labels_path=job.cell_labels,
                output_path=job.output,
                nucleus_labels_path=job.nucleus_labels,
                overwrite=overwrite,
                edge_extraction_params=edge_extraction_params,
            )
            results.append(
                ContactBatchResult(job, "built" if built else "skipped")
            )
        except Exception as exc:  # noqa: BLE001 - one bad position must not abort
            results.append(ContactBatchResult(job, "failed", str(exc)))
        if progress_cb is not None:
            progress_cb(index + 1, total, job.group_dir.name)
    return results
