"""Name-based batch discovery and headless contact-analysis builds.

The standalone contact piece is visualizer-first; the ``.h5`` it shows is a pure
derived artifact of the label inputs. This module lets users pre-compute that
artifact for many positions at once: name the three files (cell labels, optional
nucleus labels, output ``.h5``), point at one top-level folder, and every folder
that contains a cell-labels file becomes one job. A nucleus file is associated
only when it sits in that same folder; otherwise the job runs cell-only. Outputs
that already exist are skipped unless ``overwrite`` is set.

Position-agnostic and napari-free: shipped by the ``cellflow-contact`` wheel.
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


def discover_contact_batch_jobs(
    root: str | Path,
    *,
    cell_name: str,
    h5_name: str,
    nucleus_name: str | None = None,
) -> list[ContactBatchJob]:
    """Recursively find contact-analysis jobs under ``root`` by file name.

    Every folder containing a file named ``cell_name`` becomes one job whose
    output is ``<folder>/<h5_name>``. When ``nucleus_name`` is given and a file
    of that name exists in the same folder, it is associated as the nucleus
    input; otherwise the job is cell-only. Nucleus-named files in folders without
    a cell-labels file are ignored. Jobs are returned sorted by path for
    deterministic ordering.
    """
    root = Path(root)
    jobs: list[ContactBatchJob] = []
    for cell in sorted(root.rglob(cell_name)):
        if not cell.is_file():
            continue
        group_dir = cell.parent
        nucleus: Path | None = None
        if nucleus_name:
            candidate = group_dir / nucleus_name
            nucleus = candidate if candidate.is_file() else None
        jobs.append(
            ContactBatchJob(
                group_dir=group_dir,
                cell_labels=cell,
                output=group_dir / h5_name,
                nucleus_labels=nucleus,
            )
        )
    return jobs


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
