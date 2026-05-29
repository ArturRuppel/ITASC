"""Study directory discovery.

Provides ``discover_study(root)`` which scans a root directory for
condition/experiment/position trees and returns a sorted list of records
describing each position and its contact-analysis readiness.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from collections.abc import Iterable

CONTACT_ANALYSIS_REL = Path("4_contact_analysis") / "contact_analysis.h5"
NUCLEUS_LABELS_REL = Path("2_nucleus") / "tracked_labels.tif"
CELL_LABELS_REL = Path("3_cell") / "tracked_labels.tif"

STATUS_READY = "ready"
STATUS_INCOMPLETE = "incomplete"

REQUIRED_CSV_COLUMNS = ("path", "date", "condition", "id", "labels")


def discover_study(root: Path) -> list[dict]:
    """Scan *root* for ``condition/experiment/position`` trees.

    Returns a list of dicts sorted by ``condition_id``, ``experiment_id``,
    ``position_id``.  Each dict contains the keys:

    * ``condition_id``
    * ``experiment_id``
    * ``position_id``
    * ``position_path`` (:class:`~pathlib.Path`)
    * ``contact_analysis_path`` (:class:`~pathlib.Path`)
    * ``nucleus_tracked_labels_path`` (:class:`~pathlib.Path`)
    * ``cell_tracked_labels_path`` (:class:`~pathlib.Path`)
    * ``contact_analysis_status`` --- ``"ready"`` when all three files exist,
      otherwise ``"incomplete"``.
    """
    records: list[dict] = []

    if not root.is_dir():
        return records

    for cond_path in sorted(root.iterdir()):
        if not cond_path.is_dir():
            continue

        for exp_path in sorted(cond_path.iterdir()):
            if not exp_path.is_dir():
                continue

            for pos_path in sorted(exp_path.iterdir()):
                if not pos_path.is_dir():
                    continue

                contact_analysis_path = pos_path / CONTACT_ANALYSIS_REL
                nucleus_path = pos_path / NUCLEUS_LABELS_REL
                cell_path = pos_path / CELL_LABELS_REL

                all_exist = (
                    contact_analysis_path.is_file()
                    and nucleus_path.is_file()
                    and cell_path.is_file()
                )

                records.append({
                    "condition_id": cond_path.name,
                    "experiment_id": exp_path.name,
                    "position_id": pos_path.name,
                    "position_path": pos_path,
                    "contact_analysis_path": contact_analysis_path,
                    "nucleus_tracked_labels_path": nucleus_path,
                    "cell_tracked_labels_path": cell_path,
                    "contact_analysis_status": STATUS_READY if all_exist else STATUS_INCOMPLETE,
                })

    return records


def load_meta_catalog(csv_path: Path | str) -> list[dict]:
    """Load CSV catalog records and expose meta-browser compatibility keys."""
    catalog_path = Path(csv_path)
    with catalog_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Meta catalog is missing required column(s): {missing_text}")

        records = [
            _normalize_catalog_record(row, base_dir=catalog_path.parent)
            for row in reader
        ]

    return sorted(records, key=_catalog_sort_key)


def save_meta_catalog(csv_path: Path | str, records: Iterable[dict]) -> None:
    """Write catalog records to CSV using paths relative to the CSV when possible."""
    catalog_path = Path(csv_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    with catalog_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            normalized = _normalize_catalog_record(record, base_dir=catalog_path.parent)
            row = {
                "path": _path_for_csv(normalized["contact_analysis_path"], catalog_path.parent),
                "date": normalized["date"],
                "condition": normalized["condition"],
                "id": normalized["id"],
                "labels": normalized["labels"],
            }
            writer.writerow(row)


def discover_h5_files(folder: Path | str, recursive: bool = True) -> list[Path]:
    """Return sorted H5 files from *folder*."""
    root = Path(folder)
    if not root.is_dir():
        return []

    patterns = ("*.h5", "*.hdf5")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def records_from_h5_paths(
    paths: Iterable[Path | str],
    defaults: dict | None = None,
) -> list[dict]:
    """Create conservative catalog records from explicit H5 paths."""
    defaults = defaults or {}
    resolved_paths = [_resolve_path(Path(path)) for path in paths]
    ids = _ids_for_h5_paths(resolved_paths)
    records = []
    for path, source_id in zip(resolved_paths, ids):
        record = {
            "path": path,
            "date": str(defaults.get("date", "unknown_date")),
            "condition": str(defaults.get("condition", "unknown_condition")),
            "id": str(defaults.get("id", source_id)),
            "labels": str(defaults.get("labels", "")),
        }
        records.append(_normalize_catalog_record(record))
    return sorted(records, key=_catalog_sort_key)


def merge_catalog_records(existing: Iterable[dict], incoming: Iterable[dict]) -> list[dict]:
    """Append incoming records while skipping duplicate resolved H5 paths."""
    merged: list[dict] = []
    seen: set[Path] = set()

    for record in list(existing) + list(incoming):
        normalized = _normalize_catalog_record(record)
        contact_analysis_path = normalized["contact_analysis_path"]
        resolved = _resolve_path(contact_analysis_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        merged.append(normalized)

    return merged


def _normalize_catalog_record(record: dict, base_dir: Path | None = None) -> dict:
    """Return a record with required CSV fields and widget compatibility keys."""
    normalized = dict(record)

    raw_path = normalized.get("path", normalized.get("contact_analysis_path", ""))
    contact_analysis_path = Path(raw_path)
    if base_dir is not None and not contact_analysis_path.is_absolute():
        contact_analysis_path = base_dir / contact_analysis_path
    contact_analysis_path = _resolve_path(contact_analysis_path)

    date = str(normalized.get("date", normalized.get("experiment_id", "unknown_date")))
    condition = str(
        normalized.get("condition", normalized.get("condition_id", "unknown_condition"))
    )
    source_id = str(normalized.get("id", normalized.get("position_id", contact_analysis_path.stem)))
    labels = str(normalized.get("labels", ""))

    normalized.update({
        "path": contact_analysis_path,
        "date": date,
        "condition": condition,
        "id": source_id,
        "labels": labels,
        "condition_id": condition,
        "experiment_id": date,
        "position_id": source_id,
        "contact_analysis_path": contact_analysis_path,
        "contact_analysis_status": STATUS_READY if contact_analysis_path.is_file() else STATUS_INCOMPLETE,
    })
    return normalized


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _path_for_csv(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path, start=base_dir)
    except ValueError:
        return str(path)


def _catalog_sort_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("condition", record.get("condition_id", ""))),
        str(record.get("date", record.get("experiment_id", ""))),
        str(record.get("id", record.get("position_id", ""))),
    )


def _ids_for_h5_paths(paths: list[Path]) -> list[str]:
    stems = [path.stem for path in paths]
    duplicate_stems = {stem for stem in stems if stems.count(stem) > 1}
    ids: list[str] = []
    used: set[str] = set()
    for path in paths:
        parent_name = (
            path.parent.parent.name
            if path.parent.name == "4_contact_analysis"
            else path.parent.name
        )
        source_id = (
            parent_name
            if path.stem in duplicate_stems or parent_name.lower().startswith("pos")
            else path.stem
        )
        unique_id = source_id
        suffix = 2
        while unique_id in used:
            unique_id = f"{source_id}-{suffix}"
            suffix += 1
        used.add(unique_id)
        ids.append(unique_id)
    return ids
