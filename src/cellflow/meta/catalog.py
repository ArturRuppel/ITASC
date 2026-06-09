"""Catalog of contact-analysis positions.

Builds, loads, and saves a catalog of positions (each a contact-analysis ``.h5``
plus optional cell/nucleus label images) discovered by file name / relative path
via :func:`discover_catalog_entries`.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from collections.abc import Iterable

STATUS_READY = "ready"
STATUS_INCOMPLETE = "incomplete"

# Columns that identify a catalog row; validated on load. The label-path and
# notes columns are optional so older catalogs (and hand-written ones) still load.
REQUIRED_CSV_COLUMNS = ("path", "date", "condition", "id")
# Full column order written on save: contact-analysis path, metadata, the two
# (optional) label-image paths needed to re-run analysis, and free-text notes.
CSV_COLUMNS = (
    "path",
    "date",
    "condition",
    "id",
    "cell_labels",
    "nucleus_labels",
    "notes",
)


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
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            normalized = _normalize_catalog_record(record, base_dir=catalog_path.parent)
            base = catalog_path.parent
            row = {
                "path": _path_for_csv(normalized["contact_analysis_path"], base),
                "date": normalized["date"],
                "condition": normalized["condition"],
                "id": normalized["id"],
                "cell_labels": _optional_path_for_csv(
                    normalized.get("cell_tracked_labels_path"), base
                ),
                "nucleus_labels": _optional_path_for_csv(
                    normalized.get("nucleus_tracked_labels_path"), base
                ),
                "notes": normalized["notes"],
            }
            writer.writerow(row)


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

    contact_analysis_path = _resolve_with_base(
        normalized.get("path", normalized.get("contact_analysis_path", "")), base_dir
    )

    date = str(normalized.get("date", normalized.get("experiment_id", "unknown_date")))
    condition = str(
        normalized.get("condition", normalized.get("condition_id", "unknown_condition"))
    )
    source_id = str(normalized.get("id", normalized.get("position_id", contact_analysis_path.stem)))
    # ``notes`` is the free-text field; accept the legacy ``labels`` column too.
    notes = str(normalized.get("notes", normalized.get("labels", "")))
    cell_labels_path = _resolve_optional_with_base(
        normalized.get("cell_tracked_labels_path", normalized.get("cell_labels")), base_dir
    )
    nucleus_labels_path = _resolve_optional_with_base(
        normalized.get("nucleus_tracked_labels_path", normalized.get("nucleus_labels")),
        base_dir,
    )

    normalized.update({
        "path": contact_analysis_path,
        "date": date,
        "condition": condition,
        "id": source_id,
        "notes": notes,
        "condition_id": condition,
        "experiment_id": date,
        "position_id": source_id,
        "contact_analysis_path": contact_analysis_path,
        "cell_tracked_labels_path": cell_labels_path,
        "nucleus_tracked_labels_path": nucleus_labels_path,
        "contact_analysis_status": STATUS_READY if contact_analysis_path.is_file() else STATUS_INCOMPLETE,
    })
    return normalized


def discover_catalog_entries(
    root: Path | str,
    *,
    cell_name: str | None = None,
    contact_name: str = "contact_analysis.h5",
    nucleus_name: str | None = None,
) -> list[dict]:
    """Find catalog entries under *root* by file name / relative path.

    A *position* is any folder that contains at least one recognized **input** —
    cell labels and/or nucleus labels, each optional (*cell_name* / *nucleus_name*
    are a bare file name or a path relative to the position folder). A folder is
    registered once and carries whichever inputs it has; the others are ``None``.
    The contact-analysis ``.h5`` is a derived **output**, not an anchor: its path
    is computed from *contact_name* relative to the position folder and need not
    exist yet (it is built later). The returned dicts carry the discovered paths
    but no metadata (date / condition / notes) — that is assigned before adding to
    the catalog.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    contact_rel = Path(contact_name or "contact_analysis.h5")
    # position_dir -> {input_key: resolved_path}, grouped so a folder with several
    # inputs becomes one entry.
    by_position: dict[Path, dict] = {}
    for key, name in (
        ("cell_tracked_labels_path", cell_name),
        ("nucleus_tracked_labels_path", nucleus_name),
    ):
        for position_dir, path in _discover_input(root, name):
            by_position.setdefault(position_dir, {})[key] = path

    entries: list[dict] = []
    for position_dir in sorted(by_position):
        entry = {
            "id": position_dir.name,
            "position_path": position_dir,
            "contact_analysis_path": _resolve_path(position_dir / contact_rel),
            "cell_tracked_labels_path": None,
            "nucleus_tracked_labels_path": None,
        }
        entry.update(by_position[position_dir])
        entries.append(entry)
    return entries


def _discover_input(root: Path, name: str | None):
    """Yield ``(position_dir, resolved_path)`` for each file matching *name*.

    *name* is a bare file name or a path relative to the position folder; the
    position folder is the file's parent with the relative parts stripped.
    """
    if not name:
        return
    rel = Path(name)
    for match in sorted(root.rglob(rel.name)):
        if not match.is_file():
            continue
        if len(rel.parts) > 1 and not _path_ends_with(match, rel):
            continue
        position_dir = match
        for _ in rel.parts:
            position_dir = position_dir.parent
        yield _resolve_path(position_dir), _resolve_path(match)


def _path_ends_with(path: Path, rel: Path) -> bool:
    rel_parts = rel.parts
    return len(path.parts) >= len(rel_parts) and path.parts[-len(rel_parts):] == rel_parts


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _resolve_with_base(raw: Path | str, base_dir: Path | None) -> Path:
    path = Path(raw)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return _resolve_path(path)


def _resolve_optional_with_base(
    value: Path | str | None, base_dir: Path | None
) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_with_base(value, base_dir)


def _path_for_csv(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path, start=base_dir)
    except ValueError:
        return str(path)


def _optional_path_for_csv(path: Path | str | None, base_dir: Path) -> str:
    if not path:
        return ""
    return _path_for_csv(Path(path), base_dir)


def _catalog_sort_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("condition", record.get("condition_id", ""))),
        str(record.get("date", record.get("experiment_id", ""))),
        str(record.get("id", record.get("position_id", ""))),
    )
