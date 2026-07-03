"""Catalog of contact-analysis positions for Contact Analysis.

Builds, loads, and saves a catalog of positions (each a contact-analysis ``.h5``
plus optional cell/nucleus label images) discovered by file name / relative path
via :func:`discover_catalog_entries`.

Each row anchors on a ``position_path`` — the **absolute** path to the position
folder — and stores every file (the contact-analysis ``.h5``, cell labels,
nucleus labels) as a path **relative to that folder** (e.g.
``aggregate_quantification/contact_analysis.h5``). Persisting the position folder lets
downstream tools (notably the NLS classifier) resolve their own per-position
relative paths against it. Older catalogs that lack ``position_path`` still load:
their file paths are resolved relative to the CSV file's directory.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from collections.abc import Iterable

from .quantifier import OUTPUT_SUBDIR

STATUS_READY = "ready"
STATUS_INCOMPLETE = "incomplete"

#: The contacts ``.h5`` is a *derived* Contact Analysis output, not a
#: discovery input — every position's path is the fixed default location under
#: the shared :data:`OUTPUT_SUBDIR` folder.
CONTACT_ANALYSIS_RELPATH = f"{OUTPUT_SUBDIR}/contact_analysis.h5"

# Columns that identify a catalog row; validated on load. ``position_path``, the
# label-path and notes columns are optional so older catalogs (and hand-written
# ones) still load.
REQUIRED_CSV_COLUMNS = ("path", "date", "condition", "id")
# Full column order written on save: the absolute position folder, the
# contact-analysis path (relative to it), metadata, the two (optional)
# label-image paths (relative to it) needed to re-run analysis, and free-text
# notes.
CSV_COLUMNS = (
    "position_path",
    "path",
    "date",
    "condition",
    "experiment_id",
    "id",
    "cell_labels",
    "nucleus_labels",
    "notes",
)

#: The free-form metadata bag (``record["columns"]``) carries one entry per
#: folder-nesting level plus any manual constant tags. These canonical bag keys
#: have a dedicated CSV home (``position_id`` is the bag's name for the ``id``
#: column); every other bag key is an *extra* free-form column, appended to the
#: CSV header after :data:`CSV_COLUMNS`.
_BAG_TO_CSV = {
    "condition": "condition",
    "experiment_id": "experiment_id",
    "date": "date",
    "position_id": "id",
    "notes": "notes",
}
#: Flat recognized metadata fields handled explicitly by normalization — never
#: copied verbatim into the free-form bag (they map to canonical bag keys).
_RECOGNIZED_FLAT = frozenset({"condition", "experiment_id", "date", "notes"})
#: Structural / internal record keys that are never free-form metadata columns.
_NON_COLUMN_KEYS = frozenset(
    {
        "path",
        "position_path",
        "contact_analysis_path",
        "cell_labels",
        "nucleus_labels",
        "cell_tracked_labels_path",
        "nucleus_tracked_labels_path",
        "contact_analysis_status",
        "columns",
        "condition_id",
        "id",
        "position_id",
        "labels",  # legacy alias for notes
        "pixel_size_um",
        "time_interval_s",
    }
)


def relative_levels(root: Path | str, position_path: Path | str) -> tuple[str, ...]:
    """The folder-name segments from *root* (exclusive) to *position_path* (inclusive).

    These are the nesting *levels* a discovered position sits under; the UI names
    each level once and every position inherits its column values from its own
    segments (see :func:`columns_from_levels`).
    """
    root = _resolve_path(Path(root))
    pos = _resolve_path(Path(position_path))
    return pos.relative_to(root).parts


def columns_from_levels(
    level_names: Iterable[str | None], segments: Iterable[str]
) -> dict[str, str]:
    """Zip level *names* to folder-name *segments*; blank/``None`` names drop out.

    Extra segments without a paired name (or extra names without a segment) are
    ignored — only the overlap with a non-blank name becomes a column.
    """
    return {name: seg for name, seg in zip(level_names, segments) if name}


def discovered_level_depth(
    root: Path | str, position_paths: Iterable[Path | str]
) -> int | None:
    """The shared nesting depth of discovered positions, or ``None`` if it varies.

    Folder-derived columns anchor at the root (level 1 = first folder under root),
    so the level names line up across positions only when every position sits at
    the same depth. A ``None`` return tells the caller to warn rather than mislabel.
    """
    depths = {len(relative_levels(root, p)) for p in position_paths}
    return depths.pop() if len(depths) == 1 else None


def load_catalog(csv_path: Path | str) -> list[dict]:
    """Load CSV catalog records and expose normalized compatibility keys."""
    catalog_path = Path(csv_path)
    with catalog_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Catalog is missing required column(s): {missing_text}")

        records = [
            _normalize_catalog_record(row, base_dir=catalog_path.parent)
            for row in reader
        ]

    _check_unique_identity(records)
    return sorted(records, key=_catalog_sort_key)


def _identity_key(record: dict) -> tuple[str, str, str]:
    """The axes that must jointly identify a position: a collision here would
    silently pool two positions' cells under one group downstream."""
    return (
        str(record.get("experiment_id", "")),
        str(record.get("condition", "")),
        str(record.get("id", "")),
    )


def _check_unique_identity(records: Iterable[dict]) -> None:
    counts: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = _identity_key(record)
        counts[key] = counts.get(key, 0) + 1
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        listed = "; ".join(
            f"(experiment_id={e!r}, condition={c!r}, position_id={p!r})"
            for e, c, p in duplicates
        )
        raise ValueError(
            "Catalog has duplicate position identities that would silently merge "
            f"two positions' cells downstream: {listed}. Give each a distinct id."
        )


def save_catalog(csv_path: Path | str, records: Iterable[dict]) -> None:
    """Write catalog records to CSV.

    Each row stores the absolute ``position_path`` and the position's files as
    paths relative to that folder. Records without a known position folder
    (legacy / hand-made) fall back to paths relative to the CSV file.
    """
    catalog_path = Path(csv_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_records = [
        _normalize_catalog_record(record, base_dir=catalog_path.parent)
        for record in records
    ]
    # The header is the fixed core columns plus the union of every free-form
    # column (folder levels + manual tags) across records, in first-seen order.
    # Recognized bag keys map to existing core columns, so only the *extras*
    # extend the header — keeping legacy catalogs byte-identical.
    extras: list[str] = []
    seen: set[str] = set()
    for normalized in normalized_records:
        for key in normalized.get("columns", {}):
            if key in _BAG_TO_CSV or key in seen:
                continue
            seen.add(key)
            extras.append(key)
    fieldnames = list(CSV_COLUMNS) + extras

    with catalog_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for normalized in normalized_records:
            position_path = normalized.get("position_path")
            # Files are written relative to the position folder when one is
            # known; otherwise relative to the CSV file's directory.
            file_base = position_path if position_path is not None else catalog_path.parent
            columns = normalized.get("columns", {})
            row = {
                "position_path": str(position_path) if position_path is not None else "",
                "path": _path_for_csv(normalized["contact_analysis_path"], file_base),
                "date": normalized["date"],
                "condition": normalized["condition"],
                "experiment_id": normalized["experiment_id"],
                "id": normalized["id"],
                "cell_labels": _optional_path_for_csv(
                    normalized.get("cell_tracked_labels_path"), file_base
                ),
                "nucleus_labels": _optional_path_for_csv(
                    normalized.get("nucleus_tracked_labels_path"), file_base
                ),
                "notes": normalized["notes"],
            }
            row.update({key: columns[key] for key in extras if key in columns})
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

    # Absolute position folder; the anchor for this position's relative file
    # paths. Resolved against the CSV directory when given as a relative path.
    raw_position = normalized.get("position_path")
    position_path = (
        _resolve_with_base(raw_position, base_dir)
        if raw_position not in (None, "")
        else None
    )
    # File paths in a catalog row are relative to the position folder when one is
    # known; older catalogs (no position_path) stored them relative to the CSV.
    file_base = position_path if position_path is not None else base_dir

    contact_analysis_path = _resolve_with_base(
        normalized.get("path", normalized.get("contact_analysis_path", "")), file_base
    )

    # A record may carry its recognized axes flat (CSV load / legacy) or only in
    # an incoming free-form ``columns`` bag (the folder-derived UI path). Prefer
    # the flat field, fall back to the bag, then to the default.
    incoming_columns = normalized.get("columns") or {}

    def _meta(flat_key: str, bag_key: str, default: str) -> str:
        value = normalized.get(flat_key)
        if value in (None, ""):
            value = incoming_columns.get(bag_key)
        return str(value) if value not in (None, "") else default

    date = _meta("date", "date", "unknown_date")
    # The paired-replicate key. Its own annotation, *not* an alias for date;
    # defaults to date only when the catalog leaves it blank (single-replicate
    # experiments need not fill it). See the artifact-contract spec, §2.
    experiment_id = _meta("experiment_id", "experiment_id", date)
    condition = _meta(
        "condition",
        "condition",
        str(normalized.get("condition_id", "unknown_condition")),
    )
    source_id = _meta(
        "id",
        "position_id",
        str(normalized.get("position_id", contact_analysis_path.stem)),
    )
    # ``notes`` is the free-text field; accept the legacy ``labels`` column too.
    notes = str(normalized.get("notes", normalized.get("labels", "")))
    cell_labels_path = _resolve_optional_with_base(
        normalized.get("cell_tracked_labels_path", normalized.get("cell_labels")), file_base
    )
    nucleus_labels_path = _resolve_optional_with_base(
        normalized.get("nucleus_tracked_labels_path", normalized.get("nucleus_labels")),
        file_base,
    )

    columns = _columns_bag(normalized, condition, experiment_id, date, source_id)

    normalized.update({
        "path": contact_analysis_path,
        "date": date,
        "condition": condition,
        "id": source_id,
        "notes": notes,
        "condition_id": condition,
        "experiment_id": experiment_id,
        "position_id": source_id,
        "columns": columns,
        "position_path": position_path,
        "contact_analysis_path": contact_analysis_path,
        "cell_tracked_labels_path": cell_labels_path,
        "nucleus_tracked_labels_path": nucleus_labels_path,
        "contact_analysis_status": STATUS_READY if contact_analysis_path.is_file() else STATUS_INCOMPLETE,
    })
    return normalized


def _columns_bag(
    record: dict, condition: str, experiment_id: str, date: str, position_id: str
) -> dict[str, str]:
    """The canonical free-form metadata bag for a record.

    Holds one entry per folder-nesting level plus any manual constant tags. The
    recognized identity/pooling axes always appear under their canonical bag keys
    (``position_id`` for the ``id`` column); every other non-structural,
    non-``None`` field — whether a flat CSV column or an entry in an incoming
    ``columns`` dict — rides along as an extra column. ``notes`` is display-only
    and excluded (it never becomes a downstream metadata axis)."""
    bag: dict[str, str] = {}
    # Flat extras first (a CSV load surfaces extra columns as flat keys)…
    for key, value in record.items():
        if key in _NON_COLUMN_KEYS or key in _RECOGNIZED_FLAT or value is None:
            continue
        bag[key] = str(value)
    # …then an explicit incoming bag overrides (the UI / round-trip path).
    for key, value in (record.get("columns") or {}).items():
        if key == "notes":
            continue
        bag[key] = str(value)
    # Recognized axes win last so they always carry the normalized values.
    bag["condition"] = condition
    bag["experiment_id"] = experiment_id
    bag["date"] = date
    bag["position_id"] = position_id
    return bag


def discover_catalog_entries(
    root: Path | str,
    *,
    cell_name: str | None = None,
    nucleus_name: str | None = None,
) -> list[dict]:
    """Find catalog entries under *root* by file name / relative path.

    A *position* is any folder that contains at least one recognized **input** —
    cell labels and/or nucleus labels, each optional (*cell_name* / *nucleus_name*
    are a bare file name or a path relative to the position folder). A folder is
    registered once and carries whichever inputs it has; the others are ``None``.
    The contact-analysis ``.h5`` is a derived Contact Analysis **output**,
    not a discovery input: every position's contact path is the fixed default
    location (:data:`CONTACT_ANALYSIS_RELPATH`) and need not exist yet (it is built
    later by the contacts quantifier). The returned dicts carry the discovered
    paths but no metadata (date / condition / notes) — that is assigned before
    adding to the catalog.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    contact_rel = Path(CONTACT_ANALYSIS_RELPATH)
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
        str(record.get("date", "")),
        str(record.get("id", record.get("position_id", ""))),
    )
