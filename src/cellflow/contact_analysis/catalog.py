"""Catalog of contact-analysis positions for Contact Analysis.

Builds, loads, and saves a catalog of positions (each a contact-analysis ``.h5``
plus optional cell/nucleus label images) discovered by file name / relative path
via :func:`discover_catalog_entries`.

Each row anchors on a ``position_path`` — the **absolute** path to the position
folder — and stores every file (the contact-analysis ``.h5``, cell labels,
nucleus labels) as a path **relative to that folder** (e.g.
``contact_analysis.h5``, beside the committed labels). Persisting the position folder lets
downstream tools (notably the NLS classifier) resolve their own per-position
relative paths against it. Older catalogs that lack ``position_path`` still load:
their file paths are resolved relative to the CSV file's directory.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from collections.abc import Iterable

STATUS_READY = "ready"
STATUS_INCOMPLETE = "incomplete"

#: The contacts ``.h5`` is a *derived* Contact Analysis output, not a discovery
#: input. It lives in the position base folder, beside the committed
#: ``cell_labels.tif`` / ``nucleus_labels.tif`` — one homogeneous layout for every
#: downstream-stable per-position artifact.
CONTACT_ANALYSIS_RELPATH = "contact_analysis.h5"

# The only column validated on load: ``position_path`` is the anchor every
# canonical artifact path is resolved against, so it is required. A position's
# *identity* is the combination of its classification columns (whatever the
# widget defined); that is validated for uniqueness at aggregate time, not here.
REQUIRED_CSV_COLUMNS = ("position_path",)
# The fixed structural columns written first: the absolute position folder, the
# canonical contact-analysis h5 (relative to it), and the two committed label
# images (relative). The classification columns — the widget columns, whatever
# they are named — follow, verbatim (see :func:`save_catalog`). No privileged
# ``condition`` / ``experiment_id`` / ``id`` columns: a position is identified by
# the combination of its own columns, not a fixed triple.
CSV_COLUMNS = (
    "position_path",
    "contact_analysis",
    "cell_labels",
    "nucleus_labels",
)

#: Structural / internal record keys that are never classification columns.
#: Everything else on a record — a flat CSV field or an entry in its ``columns``
#: bag — is a free-form classification column and rides through verbatim.
_STRUCTURAL_KEYS = frozenset(
    {
        "path",
        "position_path",
        "contact_analysis",
        "contact_analysis_path",
        "cell_labels",
        "nucleus_labels",
        "cell_tracked_labels_path",
        "nucleus_tracked_labels_path",
        "contact_analysis_status",
        "columns",
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
    """Load CSV catalog records and expose normalized compatibility keys.

    Only ``position_path`` is required. Uniqueness of the position *identity* (the
    combination of classification columns) is **not** checked here — a catalog may
    hold more positions than any one run aggregates; the aggregator validates
    uniqueness over the in-scope subset and refuses with an explanation there (see
    :func:`cellflow.contact_analysis.shape_tables.aggregate`).
    """
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

    return sorted(records, key=_catalog_sort_key)


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
    # The header is the fixed structural columns plus every classification column
    # (the widget columns — folder levels + manual tags), in first-seen order.
    # There are no privileged identity columns: whatever the widget shows is what
    # the CSV holds, one column per column.
    extras: list[str] = []
    seen: set[str] = set()
    for normalized in normalized_records:
        for key in normalized.get("columns", {}):
            if key in CSV_COLUMNS or key in seen:
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
                "contact_analysis": _path_for_csv(
                    normalized["contact_analysis_path"], file_base
                ),
                "cell_labels": _optional_path_for_csv(
                    normalized.get("cell_tracked_labels_path"), file_base
                ),
                "nucleus_labels": _optional_path_for_csv(
                    normalized.get("nucleus_tracked_labels_path"), file_base
                ),
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
    """Return a record with resolved paths and a normalized ``columns`` bag.

    The classification columns are carried verbatim in ``record["columns"]`` (the
    widget columns). There are no privileged identity fields: a record's identity
    is the combination of its columns, validated downstream at aggregate time.
    """
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

    # Read the canonical ``contact_analysis`` header, falling back to the legacy
    # ``contact_analysis_path`` / free-form ``path`` columns so older catalogs load.
    contact_analysis_path = _resolve_with_base(
        normalized.get(
            "contact_analysis",
            normalized.get("path", normalized.get("contact_analysis_path", "")),
        ),
        file_base,
    )
    cell_labels_path = _resolve_optional_with_base(
        normalized.get("cell_tracked_labels_path", normalized.get("cell_labels")), file_base
    )
    nucleus_labels_path = _resolve_optional_with_base(
        normalized.get("nucleus_tracked_labels_path", normalized.get("nucleus_labels")),
        file_base,
    )

    columns = _columns_bag(normalized)

    normalized.update({
        "path": contact_analysis_path,
        "columns": columns,
        "position_path": position_path,
        "contact_analysis_path": contact_analysis_path,
        "cell_tracked_labels_path": cell_labels_path,
        "nucleus_tracked_labels_path": nucleus_labels_path,
        "contact_analysis_status": STATUS_READY if contact_analysis_path.is_file() else STATUS_INCOMPLETE,
    })
    return normalized


def _columns_bag(record: dict) -> dict[str, str]:
    """The classification-column bag for a record — the widget columns, verbatim.

    Every non-structural field rides along as a column: a flat CSV field surfaced
    by ``csv.DictReader`` (legacy or hand-written catalogs) first, then an explicit
    incoming ``columns`` dict (the folder-derived UI path) overriding. The
    combination of these columns *is* the position's identity."""
    bag: dict[str, str] = {}
    # Flat columns first (a CSV load surfaces every column as a flat key)…
    for key, value in record.items():
        if key in _STRUCTURAL_KEYS or value is None:
            continue
        # ``id`` was the legacy privileged position column and is the pooled
        # row-id column downstream — a classification column may not use that
        # name, so carry its value under ``position_id`` (which is what it meant).
        # A real ``position_id`` column, if also present, wins (it comes later).
        bag["position_id" if key == "id" else key] = str(value)
    # …then an explicit incoming bag overrides (the UI / round-trip path).
    for key, value in (record.get("columns") or {}).items():
        bag[key] = str(value)
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


def _catalog_sort_key(record: dict) -> tuple[tuple[str, ...], str]:
    """Order by the classification columns (name-sorted values), then the position
    folder — a stable order that groups like-classified positions together
    regardless of what the columns are named."""
    columns = record.get("columns") or {}
    values = tuple(str(columns[key]) for key in sorted(columns))
    return (values, str(record.get("position_path") or ""))
