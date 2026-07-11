"""Qt-free bridge from catalogue *records* to a quantifier's per-position files.

The studio carries positions as normalized catalogue ``record`` dicts (see
:mod:`cellflow.contact_analysis.catalog`). Turning one into the
:class:`~cellflow.contact_analysis.quantifier.PositionInputs` a quantifier
builds from — and resolving where that quantifier's artifact lives — is pure path
logic with no Qt. It lives here (rather than in the napari studio layer) so the
headless aggregation backend (:mod:`.shape_tables`) and the standalone
``cellflow-aggregate`` wheel can resolve the same paths the studio does.

The napari ``studio_plugins`` module re-exports these for backwards compatibility.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields as _dataclass_fields
from pathlib import Path

from .quantifier import PositionInputs, Quantifier

__all__ = [
    "position_inputs_from_record",
    "output_for_record",
    "records_satisfying",
    "available_fields",
    "record_build_params",
]


def position_inputs_from_record(record: dict) -> PositionInputs:
    """Build :class:`PositionInputs` from a normalized catalogue record.

    ``pixel_size_um`` / ``time_interval_s`` are **global** build params: they are
    taken only from the value stamped on the record (the Parameters panel's
    px/Δt), never auto-resolved from a position's config or label TIFF tags.
    Absent ⇒ ``None``.
    """
    out = record.get("contact_analysis_path")
    cell = record.get("cell_tracked_labels_path")
    nucleus = record.get("nucleus_tracked_labels_path")
    position_dir = record.get("position_path") or (Path(out).parent if out else Path("."))
    cell_path = Path(cell) if cell else None
    nucleus_path = Path(nucleus) if nucleus else None
    return PositionInputs(
        position_dir=Path(position_dir),
        cell_labels_path=cell_path,
        nucleus_labels_path=nucleus_path,
        pixel_size_um=_positive_float(record.get("pixel_size_um")),
        time_interval_s=_positive_float(record.get("time_interval_s")),
        # The contacts artifact is a *produced* input — the contacts quantifier
        # writes it. The catalogue stamps its expected path on every position
        # whether or not it has been built yet, so gate on the file actually
        # existing: an unbuilt contacts product is not an available input.
        contact_analysis_path=Path(out) if out and Path(out).is_file() else None,
    )


def _positive_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def output_for_record(quantifier: Quantifier, record: dict) -> Path:
    """Where *quantifier*'s artifact lives for a catalogue *record*.

    Contacts keeps using the catalogue's explicit ``contact_analysis_path``
    column — it predates the quantifier seam and may hold a custom *nested* path
    that the per-position read/visualize paths also rely on, so building must
    target the same file. Every other quantifier derives its destination from
    :meth:`Quantifier.default_output`.
    """
    if quantifier.quantity_id == "contacts":
        explicit = record.get("contact_analysis_path")
        if explicit:
            return Path(explicit)
    return quantifier.default_output(position_inputs_from_record(record))


def records_satisfying(requires: Iterable[str], records: Iterable[dict]) -> list[dict]:
    """The records whose inputs supply every field in *requires*."""
    needed = tuple(requires)
    if not needed:
        return list(records)
    out = []
    for record in records:
        inputs = position_inputs_from_record(record)
        if all(getattr(inputs, name, None) is not None for name in needed):
            out.append(record)
    return out


def available_fields(inputs: PositionInputs) -> set[str]:
    """The populated (non-``None``) ``PositionInputs`` field names — the satisfied
    prerequisites a quantifier's ``requires`` is checked against."""
    return {f.name for f in _dataclass_fields(inputs) if getattr(inputs, f.name) is not None}


def record_build_params(
    quantifier: Quantifier, record: dict, params: Mapping[str, object] | None
) -> dict:
    """Shared *params* overlaid with the record's own required-build-param values.

    A param like pixel size can be set per-position on the record (the value the
    build actually reads via ``PositionInputs``) instead of in the shared bar, so
    the build-param gate must see both. The record's own value wins where present,
    mirroring ``run()``'s per-record stamping.
    """
    merged = dict(params or {})
    for key in quantifier.required_build_params:
        value = record.get(key)
        if value is not None:
            merged[key] = value
    return merged
