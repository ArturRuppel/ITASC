"""Qt-free bridge from catalogue *records* to a quantifier's per-position files.

The studio carries positions as normalized catalogue ``record`` dicts (see
:mod:`itasc.contact_analysis.catalog`). Turning one into the
:class:`~itasc.contact_analysis.quantifier.PositionInputs` a quantifier
builds from — and resolving where that quantifier's artifact lives — is pure path
logic with no Qt. It lives here (rather than in the napari studio layer) so the
headless aggregation backend (:mod:`.shape_tables`) and the standalone
``itasc-aggregate`` wheel can resolve the same paths the pipeline does.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields as _dataclass_fields
from pathlib import Path

from .quantifier import PositionInputs, Quantifier, available_quantifiers

__all__ = [
    "position_inputs_from_record",
    "output_for_record",
    "available_fields",
    "record_build_params",
    "supported_quantities",
]


def position_inputs_from_record(
    record: dict, params: Mapping[str, object] | None = None
) -> PositionInputs:
    """Build :class:`PositionInputs` from a normalized catalogue record.

    ``pixel_size_um`` / ``time_interval_s`` are **global** build params: they are
    read from the value stamped on the record (the Parameters panel's px/Δt),
    falling back to the shared *params* mapping when the record omits them.
    Threading *params* here keeps the value the build-param gate sees (via
    :func:`record_build_params`, which also overlays *params*) identical to the
    value ``compute_object_table`` actually receives through ``inputs`` — without
    it, a ``params``-only pixel size passes the gate but reaches the shape/dynamics
    cores as ``None``. Absent from both ⇒ ``None``.
    """
    out = record.get("contact_analysis_path")
    cell = record.get("cell_tracked_labels_path")
    nucleus = record.get("nucleus_tracked_labels_path")
    position_dir = record.get("position_path") or (Path(out).parent if out else Path("."))
    # Gate every input on the file actually existing, not merely on a path string
    # being present. The catalog stamps a position's *expected* label paths
    # (``cell_labels.tif`` / ``nucleus_labels.tif``) whether or not they have been
    # produced, so a nucleus-only position still carries a ``cell_tracked_labels_path``
    # pointing at a missing file. Treating that as an available input lets
    # cell_shape / cell_dynamics pass their ``requires`` gate and then crash reading
    # a file that isn't there. An input you cannot read is not available — the same
    # rule already applied to the produced ``contact_analysis_path`` below.
    cell_path = Path(cell) if cell and Path(cell).is_file() else None
    nucleus_path = Path(nucleus) if nucleus and Path(nucleus).is_file() else None
    return PositionInputs(
        position_dir=Path(position_dir),
        cell_labels_path=cell_path,
        nucleus_labels_path=nucleus_path,
        pixel_size_um=_positive_float(_record_or_param(record, params, "pixel_size_um")),
        time_interval_s=_positive_float(_record_or_param(record, params, "time_interval_s")),
        # The contacts artifact is a *produced* input — the contacts quantifier
        # writes it. The catalogue stamps its expected path on every position
        # whether or not it has been built yet, so gate on the file actually
        # existing: an unbuilt contacts product is not an available input.
        contact_analysis_path=Path(out) if out and Path(out).is_file() else None,
    )


def _record_or_param(
    record: dict, params: Mapping[str, object] | None, key: str
) -> object:
    """The record's value for *key*, falling back to shared *params*.

    Record wins where present, mirroring ``run()``'s per-record stamping and
    :func:`record_build_params`' gate precedence.
    """
    value = record.get(key)
    if value is None and params is not None:
        value = params.get(key)
    return value


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


def supported_quantities(
    records: Sequence[dict], *, params: Mapping[str, object] | None = None
) -> set[str]:
    """``quantity_id``\\ s of the pooled quantifiers *records* can actually produce.

    A pooled quantifier (one declaring ``table_keys``) is *supported* when at least
    one record satisfies **both** gates :func:`shape_tables.build_table` applies per
    position: its ``requires`` inputs are present (:meth:`Quantifier.can_build`) and
    its ``required_build_params`` are satisfied (pixel size, FOV area, …). That makes
    "supported" ⟺ "this table would pool at least one non-empty row" — the exact
    predicate the Aggregate UI greys its checkboxes against, so an enabled box never
    promises a table the run would silently skip. Producers (no ``table_keys``, e.g.
    ``contacts``) are never pooled and so never reported.
    """
    supported: set[str] = set()
    for quantifier in (cls() for cls in available_quantifiers()):
        if not quantifier.table_keys:
            continue
        for record in records:
            inputs = position_inputs_from_record(record, params)
            if quantifier.can_build(inputs) and not quantifier.missing_build_params(
                record_build_params(quantifier, record, params)
            ):
                supported.add(quantifier.quantity_id)
                break
    return supported
