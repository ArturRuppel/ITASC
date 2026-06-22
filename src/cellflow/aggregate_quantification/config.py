"""The TOML *run-config* — Aggregate Quantification's hand-authored knob file.

One small file, authored once and git-versioned (it carries *code status*, per the
artifact-contract spec §1): it names the per-position **catalog** CSV, selects
which **quantities** to compute, supplies the shared build **params**, and points
at an optional **curation** table. Everything in it is a run-level choice — the
per-position table itself stays a CSV (tabular, many-row, with its own
relative-path resolution); this file is the "author once, then run" surface a
single ``run(config)`` entry point consumes.

The run produces **label-agnostic** tidy tables only; no classification step and
no plot/figure export live here (a subpopulation classification and any plots are a
downstream, dataset-specific concern).

Paths resolve relative to the config file's own directory, so a project folder is
relocatable: keep ``config.toml`` next to ``catalog.csv`` and the whole thing moves
together.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

from .quantifier import available_quantifiers

__all__ = ["RunConfig", "load_config", "write_config"]

#: Default for the optional curation-table key, relative to the config dir.
_DEFAULT_CURATION = "curation.csv"


@dataclass(frozen=True)
class RunConfig:
    """A parsed run-config. Paths are absolute (resolved against the config dir).

    *quantities* empty means "every available quantifier"; a non-empty tuple
    selects a subset by ``quantity_id`` (dependency producers are pulled in at run
    time even when omitted). *params* is the shared build-knob mapping threaded to
    quantifiers that opt in. *out_dir* is where the pooled tidy tables land (flat);
    ``None`` defaults to the catalogue root (the positions' common ancestor).
    *curation* names an optional exclusion table authored by the curation tool for
    downstream consumers; the aggregate step itself writes all rows.
    """

    catalog: Path
    out_dir: Path | None = None
    curation: Path = field(default=Path(_DEFAULT_CURATION))
    quantities: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)


def load_config(config_path: Path | str) -> RunConfig:
    """Parse the TOML run-config at *config_path* into a :class:`RunConfig`.

    ``catalog`` is required. Relative paths resolve against the config file's
    directory. Selected ``quantities`` are validated against the registered
    quantifiers so a typo fails loudly rather than silently computing nothing.
    """
    path = Path(config_path)
    base = path.parent
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    if "catalog" not in data:
        raise ValueError(
            f"Run-config {path.name!r} is missing the required 'catalog' key "
            "(the path to the per-position catalog CSV)."
        )

    quantities = tuple(data.get("quantities", ()))
    _check_known_quantities(quantities)

    out_dir = data.get("out_dir")
    return RunConfig(
        catalog=_resolve(base, data["catalog"]),
        out_dir=_resolve(base, out_dir) if out_dir else None,
        curation=_resolve(base, data.get("curation", _DEFAULT_CURATION)),
        quantities=quantities,
        params=dict(data.get("params", {})),
    )


def _resolve(base: Path, raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.expanduser().resolve(strict=False)


def _check_known_quantities(quantities: tuple[str, ...]) -> None:
    known = {cls.quantity_id for cls in available_quantifiers()}
    unknown = [q for q in quantities if q not in known]
    if unknown:
        listed = ", ".join(repr(q) for q in unknown)
        available = ", ".join(sorted(known))
        raise ValueError(
            f"Run-config selects unknown quantit(y/ies): {listed}. "
            f"Available: {available}."
        )


def write_config(
    path: Path | str,
    *,
    catalog: str = "catalog.csv",
    out_dir: str | None = None,
    curation: str = _DEFAULT_CURATION,
    quantities: Sequence[str] = (),
    params: Mapping[str, object] | None = None,
) -> Path:
    """Author a TOML run-config at *path* — the inverse of :func:`load_config`.

    Paths are written **relative** (verbatim), so the project folder stays
    relocatable. ``out_dir`` (where the flat tables land) is emitted only when
    given. ``quantities`` is emitted only when non-empty (empty round-trips to
    ``()`` = "all"). ``params`` keys that are ``None`` are dropped (an unset pixel
    size etc.). ``load_config(write_config(path, ...))`` reproduces the inputs
    (paths resolved against ``path.parent``). Returns *path*.
    """
    path = Path(path)
    lines: list[str] = [
        f"catalog = {_toml_str(catalog)}",
        f"curation = {_toml_str(curation)}",
    ]
    if out_dir is not None:
        lines.append(f"out_dir = {_toml_str(out_dir)}")
    if quantities:
        lines.append(f"quantities = {_toml_array(quantities)}")

    if params:
        kept = {k: v for k, v in params.items() if v is not None}
        if kept:
            lines.append("")
            lines.append("[params]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in kept.items())

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _toml_str(value: object) -> str:
    """A TOML basic string: backslash and double-quote escaped."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _toml_array(values: Sequence[object]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _toml_value(value: object) -> str:
    """Serialize a scalar for our closed schema (bool / int / float / str)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return _toml_str(value)
