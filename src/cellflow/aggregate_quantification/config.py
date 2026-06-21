"""The TOML *run-config* — Aggregate Quantification's hand-authored knob file.

One small file, authored once and git-versioned (it carries *code status*, per the
artifact-contract spec §1): it names the per-position **catalog** CSV, selects
which **quantities** to compute, supplies the shared build **params**, and points
at the **curation** file and **export** directory. Everything in it is a run-level
choice — the per-position table itself stays a CSV (tabular, many-row, with its own
relative-path resolution); this file is the "author once, then run" surface a
single ``run(config)`` entry point consumes.

Paths resolve relative to the config file's own directory, so a project folder is
relocatable: keep ``config.toml`` next to ``catalog.csv`` / ``curation.csv`` /
``export/`` and the whole thing moves together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

from .quantifier import available_quantifiers

__all__ = ["RunConfig", "load_config"]

#: Defaults for the optional path keys, relative to the config file's directory.
_DEFAULT_EXPORT_DIR = "export"
_DEFAULT_CURATION = "curation.csv"


@dataclass(frozen=True)
class RunConfig:
    """A parsed run-config. Paths are absolute (resolved against the config dir).

    *quantities* empty means "every available quantifier"; a non-empty tuple
    selects a subset by ``quantity_id`` (dependency producers are pulled in at run
    time even when omitted). *params* is the shared build-knob mapping threaded to
    quantifiers that opt in.
    """

    catalog: Path
    export_dir: Path
    curation: Path
    quantities: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)


def load_config(config_path: Path | str) -> RunConfig:
    """Parse the TOML run-config at *config_path* into a :class:`RunConfig`.

    ``catalog`` is required; ``export_dir`` / ``curation`` default to ``export/``
    and ``curation.csv`` beside the config. Relative paths resolve against the
    config file's directory. Selected ``quantities`` are validated against the
    registered quantifiers so a typo fails loudly rather than silently computing
    nothing.
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

    return RunConfig(
        catalog=_resolve(base, data["catalog"]),
        export_dir=_resolve(base, data.get("export_dir", _DEFAULT_EXPORT_DIR)),
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
