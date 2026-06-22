"""The TOML *run-config* — Aggregate Quantification's hand-authored knob file.

One small file, authored once and git-versioned (it carries *code status*, per the
artifact-contract spec §1): it names the per-position **catalog** CSV, selects
which **quantities** to compute, supplies the shared build **params**, and points
at the **export** directory. Everything in it is a run-level choice — the
per-position table itself stays a CSV (tabular, many-row, with its own
relative-path resolution); this file is the "author once, then run" surface a
single ``run(config)`` entry point consumes.

Paths resolve relative to the config file's own directory, so a project folder is
relocatable: keep ``config.toml`` next to ``catalog.csv`` / ``export/`` and the
whole thing moves together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

from .quantifier import available_quantifiers

__all__ = ["NlsConfig", "RunConfig", "load_config"]

#: Default for the optional export-dir key, relative to the config file's directory.
_DEFAULT_EXPORT_DIR = "export"

#: Default for the optional curation-table key, relative to the config dir.
_DEFAULT_CURATION = "curation.csv"

#: Default per-position-relative marker image for the optional [nls] step.
_DEFAULT_NLS_IMAGE = "0_input/NLS_zavg.tif"

#: The NLS thresholding methods the classify step understands.
_NLS_METHODS = ("auto", "otsu", "two_cluster", "fixed")


@dataclass(frozen=True)
class NlsConfig:
    """Parsed ``[nls]`` table — the optional NLS classification step's knobs.

    *image* is the marker image **relative to each position directory** (e.g.
    ``0_input/NLS_zavg.tif``) so one entry resolves across a batch; an absolute
    path is used verbatim. *method* picks the thresholding: ``auto`` (per-position
    two-cluster, Otsu fallback — the default), ``otsu``, ``two_cluster``, or
    ``fixed`` (pins *threshold* across the series).
    """

    enabled: bool = False
    image: str = _DEFAULT_NLS_IMAGE
    method: str = "auto"
    threshold: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    """A parsed run-config. Paths are absolute (resolved against the config dir).

    *quantities* empty means "every available quantifier"; a non-empty tuple
    selects a subset by ``quantity_id`` (dependency producers are pulled in at run
    time even when omitted). *params* is the shared build-knob mapping threaded to
    quantifiers that opt in.

    *render_plots* (the ``[plots]`` table's ``render``) turns on rendering the
    premade SuperPlots to static figures via the Iris engine (the optional
    ``cellflow[plots]`` extra); *plot_formats* (``[plots].formats``) picks the
    output formats. When off, the run stays Iris-only with no engine dependency.
    """

    catalog: Path
    export_dir: Path
    curation: Path = field(default=Path(_DEFAULT_CURATION))
    nls: NlsConfig | None = None
    quantities: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    render_plots: bool = False
    plot_formats: tuple[str, ...] = ("png", "svg")


def load_config(config_path: Path | str) -> RunConfig:
    """Parse the TOML run-config at *config_path* into a :class:`RunConfig`.

    ``catalog`` is required; ``export_dir`` defaults to ``export/`` beside the
    config. Relative paths resolve against the config file's directory. Selected
    ``quantities`` are validated against the registered quantifiers so a typo fails
    loudly rather than silently computing nothing.
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

    plots = data.get("plots", {})
    render_plots = bool(plots.get("render", False))
    plot_formats = tuple(plots.get("formats", ("png", "svg")))

    nls = _parse_nls(data.get("nls"))

    return RunConfig(
        catalog=_resolve(base, data["catalog"]),
        export_dir=_resolve(base, data.get("export_dir", _DEFAULT_EXPORT_DIR)),
        curation=_resolve(base, data.get("curation", _DEFAULT_CURATION)),
        nls=nls,
        quantities=quantities,
        params=dict(data.get("params", {})),
        render_plots=render_plots,
        plot_formats=plot_formats,
    )


def _resolve(base: Path, raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.expanduser().resolve(strict=False)


def _parse_nls(table: dict | None) -> NlsConfig | None:
    """Parse the optional ``[nls]`` table into an :class:`NlsConfig` (or ``None``).

    A missing table means the step is off. *method* is validated so a typo fails
    loudly rather than silently classifying with the wrong splitter.
    """
    if table is None:
        return None
    method = str(table.get("method", "auto"))
    if method not in _NLS_METHODS:
        listed = ", ".join(_NLS_METHODS)
        raise ValueError(
            f"Run-config [nls] selects unknown method {method!r}. Available: {listed}."
        )
    return NlsConfig(
        enabled=bool(table.get("enabled", False)),
        image=str(table.get("image", _DEFAULT_NLS_IMAGE)),
        method=method,
        threshold=float(table.get("threshold", 0.0)),
    )


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
