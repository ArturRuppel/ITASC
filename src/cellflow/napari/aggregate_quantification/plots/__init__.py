"""The plot-consumer seam: presentation plots, discovered by subclassing.

A :class:`Plot` is the symmetric counterpart of a
:class:`~cellflow.aggregate_quantification.quantifier.Quantifier`. Where a
quantifier *produces* a per-position product (an ``object_table`` persisted under
a ``quantity_id``), a plot *consumes* one or more such products and renders them.
A plot declares only the ``quantity_id``\\s it needs — never which quantifier
produced them — so "is this plottable right now?" reduces to "are the products it
consumes built for the in-scope positions?".

The mechanics mirror the quantifier and analysis-plugin registries: subclassing
with a non-empty ``plot_id`` auto-registers the plot, and :func:`available_plots`
imports every module in this package so dropping a new ``*.py`` file here is all
it takes to add a plot.

Two flavours register through the same contract and differ only in
:meth:`Plot.create_panel`:

* **generic statistical plots** route the pooled products through the
  quantity-agnostic
  :class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel`;
* **bespoke plots** (e.g. the dynamics MSD / DAC / C(r) curve views) build their
  own figure widget.

The base class is **headless** — ``plot_id`` / ``family`` / ``consumes`` and the
availability helpers carry no Qt, so the registry and gating are unit-testable
without a viewer. Only :meth:`create_panel` (called at click time) touches Qt.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "Plot",
    "PlotContext",
    "PlotParams",
    "available_plots",
]

#: Default permutation count for the contact-type z-score null.
_DEFAULT_SHUFFLES = 1000


@dataclass(frozen=True)
class PlotParams:
    """Shared, plot-time tuning the Plot area applies to every :meth:`Plot.prepare`.

    One set of fields drives every plot: a plot reads only the ones it needs and
    ignores the rest. ``None`` means "auto" — resolve per position rather than
    forcing a single value.

    * ``pixel_size_um`` — µm/px for axes in physical units (the potential
      landscape's signed length); ``None`` auto-resolves per position.
    * ``fov_area_mm2`` — field-of-view area for the density view; ``None`` uses
      each position's full image area.
    * ``shuffles`` — label permutations for the contact-type z-score null.
    """

    pixel_size_um: float | None = None
    fov_area_mm2: float | None = None
    shuffles: int = _DEFAULT_SHUFFLES


@dataclass
class PlotContext:
    """What the plot area hands a plot when a panel is opened.

    ``records`` are the in-scope catalogue rows (the snapshot the panel binds to);
    ``built`` is the set of ``quantity_id``\\s available for that scope (so a plot
    need not re-probe the filesystem); ``load`` reads/caches a per-position
    artifact, shared with the rest of the studio so plots do not each re-open the
    same files.
    """

    records: list[dict] = field(default_factory=list)
    viewer: Any | None = None
    built: frozenset[str] = frozenset()
    loader: Callable[[Path], Any] | None = None

    def load(self, path: Path) -> Any:
        if self.loader is None:
            raise RuntimeError("PlotContext has no loader configured")
        return self.loader(Path(path))


#: plot_id -> plot class, populated by ``__init_subclass__``.
_REGISTRY: dict[str, type[Plot]] = {}


class Plot:
    """Base class for Aggregate Quantification presentation plots.

    Subclasses set the ``plot_id`` / ``display_name`` / ``family`` class
    attributes, declare the ``consumes`` products, and implement
    :meth:`create_panel`. Defining a subclass with a non-empty ``plot_id``
    registers it; the studio discovers it via :func:`available_plots`.
    """

    #: Stable key; an empty value marks an intermediate (non-registered) base.
    plot_id: ClassVar[str] = ""
    #: Human-readable label shown in the plot area.
    display_name: ClassVar[str] = ""
    #: Grouping bucket in the plot area — the *product family* / input data type
    #: ("Shape", "Dynamics", "Contacts", …). Plots that read a single family list
    #: under it; a cross-family plot picks its primary family.
    family: ClassVar[str] = ""
    #: ``quantity_id``\\s this plot needs, all required (AND). A plot is available
    #: only when every one is built for at least one in-scope position.
    consumes: ClassVar[tuple[str, ...]] = ()
    #: Which render-type button hosts this plot in the Plot area. Plots of the
    #: same render type share **one** button; their values merge into a single
    #: source-grouped value picker (``family`` is the visible group header).
    #: Bespoke renders (``"curve"``) get their own button instead of a picker.
    render_type: ClassVar[str] = "distribution"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.plot_id:
            _REGISTRY[cls.plot_id] = cls

    def is_available(self, built: Collection[str]) -> bool:
        """True when every consumed product is in *built*."""
        return all(quantity_id in built for quantity_id in self.consumes)

    def missing(self, built: Collection[str]) -> tuple[str, ...]:
        """The consumed products absent from *built*, in declaration order."""
        return tuple(quantity_id for quantity_id in self.consumes if quantity_id not in built)

    def prepare(self, records: list[dict], params: PlotParams = PlotParams()) -> Any:
        """Heavy, **headless** read for *records* — the part the plot area runs
        off the GUI thread before building the panel.

        *params* is the shared plot-time tuning (pixel size, FOV, shuffles); a
        plot uses only the fields it needs. Returns whatever :meth:`create_panel`
        consumes as its ``prepared`` payload (a pooled DataFrame, a list of curve
        sets, …). Defaults to ``None`` for plots whose read is cheap enough to do
        inline in :meth:`create_panel`.
        """
        return None

    def create_panel(self, ctx: PlotContext, prepared: Any = None) -> Any:  # pragma: no cover - overridden
        """Build and return the plot's Qt widget, bound to *ctx*'s snapshot.

        *prepared* is the off-thread :meth:`prepare` result when the plot area
        pre-read it; ``None`` means build inline. Must run on the GUI thread.
        """
        raise NotImplementedError


def _import_plot_modules() -> None:
    """Import every (non-private) submodule so its plots self-register."""
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")


def available_plots() -> list[type[Plot]]:
    """Return registered plot classes, sorted by family then display name."""
    _import_plot_modules()
    return sorted(
        _REGISTRY.values(),
        key=lambda cls: (cls.family.lower(), cls.display_name.lower()),
    )
