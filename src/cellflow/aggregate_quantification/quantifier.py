"""The quantifier seam: per-position compute units, discovered by subclassing.

A :class:`Quantifier` turns a position's source files (:class:`PositionInputs`)
into a persisted, plottable quantity. Contacts is the first one; new quantities
(nucleus-track kinematics, nucleus-vs-cell offset, cell shape, …) drop in as
modules under :mod:`cellflow.aggregate_quantification.quantifiers` without
touching the studio.

The mechanics mirror :mod:`cellflow.napari.aggregate_quantification.plugins`: subclassing with a
non-empty ``quantity_id`` auto-registers the quantifier, and
:func:`available_quantifiers` imports every module in the ``quantifiers`` package
so its plugins self-register. This module stays backend-only (no Qt / napari) so
the standalone wheel and headless batch runs can use it.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

__all__ = ["PositionInputs", "Quantifier", "available_quantifiers"]


@dataclass(frozen=True)
class PositionInputs:
    """Resolved source files for one position. Quantifiers read what they need.

    Every field has a live consumer. A future track-based quantifier adds its own
    field (e.g. ``tracks_db_path``) in the same commit that first reads it — no
    speculative placeholders here.
    """

    position_dir: Path
    cell_labels_path: Path | None = None
    nucleus_labels_path: Path | None = None


#: quantity_id -> quantifier class, populated by ``__init_subclass__``.
_REGISTRY: dict[str, type[Quantifier]] = {}


class Quantifier:
    """Base class for per-position quantifiers.

    Subclasses set ``quantity_id`` / ``display_name`` (and usually ``requires``)
    and implement :meth:`build` and :meth:`read`. A quantifier **owns its own
    persistence**: :meth:`build` writes whatever artifact format suits the
    quantity and :meth:`read` parses it back — the framework imposes no schema.
    """

    #: Stable key; an empty value marks an intermediate (non-registered) base.
    quantity_id: ClassVar[str] = ""
    #: Human-readable label shown wherever a quantity is selected.
    display_name: ClassVar[str] = ""
    #: ``PositionInputs`` field names this quantifier needs to build.
    requires: ClassVar[tuple[str, ...]] = ()
    #: Default artifact file name (relative to a position); empty for an
    #: intermediate base that does not persist.
    default_output_name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.quantity_id:
            _REGISTRY[cls.quantity_id] = cls

    def can_build(self, inputs: PositionInputs) -> bool:
        """True when *inputs* supplies every field named in :attr:`requires`."""
        return all(getattr(inputs, name, None) is not None for name in self.requires)

    def default_output(self, inputs: PositionInputs) -> Path:
        """Where this quantifier's artifact lives for *inputs*, by default.

        ``position_dir / default_output_name``. The studio uses this to decide a
        build's destination, so a second quantifier no longer inherits the
        contacts artifact path. Subclasses may override for richer layouts.
        """
        if not self.default_output_name:
            raise NotImplementedError(
                f"{type(self).__name__} sets no default_output_name"
            )
        return inputs.position_dir / self.default_output_name

    def is_built(self, output_path: Path) -> bool:
        """True when the artifact at *output_path* already exists."""
        return Path(output_path).is_file()

    def object_table(self, output_path: Path) -> Mapping[str, Any] | None:
        """A tidy, column-major per-object table for the plotting backend.

        At least a ``frame`` key plus a per-object key (e.g. ``cell_id``).
        Returns ``None`` when this quantifier produces no per-object table; the
        plotting backend then skips it.
        """
        return None

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:  # pragma: no cover - overridden
        raise NotImplementedError

    def read(self, output_path: Path) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


def _import_quantifier_modules() -> None:
    """Import every (non-private) ``quantifiers`` submodule so it self-registers."""
    from . import quantifiers

    for info in pkgutil.iter_modules(quantifiers.__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{quantifiers.__name__}.{info.name}")


def available_quantifiers() -> list[type[Quantifier]]:
    """Return registered quantifier classes, sorted by display name."""
    _import_quantifier_modules()
    return sorted(_REGISTRY.values(), key=lambda cls: cls.display_name.lower())
