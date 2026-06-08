"""Meta-analysis plugin contract and in-process registry.

The meta-analysis widget is a thin base that manages a *catalog* of positions
and hosts pluggable analysis views on top. A plugin is a ``QWidget`` subclass of
:class:`MetaAnalysisPlugin` that declares a ``plugin_id`` + ``display_name`` and
implements :meth:`~MetaAnalysisPlugin.set_context`; the base instantiates it and
feeds it the currently-selected catalog records.

Plugins are **not** napari manifest contributions — they must not show up as
top-level dock widgets alongside the real tools. Instead, subclassing
:class:`MetaAnalysisPlugin` auto-registers the plugin (via ``__init_subclass__``),
and :func:`available_meta_plugins` imports every module in this package so that
dropping a new ``*.py`` file here is all it takes to add a plugin.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from qtpy.QtWidgets import QWidget

__all__ = [
    "MetaContext",
    "MetaAnalysisPlugin",
    "available_meta_plugins",
]


@dataclass
class MetaContext:
    """What the base hands a plugin: the in-scope catalog rows + a cached loader.

    ``records`` are normalized catalog dicts (see :mod:`cellflow.meta.catalog`);
    each has at least ``contact_analysis_path``, ``condition``, ``date``, ``id``
    and ``contact_analysis_status``. ``load`` reads (and caches) the
    :class:`~cellflow.contact_analysis.reader.PositionContactAnalysis` for a row,
    so plugins do not each re-open the same HDF5.
    """

    records: list[dict] = field(default_factory=list)
    viewer: Any | None = None
    #: Injected by the base; maps a contact-analysis path to its parsed object.
    loader: Callable[[Path], Any] | None = None

    def load(self, record: dict) -> Any:
        if self.loader is None:
            raise RuntimeError("MetaContext has no loader configured")
        return self.loader(Path(record["contact_analysis_path"]))


#: plugin_id -> plugin class, populated by ``__init_subclass__``.
_REGISTRY: dict[str, type[MetaAnalysisPlugin]] = {}


class MetaAnalysisPlugin(QWidget):
    """Base class for meta-analysis plugins.

    Subclasses set the ``plugin_id`` / ``display_name`` class attributes and
    override :meth:`set_context`. Defining a subclass with a non-empty
    ``plugin_id`` registers it; the Meta Analysis widget discovers it via
    :func:`available_meta_plugins`.
    """

    #: Stable key; an empty value marks an intermediate (non-registered) base.
    plugin_id: ClassVar[str] = ""
    #: Human-readable label shown in the Meta Analysis plugin selector.
    display_name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.plugin_id:
            _REGISTRY[cls.plugin_id] = cls

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer

    def set_context(self, ctx: MetaContext) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


def _import_plugin_modules() -> None:
    """Import every (non-private) submodule so its plugins self-register."""
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")


def available_meta_plugins() -> list[type[MetaAnalysisPlugin]]:
    """Return registered plugin classes, sorted by display name."""
    _import_plugin_modules()
    return sorted(_REGISTRY.values(), key=lambda cls: cls.display_name.lower())
