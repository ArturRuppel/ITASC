"""Analysis module interface for TissueGraph data.

Defines the base class that all analysis modules implement, plus
parameter declarations for auto-generating dashboard UI widgets.

Modules are discovered via the ``napariTissueFlow.analysis_modules``
entry-point group, or by scanning a plugin folder.

Example third-party module (in ``pyproject.toml``)::

    [project.entry-points."napariTissueFlow.analysis_modules"]
    my_module = "my_package:MyModule"

Example usage::

    from napariTissueFlow.analysis.modules import discover_modules

    for name, module_cls in discover_modules().items():
        mod = module_cls()
        print(mod.name, mod.description)
        result = mod.compute(source)
        figs = mod.visualize(result)
"""
from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import pandas as pd

from ..core.api import Source

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "napariTissueFlow.analysis_modules"


# ------------------------------------------------------------------
# Parameter declarations
# ------------------------------------------------------------------


class ParamType(Enum):
    """Supported parameter types for auto-generated UI widgets."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    CHOICE = "choice"  # dropdown from a fixed set
    TAG = "tag"  # tag name(s) from the dataset
    FRAME_RANGE = "frame_range"  # (start, end) tuple


@dataclass
class Parameter:
    """Declarative parameter specification.

    Parameters
    ----------
    name : str
        Internal name (valid Python identifier, used as keyword arg).
    label : str
        Human-readable label shown in UI.
    type : ParamType
        Controls which widget is generated.
    default : Any
        Default value.
    description : str
        Tooltip / help text.
    min : float | int | None
        Minimum value (INT / FLOAT only).
    max : float | int | None
        Maximum value (INT / FLOAT only).
    step : float | int | None
        Step size (INT / FLOAT only).
    choices : list[str] | None
        Allowed values (CHOICE only).
    """

    name: str
    label: str
    type: ParamType
    default: Any = None
    description: str = ""
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[List[str]] = None


# ------------------------------------------------------------------
# Analysis result
# ------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Container for analysis module output.

    Attributes
    ----------
    tables : dict[str, DataFrame]
        Named result tables. At minimum, modules should return a
        ``"main"`` table with the primary result.
    metadata : dict[str, Any]
        Scalar values, labels, summary statistics.
    """

    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------


class AnalysisModule(ABC):
    """Base class for all analysis modules.

    Subclasses must implement :pyattr:`name`, :pyattr:`description`,
    :pymeth:`parameters`, :pymeth:`compute`, and :pymeth:`visualize`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier (snake_case)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description."""

    @abstractmethod
    def parameters(self) -> List[Parameter]:
        """Declare the parameters this module accepts.

        The dashboard auto-generates UI widgets from this list.
        """

    @abstractmethod
    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        """Run the analysis.

        Parameters
        ----------
        source : Source
            A TissueGraphTimeSeries, TissueGraphDataset, or path.
        **params
            Keyword arguments matching :meth:`parameters` names.

        Returns
        -------
        AnalysisResult
        """

    @abstractmethod
    def visualize(self, result: AnalysisResult) -> List[Any]:
        """Create figures from a computed result.

        Parameters
        ----------
        result : AnalysisResult
            Output of :meth:`compute`.

        Returns
        -------
        list of plotly Figure objects (``plotly.graph_objects.Figure``)
        """

    def validate_params(self, **params: Any) -> Dict[str, Any]:
        """Fill missing params with defaults and validate types.

        Returns the cleaned parameter dict.
        """
        declared = {p.name: p for p in self.parameters()}
        clean: Dict[str, Any] = {}
        for pname, pspec in declared.items():
            val = params.get(pname, pspec.default)
            if pspec.type == ParamType.INT and val is not None:
                val = int(val)
                if pspec.min is not None and val < pspec.min:
                    val = int(pspec.min)
                if pspec.max is not None and val > pspec.max:
                    val = int(pspec.max)
            elif pspec.type == ParamType.FLOAT and val is not None:
                val = float(val)
                if pspec.min is not None and val < pspec.min:
                    val = pspec.min
                if pspec.max is not None and val > pspec.max:
                    val = pspec.max
            elif pspec.type == ParamType.CHOICE and val is not None:
                if pspec.choices and val not in pspec.choices:
                    val = pspec.default
            clean[pname] = val
        return clean


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------


def discover_modules(
    plugin_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> Dict[str, Type[AnalysisModule]]:
    """Find all installed analysis modules.

    Discovery sources (checked in order):

    1. Built-in modules in ``napariTissueFlow.analysis.builtins``
    2. Entry points in the ``napariTissueFlow.analysis_modules`` group
    3. Python files in *plugin_dirs* (each file may define one or more
       ``AnalysisModule`` subclasses)

    Returns
    -------
    dict[str, type[AnalysisModule]]
        Mapping from module name to class.
    """
    found: Dict[str, Type[AnalysisModule]] = {}

    # 1. Built-in modules
    try:
        from . import builtins as _builtins_pkg

        for attr_name in dir(_builtins_pkg):
            obj = getattr(_builtins_pkg, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, AnalysisModule)
                and obj is not AnalysisModule
            ):
                try:
                    inst = obj()
                    found[inst.name] = obj
                except Exception:
                    logger.warning("Failed to instantiate builtin %s", attr_name)
    except ImportError:
        pass  # builtins package not yet created

    # 2. Entry points
    eps = entry_points()
    if hasattr(eps, "select"):
        # Python 3.12+
        group = eps.select(group=ENTRY_POINT_GROUP)
    else:
        group = eps.get(ENTRY_POINT_GROUP, [])

    for ep in group:
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, AnalysisModule):
                inst = cls()
                found[inst.name] = cls
            else:
                logger.warning(
                    "Entry point %s does not point to an AnalysisModule subclass",
                    ep.name,
                )
        except Exception:
            logger.warning("Failed to load entry point %s", ep.name, exc_info=True)

    # 3. Plugin directories
    for d in plugin_dirs or []:
        dirpath = Path(d)
        if not dirpath.is_dir():
            continue
        for py_file in sorted(dirpath.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"napariTissueFlow_plugin_{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        obj = getattr(mod, attr_name)
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, AnalysisModule)
                            and obj is not AnalysisModule
                        ):
                            inst = obj()
                            found[inst.name] = obj
            except Exception:
                logger.warning(
                    "Failed to load plugin %s", py_file, exc_info=True
                )

    return found
