__all__ = ["CellFlowWidget"]


def __getattr__(name: str):
    if name == "CellFlowWidget":
        from cellflow import _napari_patches  # noqa: F401
        from .analysis_widget import CellFlowWidget
        return CellFlowWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
