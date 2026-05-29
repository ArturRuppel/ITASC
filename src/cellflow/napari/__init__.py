from cellflow.napari._napari_compat import patch_napari_layer_delegate

patch_napari_layer_delegate()

from cellflow.napari.main_widget import CellFlowMainWidget as CellFlowWidget  # noqa: E402

__all__ = ["CellFlowWidget"]
