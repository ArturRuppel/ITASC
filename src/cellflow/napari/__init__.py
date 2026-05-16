from cellflow.napari._napari_compat import patch_napari_layer_delegate

patch_napari_layer_delegate()

from cellflow.napari.main_widget import CellFlowMainWidget as CellFlowWidget
from cellflow.napari.data_prep_standalone_widget import DataPrepStandaloneWidget

__all__ = ["CellFlowWidget", "DataPrepStandaloneWidget"]
