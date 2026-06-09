"""Private CellFlow utility widgets exposed as a separate napari plugin."""

from cellflow.napari._napari_compat import patch_napari_layer_delegate

patch_napari_layer_delegate()

from cellflow_utils.napari.data_prep_standalone_widget import DataPrepStandaloneWidget
from cellflow_utils.napari.hpc_cellpose_widget import HpcCellposeWidget
from cellflow_utils.napari.nls_classification_widget import NLSClassificationWidget

__all__ = [
    "DataPrepStandaloneWidget",
    "HpcCellposeWidget",
    "NLSClassificationWidget",
]
