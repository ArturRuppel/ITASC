from itasc.napari._napari_compat import patch_napari_layer_delegate

patch_napari_layer_delegate()

from itasc.napari.main_widget import ITASCMainWidget as ITASCWidget  # noqa: E402

__all__ = ["ITASCWidget"]
