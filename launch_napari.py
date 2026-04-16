"""Launch napari with the CellFlow plugin pre-loaded and the analysis project opened.

Usage:
    python launch_napari.py
"""
from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(
    "/home/aruppel/Data"
    "/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis"
)


def main() -> None:
    import napari
    from cellflow import _napari_patches  # noqa: F401 — must be imported before widget
    from cellflow.napari.analysis_widget import CellFlowWidget
    from cellflow.napari.registry import get_state

    viewer = napari.Viewer(title="CellFlow")

    widget = CellFlowWidget(viewer)
    viewer.window.add_dock_widget(widget, name="CellFlow", area="right")

    # Pre-load the project directory so the pipeline file tracker is live
    state = get_state(viewer)
    state.set_project_dir(PROJECT_DIR)

    napari.run()


if __name__ == "__main__":
    main()
