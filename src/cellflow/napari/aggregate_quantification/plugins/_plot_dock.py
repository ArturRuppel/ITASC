"""Shared dock that collects a plugin's plots as tabs.

Opening each plot as its own ``add_dock_widget(area="right")`` made napari split
the right dock area every time, so the plot pane — and thus every plot — shrank
with each new plot, and closing docks didn't give the space back. Instead all of
a plugin's plots share **one** dock holding a ``QTabWidget``: the first plot
creates the dock (split off as its own column beside the controls); every later
plot is just a new closable tab. The pane is constant, so every plot keeps the
same size, and closing a tab frees it.

When that dock is first created we **widen the whole window** by the plot's
width rather than letting napari steal it from the controls: ``splitDockWidget``
keeps the main window's total width constant, which crushes the control panel.
We grow the window and then ``resizeDocks`` so the controls keep their width and
the plot gets the new space. The plot panel's own fields are built to shrink
(see :class:`PlotPanel`), so the pane can be dragged narrower than the plot's
nominal width without a scrollbar.

A plugin keeps one :class:`PlotDockTabs` and calls :meth:`add` with each panel.
The underscore filename keeps this out of the plugin auto-discovery in
``plugins/__init__.py``.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDockWidget, QMainWindow, QTabWidget, QWidget

#: Fallback width (px) for a freshly opened plot pane when the panel can't yet
#: report a sensible size hint.
_DEFAULT_PLOT_WIDTH = 460


class PlotDockTabs:
    """Collects one plugin's plot panels as tabs in a single shared dock."""

    def __init__(self, plugin: QWidget, dock_name: str) -> None:
        self._plugin = plugin
        self._dock_name = dock_name
        self._dock: QDockWidget | None = None
        self._tabs: QTabWidget | None = None

    def add(self, panel: QWidget, title: str) -> None:
        """Add *panel* as a new tab, creating the shared dock on first use.

        Degrades to a no-op when the plugin has no viewer (the unit-test path
        before a viewer is injected)."""
        viewer = getattr(self._plugin, "viewer", None)
        if viewer is None:
            return
        if not self._is_live():
            self._create_dock(viewer, panel)
        index = self._tabs.addTab(panel, title)
        self._tabs.setCurrentIndex(index)

    # ------------------------------------------------------------------ internals
    def _is_live(self) -> bool:
        """True while the tab widget still exists (the user hasn't closed the
        whole dock)."""
        if self._tabs is None:
            return False
        try:
            self._tabs.count()  # raises if the C++ object was deleted
            return True
        except RuntimeError:
            return False

    def _create_dock(self, viewer, panel: QWidget) -> None:
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        # Let the dock be dragged narrower than the tab bar's hint; the panel's
        # own fields shrink to follow (see PlotPanel).
        self._tabs.setMinimumWidth(0)
        self._dock = viewer.window.add_dock_widget(
            self._tabs, area="right", name=self._dock_name
        )
        # One-time split: give the plots their own full-height column beside the
        # controls instead of stacking under them.
        main = _main_window(self._dock)
        host = _host_dock(self._plugin)
        if main is not None and host is not None and isinstance(self._dock, QDockWidget):
            main.splitDockWidget(host, self._dock, Qt.Horizontal)
            self._grow_window_for_plot(main, host, panel)

    def _grow_window_for_plot(self, main: QMainWindow, host: QDockWidget, panel: QWidget) -> None:
        """Widen the window by the plot's width so the controls keep theirs.

        ``splitDockWidget`` divides the host's existing width between the two
        docks; instead we add the plot's width to the whole window and then pin
        the dock sizes so the controls stay put and the plot fills the new
        space."""
        host_w = host.width() or host.sizeHint().width()
        plot_w = max(panel.sizeHint().width(), _DEFAULT_PLOT_WIDTH)
        if not (main.isMaximized() or main.isFullScreen()):
            main.resize(main.width() + plot_w, main.height())
        main.resizeDocks([host, self._dock], [host_w, plot_w], Qt.Horizontal)

    def _close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()


def _host_dock(widget: QWidget) -> QDockWidget | None:
    """The QDockWidget wrapping *widget* (the studio controls dock)."""
    node = widget.parentWidget()
    while node is not None:
        if isinstance(node, QDockWidget):
            return node
        node = node.parentWidget()
    return None


def _main_window(dock: object) -> QMainWindow | None:
    """The QMainWindow *dock* lives in, walking up its parent chain."""
    widget = dock.parentWidget() if isinstance(dock, QDockWidget) else None
    while widget is not None:
        if isinstance(widget, QMainWindow):
            return widget
        widget = widget.parentWidget()
    return None
