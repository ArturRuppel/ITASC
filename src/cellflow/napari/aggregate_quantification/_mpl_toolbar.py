"""Theme matplotlib's ``NavigationToolbar2QT`` for dark napari themes.

napari themes the application through Qt **stylesheets**, which do not update the
widget ``QPalette``. matplotlib's toolbar recolours its glyph icons only when
*its own palette's* background reads dark, so under a napari dark theme (light
palette, dark stylesheet) the glyphs stay black on a dark bar and all but vanish.

:func:`theme_toolbar_icons` sidesteps that by reading the active napari theme
directly and tinting the toolbar's icons to the theme's icon colour. The napari
import is guarded, so the module still imports (and the call no-ops) in the
napari-free unit tests that exercise the plot panels headless.
"""
from __future__ import annotations

import re

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap

#: Default icon tint when the theme exposes no usable icon colour but reads dark.
_FALLBACK_TINT = "#d4d4d4"


def theme_toolbar_icons(toolbar: NavigationToolbar2QT) -> None:
    """Tint the toolbar glyphs to the active napari theme's icon colour when that
    theme is dark.

    A no-op on light themes (matplotlib's black glyphs read fine there) and when
    napari is not importable — so headless tests are unaffected."""
    color = _dark_theme_icon_color()
    if color is None:
        return
    for action in toolbar.actions():
        icon = action.icon()
        if icon.isNull():
            continue
        sizes = icon.availableSizes()
        size = sizes[0] if sizes else QSize(24, 24)
        action.setIcon(_tinted(icon.pixmap(size), color))


def _tinted(pixmap: QPixmap, color: QColor) -> QIcon:
    """A copy of *pixmap* with every opaque pixel recoloured to *color* (the glyph
    shape kept, its black fill swapped for the tint)."""
    out = QPixmap(pixmap.size())
    out.setDevicePixelRatio(pixmap.devicePixelRatio())
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(out.rect(), color)
    painter.end()
    return QIcon(out)


def _dark_theme_icon_color() -> QColor | None:
    """The active napari theme's icon colour when that theme is dark, else None
    (light theme, or napari unavailable)."""
    try:
        from napari.settings import get_settings
        from napari.utils.theme import get_theme
    except Exception:
        return None
    try:
        theme = get_theme(get_settings().appearance.theme)
    except Exception:
        return None
    background = _to_qcolor(_field(theme, "background"))
    if background is None or _luminance(background) >= 128:
        return None  # light (or unknown) theme — leave matplotlib's black glyphs
    icon = _to_qcolor(_field(theme, "icon")) or _to_qcolor(_field(theme, "text"))
    return icon or QColor(_FALLBACK_TINT)


def _field(theme: object, key: str):
    """Read *key* off a napari Theme (object or, on older napari, a dict)."""
    if isinstance(theme, dict):
        return theme.get(key)
    return getattr(theme, key, None)


def _to_qcolor(value: object) -> QColor | None:
    """Best-effort conversion of a napari theme colour (hex / ``rgb(...)`` string
    or pydantic Color) to a ``QColor``; None when it can't be parsed."""
    if value is None:
        return None
    as_hex = getattr(value, "as_hex", None)
    if callable(as_hex):
        color = QColor(as_hex())
        return color if color.isValid() else None
    text = str(value)
    color = QColor(text)
    if color.isValid():
        return color
    nums = re.findall(r"[\d.]+", text)
    if len(nums) >= 3:
        return QColor(int(float(nums[0])), int(float(nums[1])), int(float(nums[2])))
    return None


def _luminance(color: QColor) -> float:
    return 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
