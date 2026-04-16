# Plan: Widget UX Fix — Expand to Full Size + Resize Handles

## Problem

When a `CollapsibleSection` expands, the inner widget appears tiny and requires scrolling.
Two root causes:

1. **`CollapsibleSection` uses `QSizePolicy.Expanding` when open.** Combined with
   `addStretch(1)` in `_plugin_layout`, Qt distributes leftover space between the
   stretch *and* the section — the section only gets a fraction of available height.

2. **Six inner widgets wrap themselves in a `QScrollArea` with no minimum height.**
   `QScrollArea` defaults to a very small size, so the wrapped content appears squished.

---

## Desired Behaviour

- Expanding a section → all controls visible at once, no scrolling needed
- Sections with scrollable sub-boxes (log viewers, file tables) → sub-boxes stay at
  their minimum height; they do NOT push the section taller
- User can drag a resize handle at the bottom of an expanded section to make it taller;
  scrollable sub-boxes grow to fill the extra space
- If user drags the section smaller than its natural height → scroll bar appears on
  the section's content area
- Collapsed sub-sections within an expanded section stay collapsed (unchanged)

---

## Approach

`CollapsibleSection` switches to **always `Preferred` policy** and wraps its inner
widget in an internal `QScrollArea` + a `_ResizeHandle` drag bar.  On expand the
scroll area's `minimumHeight` is set to the inner widget's `sizeHint`, giving the
"show everything" default.  Dragging the handle pins the height (min = max = dragged
value); scroll bar appears if content overflows.

Inner widgets that currently self-wrap in a `QScrollArea` are simplified to a plain
`QVBoxLayout(self)` — they rely on the section-level scroll area instead.

Sub-boxes (log viewers, tables) have their `setFixedHeight` / `setMaximumHeight`
replaced with `setMinimumHeight` so they can grow when the section is enlarged.

`analysis_widget.py` needs **no changes** — `addStretch(1)` stays and is correct.

---

## Files Affected

| File | Change |
|------|--------|
| `widgets.py` | Add `_ResizeHandle`; rework `CollapsibleSection` (always Preferred, inner QScrollArea + handle, height from sizeHint) |
| `ultrack_widgets/data_prep.py` | Remove outer QScrollArea; `QVBoxLayout(self)` + `AlignTop` |
| `ultrack_widgets/cellpose.py` | Same |
| `ultrack_widgets/ultrack_widget.py` | Same |
| `ultrack_widgets/flow_watershed.py` | Same |
| `edge_analysis_widget.py` | Same |
| `forces_widget.py` | Same |
| `project_panel.py` | `files_scroll.setFixedHeight(210)` → `setMinimumHeight(100)`; `_pipeline_table.setMaximumHeight(110)` → `setMinimumHeight(60)`; `_catalog_table.setMaximumHeight(130)` → `setMinimumHeight(60)` |
| `log_viewer.py` | Remove `setMaximumHeight(220)`; keep `setMinimumHeight(120)`; add `Expanding` policy |
| `tracking_widget.py` | `self._log.setFixedHeight(140)` → `setMinimumHeight(100)` + `Expanding` policy |

---

## Implementation Steps

### 1. `widgets.py` — add `_ResizeHandle`

```python
class _ResizeHandle(QWidget):
    """Draggable bar at the bottom of an expanded CollapsibleSection."""

    def __init__(self, scroll_area: "QScrollArea", parent=None) -> None:
        super().__init__(parent)
        self._scroll = scroll_area
        self._start_y: int | None = None
        self._start_h: int | None = None
        self.setFixedHeight(6)
        self.setCursor(Qt.SizeVerCursor)
        self.setStyleSheet(
            "background: #505050; border-radius: 3px; margin: 0 8px;"
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._start_y = event.globalPos().y()
            self._start_h = self._scroll.height()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._start_y is not None:
            delta = event.globalPos().y() - self._start_y
            new_h = max(40, self._start_h + delta)
            self._scroll.setMinimumHeight(new_h)
            self._scroll.setMaximumHeight(new_h)  # pin height
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._start_y = None
        self._start_h = None
        event.accept()
```

### 2. `widgets.py` — rework `CollapsibleSection`

**`__init__` changes:**
- Remove `v_policy` variable (always `Preferred`)
- `frame_layout` contains: `_scroll_area` (QScrollArea wrapping `inner`) +
  `_resize_handle` (_ResizeHandle)
- `_scroll_area`: `setWidgetResizable(True)`,
  `setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)`,
  `setFrameShape(QFrame.NoFrame)`
- `setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)` unconditionally

**`_on_toggled` changes:**
```python
def _on_toggled(self, checked: bool) -> None:
    self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
    self._content_frame.setVisible(checked)
    if checked:
        # Reset to natural height (unpin any previous drag)
        self._scroll_area.setMaximumHeight(16777215)
        QTimer.singleShot(0, self._reset_natural_height)
    else:
        self._scroll_area.setMinimumHeight(0)
        self._scroll_area.setMaximumHeight(16777215)

def _reset_natural_height(self) -> None:
    h = self._inner.sizeHint().height()
    if h > 10:
        self._scroll_area.setMinimumHeight(h)
    else:
        QTimer.singleShot(50, self._reset_natural_height)
```

### 3. Six inner widgets — remove outer QScrollArea

Pattern to remove (6–8 lines):
```python
# REMOVE:
scroll = QScrollArea()
scroll.setWidgetResizable(True)
scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
inner = QWidget()
inner_layout = QVBoxLayout()
inner_layout.setAlignment(Qt.AlignTop)
inner.setLayout(inner_layout)
scroll.setWidget(inner)
outer = QVBoxLayout()
outer.setContentsMargins(0, 0, 0, 0)
outer.addWidget(scroll)
self.setLayout(outer)

# REPLACE WITH:
inner_layout = QVBoxLayout(self)
inner_layout.setContentsMargins(0, 0, 0, 0)
inner_layout.setAlignment(Qt.AlignTop)
```

All subsequent `inner_layout.addWidget(...)` calls stay unchanged; just remove the
`inner.` prefix where needed.

### 4. `project_panel.py` — fix sub-box heights

```python
# files_scroll
files_scroll.setMinimumHeight(100)          # was setFixedHeight(210)

# pipeline status table
self._pipeline_table.setMinimumHeight(60)   # was setMaximumHeight(110)

# catalog table
self._catalog_table.setMinimumHeight(60)    # was setMaximumHeight(130)
```

### 5. `log_viewer.py` — fix text edit height

```python
self._text_edit.setMinimumHeight(120)
# remove: self._text_edit.setMaximumHeight(220)
self._text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
```

### 6. `tracking_widget.py` — fix log height

```python
self._log.setMinimumHeight(100)             # was setFixedHeight(140)
self._log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
```

---

## Risks & Open Questions

- **`sizeHint()` on hidden widgets**: Inner widgets were created but never shown
  (starts collapsed). Qt computes layout sizeHints regardless of visibility, so
  `sizeHint()` should be valid. The `QTimer.singleShot` guard handles edge cases
  where complex children haven't finished initialising.

- **Nested scroll areas in ProjectPanel**: `Data State` section's `_tissue_body`
  contains `files_scroll` (inner scroll for the pipeline file list). After our change,
  `CollapsibleSection` wraps `_tissue_body` in its own outer QScrollArea — so there
  are two nested vertical scroll areas. This is intentional (outer = section, inner =
  file list) but needs testing to confirm no conflicts.

- **Two-level nesting in Cellpose**: `CellposeWidget` contains nested
  `CollapsibleSection` items (3D Nucleus / 2D Cell). Both levels now use the new
  behaviour. Need to verify double-nesting works: outer section's scroll area shows
  the inner sections; inner sections each have their own scroll area + handle.

- **`correction_widget.py` and `tracking_widget.py`** already use `QVBoxLayout(self)`
  directly with `root.addStretch()` — no outer-scroll-area removal needed.

- **`segmentation_widget.py`** (`SegmentationTab`) is **not in the main accordion**
  and requires no changes.

---

## Testing Strategy

1. Expand each accordion section → verify all controls visible without outer scrollbar
2. Sections with log boxes / file tables → verify sub-boxes are visible at min height,
   section is not taller than its controls
3. Drag resize handle **down** → section grows; log/table boxes fill extra space
4. Drag resize handle **up** past natural height → scroll bar appears on section content
5. Collapse → re-expand → section resets to natural height
6. Expand multiple sections simultaneously → they coexist; outer scroll area handles
   overflow when dock panel is small
