# Plan: Replace Tab Widget with Collapsible Accordion Layout

## Approach

Replace the top-level `QTabWidget` in `CellFlowWidget` with a vertically stacked accordion
of collapsible sections — one per current tab. Each section gets a `QToolButton` toggle header
(with a right/down arrow) that shows or hides the inner widget. The `ProjectPanel` sections
(Pipeline Project, Tissue, Dataset) are likewise converted from plain `QGroupBox` to the same
collapsible pattern.

Rather than repeating the toggle-button pattern 12+ times inline, a small `CollapsibleSection`
helper widget is extracted first. This helper already exists in spirit inside
`segmentation_widget.py` (the "Cellpose Parameters" and "Guided Segmentation" toggles) — it
just needs to be promoted to a shared location.

Alternatives considered:
- **Keep `QTabWidget`, add collapsible inner sections** — already partly done; doesn't address
  the top-level navigation complaint.
- **Use a third-party accordion widget** — adds a dependency; the QToolButton pattern is
  idiomatic Qt, already in use in this codebase, and trivial to implement.

---

## Files Affected

- `packages/napari-plugin/src/cellflow/napari/widgets.py` *(new file)*
  Shared `CollapsibleSection` widget used by both `analysis_widget.py` and `project_panel.py`.

- `packages/napari-plugin/src/cellflow/napari/analysis_widget.py`
  - Remove `QTabWidget` import and `tab_widget` construction.
  - Import and use `CollapsibleSection` for each of the 8 stage sections.
  - Rewrite `_refresh_tab_badges()` to update `CollapsibleSection` header text instead of
    calling `tab_widget.setTabText()`.
  - Remove `_badge_timer` interval (still needed, just target changes).

- `packages/napari-plugin/src/cellflow/napari/project_panel.py`
  - Replace the three `QGroupBox` sections (Pipeline Project, Tissue, Dataset) with
    `CollapsibleSection` instances.
  - Pipeline Project: starts expanded.
  - Tissue: starts expanded.
  - Dataset: starts collapsed (saves vertical space by default).
  - Remove `QGroupBox` from imports if no longer used.

- `packages/napari-plugin/src/cellflow/napari/segmentation_widget.py` *(minor cleanup)*
  - The two inline toggle patterns ("Cellpose Parameters", "Guided Segmentation") can
    optionally be replaced with `CollapsibleSection` to remove duplication. Low priority —
    keep for a follow-up if desired.

---

## Implementation Steps

1. **Create `widgets.py`** with `CollapsibleSection`:
   ```python
   class CollapsibleSection(QWidget):
       def __init__(self, title: str, inner: QWidget, expanded: bool = False, parent=None):
           ...
       def set_title(self, title: str) -> None:
           """Update header text (used by badge refresh)."""
           ...
       @property
       def title(self) -> str: ...
   ```
   The toggle button uses `Qt.RightArrow` / `Qt.DownArrow` and `setToolButtonStyle(Qt.ToolButtonTextBesideIcon)`.
   A thin `QFrame` line sits below the header for visual separation.

2. **Refactor `analysis_widget.py`**:
   - Remove `QTabWidget` and all `addTab()` calls.
   - Instantiate each stage widget as before, then wrap each in `CollapsibleSection`.
   - First section (Data Prep) starts expanded; rest start collapsed.
   - Store sections in `self._sections: dict[str, CollapsibleSection]` keyed by base title
     (same keys as `TAB_STAGE_KEYS`).
   - Rewrite `_refresh_tab_badges()`:
     - Iterate `self._sections.items()` instead of `tab_widget.count()`.
     - Call `section.set_title(base_title + badge)` instead of `setTabText()`.

3. **Refactor `project_panel.py`**:
   - Import `CollapsibleSection` from `.widgets`.
   - Replace the `pipeline_group = QGroupBox("Pipeline Project")` block with
     `CollapsibleSection("Pipeline Project", pipeline_inner, expanded=True)`.
   - Same for Tissue and Dataset.
   - `self.dataset_widget` is currently exposed as the `QGroupBox` and added to
     `_plugin_layout` in `analysis_widget.py`. After the refactor it becomes the
     `CollapsibleSection` wrapper — update the reference in `analysis_widget.py`
     accordingly.

4. **Wire and test manually** in napari:
   - Confirm all 8 stage sections render and collapse/expand correctly.
   - Confirm badge text updates on the correct section headers.
   - Confirm ProjectPanel sections collapse independently.
   - Confirm Ctrl+S and other keybinds still work (they're on `viewer`, not the tab widget).

---

## Risks & Open Questions

- **`self.dataset_widget` coupling**: `ProjectPanel.dataset_widget` is referenced directly
  in `analysis_widget.py` (`_plugin_layout.addWidget(self._project_panel.dataset_widget)`).
  If the Dataset section is folded into `ProjectPanel`'s own layout (all sections together),
  this reference disappears and `analysis_widget.py` no longer needs to place it separately.
  That simplification is clean but requires adjusting both files. **Decision needed**: should
  Dataset remain below the stage sections (current position) or move inside the ProjectPanel
  block above them?

- **Vertical space**: With everything stacked, the scroll area becomes the primary navigation
  mechanism. The outer `QScrollArea` already exists, so this should work fine — but it's worth
  testing with a small screen or docked panel to make sure nothing is unusably cramped when
  multiple sections are expanded simultaneously.

- **Badge strip**: The current badge appears in the tab title. In the accordion, it will appear
  in the section header button text. The visual weight may feel different — consider bolding
  the header text or using a colored indicator label next to the button if plain text badges
  look weak.

- **No automated UI tests**: There are 179 passing core tests but no widget-level tests. This
  change is verified manually only.

---

## Testing Strategy

Manual verification in napari:
1. Plugin loads without errors; all 8 sections are present and show correct labels.
2. Clicking a section header expands/collapses it; arrow icon flips correctly.
3. Badge text (✓ / ✗ / ↻ / ⚠) appears on correct section header after opening a project.
4. ProjectPanel: Pipeline Project, Tissue, Dataset each collapse independently.
5. `dataset_widget` is still visible below the stage sections (or inside ProjectPanel,
   per the decision above).
6. Ctrl+S still triggers save (keybind on viewer, unaffected by layout change).
7. All existing core tests still pass: `uv run pytest packages/core/`.
