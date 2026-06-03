"""Application-wide UI gating for the CellFlow napari plugin.

Historically each widget rolled its own ``setEnabled`` logic from local
booleans, and a handful of overlapping guards (the nucleus "viewer activity"
guard, the main-widget position lock, per-widget running/preview flags) tried
to coordinate without knowing about each other. The result was incoherent:
config-load could move the position out from under an in-progress correction,
harmless ⚙ params buttons got disabled, and the nucleus section was guarded
while the cell section was not.

``UiGate`` replaces all of that with a single source of truth. The one
contended resource is **write/auto-update ownership of the viewer**: a
correction session, the database browser, and every live preview are all
*viewer owners*, and only one may be active at a time. Headless jobs (writing
to disk only) run independently of ownership.

Every gated control is registered once with a :class:`ControlClass`; on each
state transition :meth:`UiGate.recompute` derives ``enabled`` (and a tooltip)
for every control from a single policy table. Widgets never call ``setEnabled``
on gated controls themselves — they call :meth:`claim_viewer`,
:meth:`release_viewer`, :meth:`set_task`, or :meth:`recompute` and let the gate
apply the policy.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMessageBox, QWidget

ReasonSource = Union[str, Callable[[], str], None]


class ControlClass(Enum):
    """How a control relates to viewer ownership and the data context.

    The class — not the widget it lives in — determines when a control is
    enabled. See the policy table in :meth:`UiGate._enabled_for`.
    """

    #: A checkable control that *enters* an exclusive viewer-owner state
    #: (correction, db-browser, a live preview). Mutually exclusive: enabled
    #: only when nothing else owns the viewer, or this control owns it.
    VIEWER_OWNER = auto()

    #: Launches a job that mutates state an active viewer owner depends on
    #: (live previews, pipeline rebuilds of the data being viewed). Blocked
    #: while any viewer owner is active.
    RUN_VIEWER = auto()

    #: Launches a disk-only job that no active owner is viewing. Independent of
    #: viewer ownership; gated only by its own ``when`` predicate.
    RUN_HEADLESS = auto()

    #: Swaps the underlying data (position, project directory, Load Config).
    #: Always *enabled*; the danger is guarded at click time via
    #: :meth:`UiGate.confirm_context_change`, which offers to exit the owner
    #: first.
    CONTEXT_CHANGING = auto()

    #: No viewer/context side effect (⚙ params dialogs, Save Config, param
    #: spinboxes). Always enabled.
    HARMLESS = auto()

    #: Only meaningful inside one owner (correction cleanup/outline/goto).
    #: Enabled only while that owner is active and its ``when`` holds.
    MODE_LOCAL = auto()


@dataclass
class _Registration:
    control: object
    klass: ControlClass
    owner_token: Optional[str] = None
    when: Optional[Callable[[], bool]] = None
    reason: ReasonSource = None


@dataclass
class _Owner:
    label: str
    exit_fn: Callable[[], None]


@dataclass
class _GateState:
    owner: Optional[str] = None
    tasks: set[str] = field(default_factory=set)
    owners: dict[str, _Owner] = field(default_factory=dict)


class UiGate(QObject):
    """Single source of truth for control enablement across the plugin."""

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = _GateState()
        self._registrations: list[_Registration] = []
        self._gated_disabled: set[int] = set()
        self._idle_tooltips: dict[int, str] = {}
        #: Overridable in tests; returns True if the user confirms exiting the
        #: active owner. Defaults to a modal yes/no dialog.
        self.confirm_handler: Callable[[Optional[QWidget], str], bool] = (
            self._default_confirm
        )

    # ── Registration ──────────────────────────────────────────────────────
    def register(
        self,
        control,
        klass: ControlClass,
        *,
        owner_token: str | None = None,
        when: Callable[[], bool] | None = None,
        reason: ReasonSource = None,
    ) -> None:
        """Register *control* under *klass*.

        ``owner_token`` ties a ``VIEWER_OWNER``/``MODE_LOCAL`` control to a
        viewer-owner token. ``when`` is an optional zero-arg predicate for
        additional readiness (e.g. inputs present, stage not running);
        ``reason`` is the tooltip shown while the control is gate-disabled.
        """
        self._registrations.append(
            _Registration(control, klass, owner_token, when, reason)
        )

    def register_owner(
        self, token: str, label: str, exit_fn: Callable[[], None]
    ) -> None:
        """Describe a viewer-owner token: a human label and how to leave it."""
        self._state.owners[token] = _Owner(label, exit_fn)

    # ── State transitions ─────────────────────────────────────────────────
    @property
    def owner(self) -> str | None:
        return self._state.owner

    def owner_label(self) -> str | None:
        """Human label of the active viewer owner, or ``None`` when idle."""
        owner = self._state.owner
        info = self._state.owners.get(owner) if owner else None
        return info.label if info else None

    def is_busy(self) -> bool:
        return bool(self._state.tasks)

    def claim_viewer(self, token: str) -> None:
        """Mark *token* as the active viewer owner and recompute."""
        if self._state.owner == token:
            return
        self._state.owner = token
        self.recompute()

    def release_viewer(self, token: str) -> None:
        """Clear ownership if *token* currently owns the viewer."""
        if self._state.owner != token:
            return
        self._state.owner = None
        self.recompute()

    def set_task(self, token: str, running: bool) -> None:
        """Track a running job (headless or viewer-writing) and recompute."""
        before = len(self._state.tasks)
        if running:
            self._state.tasks.add(token)
        else:
            self._state.tasks.discard(token)
        if len(self._state.tasks) != before or running:
            self.recompute()

    # ── Context-change guard ──────────────────────────────────────────────
    def can_change_context(self) -> bool:
        """True when no viewer owner is active, so a context change is safe."""
        return self._state.owner is None

    def confirm_context_change(
        self, parent, action: Callable[[], None]
    ) -> bool:
        """Run *action*, first offering to exit the active owner if any.

        Returns ``True`` if *action* ran, ``False`` if the user declined.
        """
        owner = self._state.owner
        if owner is None:
            action()
            return True
        info = self._state.owners.get(owner)
        label = info.label if info else "the current mode"
        if not self.confirm_handler(parent, label):
            return False
        if info is not None:
            info.exit_fn()
        action()
        return True

    @staticmethod
    def _default_confirm(parent, label: str) -> bool:
        reply = QMessageBox.question(
            parent,
            f"Exit {label}?",
            f"This will exit {label}. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    # ── Policy ────────────────────────────────────────────────────────────
    def _predicate(self, reg: _Registration) -> bool:
        if reg.when is None:
            return True
        try:
            return bool(reg.when())
        except Exception:
            return True

    def _enabled_for(self, reg: _Registration) -> bool:
        owner = self._state.owner
        klass = reg.klass
        if klass is ControlClass.HARMLESS or klass is ControlClass.CONTEXT_CHANGING:
            return True
        if klass is ControlClass.VIEWER_OWNER:
            return (owner is None or owner == reg.owner_token) and self._predicate(reg)
        if klass is ControlClass.RUN_VIEWER:
            return owner is None and self._predicate(reg)
        if klass is ControlClass.RUN_HEADLESS:
            return self._predicate(reg)
        if klass is ControlClass.MODE_LOCAL:
            return owner == reg.owner_token and self._predicate(reg)
        return True

    def _reason_text(self, reg: _Registration) -> str:
        if callable(reg.reason):
            try:
                return reg.reason()
            except Exception:
                pass
        elif isinstance(reg.reason, str):
            return reg.reason
        owner = self._state.owner
        info = self._state.owners.get(owner) if owner else None
        if info is not None:
            if reg.klass is ControlClass.VIEWER_OWNER:
                return f"Exit {info.label} first."
            return f"Unavailable while {info.label} is active."
        return "Currently unavailable."

    def recompute(self) -> None:
        """Apply the policy to every registered control."""
        for reg in self._registrations:
            control = reg.control
            key = id(control)
            enabled = self._enabled_for(reg)
            if enabled:
                control.setEnabled(True)
                if key in self._gated_disabled:
                    control.setToolTip(self._idle_tooltips.pop(key, control.toolTip()))
                    self._gated_disabled.discard(key)
            else:
                if key not in self._gated_disabled:
                    self._idle_tooltips[key] = control.toolTip()
                    self._gated_disabled.add(key)
                control.setEnabled(False)
                control.setToolTip(self._reason_text(reg))
        self.changed.emit()
