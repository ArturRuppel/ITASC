"""The ``stack_relabeled`` event is the full post-relabel refresh.

Reassign-IDs / remove-unvalidated / commit / anchor rewrite the label stack, so
they emit ``stack_relabeled`` and the full display set repaints: recolour,
validated overlay, validation counter and a whole-stack lineage rebuild.
"""

from __future__ import annotations


def test_stack_relabeled_does_the_full_refresh_set(wired_stub):
    stub = wired_stub()

    stub.events.stack_relabeled.emit()

    stub._refresh_correction_label_visuals.assert_called_once_with()
    stub._refresh_validated_overlay.assert_called_once_with()
    stub._refresh_validation_counter.assert_called_once_with()
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
