"""RunArea — the studio's single Run section (controls + RunChoices)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.config import NlsConfig
from cellflow.napari.aggregate_quantification_run_area import RunArea, RunChoices


def _app():
    return QApplication.instance() or QApplication([])


class _Ctx:
    def __init__(self, records):
        self.records = records


def _area():
    return RunArea(save_callback=lambda c: None, run_callback=lambda c: None)


def test_quantities_default_all_checked():
    app = _app()
    area = _area()
    choices = area.choices()
    assert isinstance(choices, RunChoices)
    assert len(choices.quantities) >= 1  # every registered quantifier, checked
    area.deleteLater()
    app.processEvents()


def test_nls_off_by_default_gives_none():
    app = _app()
    area = _area()
    assert area.choices().nls is None
    area.deleteLater()
    app.processEvents()


def test_nls_enabled_populates_config():
    app = _app()
    area = _area()
    area._nls_enabled.setChecked(True)
    area._nls_image.setText("0_input/NLS_zavg.tif")
    area._nls_method.setCurrentText("otsu")
    nls = area.choices().nls
    assert nls == NlsConfig(enabled=True, image="0_input/NLS_zavg.tif",
                            method="otsu", threshold=0.0)
    area.deleteLater()
    app.processEvents()


def test_plots_choices():
    app = _app()
    area = _area()
    area._render_plots.setChecked(True)
    area._formats.setText("png, pdf")
    choices = area.choices()
    assert choices.render_plots is True
    assert choices.plot_formats == ("png", "pdf")
    area.deleteLater()
    app.processEvents()


def test_buttons_disabled_with_no_records():
    app = _app()
    area = _area()
    area.set_context(_Ctx([]))
    assert not area._run_btn.isEnabled()
    assert not area._save_btn.isEnabled()
    area.set_context(_Ctx([{"id": "p1"}]))
    assert area._run_btn.isEnabled()
    assert area._save_btn.isEnabled()
    area.deleteLater()
    app.processEvents()


def test_zero_quantities_disables_actions():
    app = _app()
    area = _area()
    area.set_context(_Ctx([{"id": "p1"}]))
    for cb in area._quantity_checks.values():
        cb.setChecked(False)
    assert not area._run_btn.isEnabled()
    area.deleteLater()
    app.processEvents()


def test_run_button_invokes_callback_with_choices():
    app = _app()
    seen = []
    area = RunArea(save_callback=lambda c: None,
                   run_callback=lambda c: seen.append(c))
    area.set_context(_Ctx([{"id": "p1"}]))
    area._run_btn.click()
    assert len(seen) == 1 and isinstance(seen[0], RunChoices)
    area.deleteLater()
    app.processEvents()


def test_save_button_invokes_callback_with_choices():
    app = _app()
    seen = []
    area = RunArea(save_callback=lambda c: seen.append(c),
                   run_callback=lambda c: None)
    area.set_context(_Ctx([{"id": "p1"}]))
    area._save_btn.click()
    assert len(seen) == 1 and isinstance(seen[0], RunChoices)
    area.deleteLater()
    app.processEvents()
