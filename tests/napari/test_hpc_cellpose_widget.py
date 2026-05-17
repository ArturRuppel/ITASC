"""Tests for the standalone HPC Cellpose napari widget."""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


PIPELINE_SCRIPT = Path("/home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh")
DEFAULT_CONFIG = Path(
    "/home/aruppel/Projects/HPC/cellpose_full/cellpose_full.json"
)
FORBIDDEN_AUTH_MATERIAL = (
    "IdentityFile",
    "SSH_AUTH_SOCK",
    "id_rsa",
    "id_ed25519",
    "--identity",
    " -i ",
)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = {}
        self.dims = SimpleNamespace(current_step=(0,))


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.hpc_cellpose_widget", None)
    return importlib.import_module("cellflow.napari.hpc_cellpose_widget")


def _make_widget(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.HpcCellposeWidget(_FakeViewer())
    return app, mod, widget


def _text(control) -> str:
    if hasattr(control, "text"):
        return control.text()
    return str(control)


def _set_text(control, value: str) -> None:
    control.setText(value)


def _set_check(control, value: bool) -> None:
    control.setChecked(value)


def _set_value(control, value) -> None:
    control.setValue(value)


def _assert_no_auth_material(payload: str) -> None:
    for forbidden in FORBIDDEN_AUTH_MATERIAL:
        assert forbidden not in payload


def test_widget_defaults_expose_pipeline_controls(monkeypatch):
    app, _mod, widget = _make_widget(monkeypatch)

    assert widget.pipeline_script_path == PIPELINE_SCRIPT
    assert _text(widget.config_path_edit) == str(DEFAULT_CONFIG)
    assert _text(widget.nuclei_input_edit) == "nucleus_3dt.tif"
    assert _text(widget.cells_input_edit) == "cell_3dt.tif"
    assert _text(widget.frames_edit) == "all"
    assert widget.nuclei_do_3d_check.isChecked() is False
    assert widget.nuclei_anisotropy_spin.value() == 1.5
    assert widget.nuclei_diameter_spin.value() == 25
    assert widget.nuclei_size_spin.value() == 0
    assert widget.nuclei_gamma_spin.value() == 1.0
    assert widget.cells_size_spin.value() == 0
    assert widget.cells_gamma_spin.value() == 1.0
    assert widget.max_concurrent_jobs_spin.value() == 4
    assert _text(widget.remote_user_edit) == "aruppel"
    assert _text(widget.remote_host_edit) == "maestro.pasteur.fr"

    widget.deleteLater()
    app.processEvents()


def test_refresh_derives_input_and_output_dirs_from_position_dir(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)

    pos_dir = tmp_path / "pos07"
    pos_dir.mkdir()
    widget.refresh(pos_dir)

    assert _text(widget.input_dir_edit) == str(pos_dir / "0_input")
    assert _text(widget.output_dir_edit) == str(pos_dir / "1_cellpose")

    widget.deleteLater()
    app.processEvents()


def test_get_set_state_round_trips_hpc_cellpose_controls(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)
    input_dir = tmp_path / "custom_input"
    output_dir = tmp_path / "custom_output"
    config = tmp_path / "cellpose_full.json"

    state = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config_path": str(config),
        "nuclei_input": "nuclei_custom.tif",
        "cells_input": "cells_custom.tif",
        "frames": "0,2,5",
        "nuclei_do_3d": True,
        "nuclei_anisotropy": 2.25,
        "nuclei_diameter": 31,
        "nuclei_size": 17,
        "nuclei_gamma": 1.4,
        "cells_size": 19,
        "cells_gamma": 0.8,
        "max_concurrent_jobs": 6,
        "remote_user": "science-user",
        "remote_host": "maestro.example.org",
    }

    widget.set_state(state)

    assert widget.get_state() == state

    widget.deleteLater()
    app.processEvents()


def test_missing_inputs_prevent_terminal_launch(monkeypatch, tmp_path):
    app, mod, widget = _make_widget(monkeypatch)
    launched: list[str] = []
    monkeypatch.setattr(mod, "launch_in_terminal", launched.append)

    script = tmp_path / "run_pipeline.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    config = tmp_path / "cellpose_full.json"
    config.write_text("{}", encoding="utf-8")
    input_dir = tmp_path / "0_input"
    input_dir.mkdir()

    widget.pipeline_script_path = script
    _set_text(widget.input_dir_edit, str(input_dir))
    _set_text(widget.output_dir_edit, str(tmp_path / "1_cellpose"))
    _set_text(widget.config_path_edit, str(config))
    _set_text(widget.nuclei_input_edit, "nucleus_3dt.tif")
    _set_text(widget.cells_input_edit, "cell_3dt.tif")

    widget._on_run_terminal()

    assert launched == []
    assert "missing" in widget.status_lbl.text().lower()

    widget.deleteLater()
    app.processEvents()


def test_successful_launch_writes_runtime_json_and_calls_terminal(monkeypatch, tmp_path):
    app, mod, widget = _make_widget(monkeypatch)
    launched: list[str] = []
    monkeypatch.setattr(mod, "launch_in_terminal", launched.append)

    script = tmp_path / "run pipeline.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    config = tmp_path / "cellpose full.json"
    config.write_text("{}", encoding="utf-8")
    input_dir = tmp_path / "0_input"
    output_dir = tmp_path / "1_cellpose"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "nucleus_3dt.tif").write_bytes(b"nucleus")
    (input_dir / "cell_3dt.tif").write_bytes(b"cell")

    widget.pipeline_script_path = script
    _set_text(widget.input_dir_edit, str(input_dir))
    _set_text(widget.output_dir_edit, str(output_dir))
    _set_text(widget.config_path_edit, str(config))
    _set_text(widget.frames_edit, "0,3")
    _set_check(widget.nuclei_do_3d_check, True)
    _set_value(widget.nuclei_anisotropy_spin, 2.5)
    _set_value(widget.nuclei_diameter_spin, 33)
    _set_value(widget.nuclei_size_spin, 12)
    _set_value(widget.nuclei_gamma_spin, 1.3)
    _set_value(widget.cells_size_spin, 21)
    _set_value(widget.cells_gamma_spin, 0.7)
    _set_value(widget.max_concurrent_jobs_spin, 5)
    _set_text(widget.remote_user_edit, "aruppel")
    _set_text(widget.remote_host_edit, "maestro.pasteur.fr")

    widget._on_run_terminal()

    assert len(launched) == 1
    command = launched[0]
    assert "bash" in command
    assert str(script) in command
    assert "--input-dir" in command
    assert str(input_dir) in command
    assert "--output-dir" in command
    assert str(output_dir) in command
    assert "--nuclei-input" in command
    assert "nucleus_3dt.tif" in command
    assert "--cells-input" in command
    assert "cell_3dt.tif" in command
    assert "--max-concurrent-jobs" in command
    assert "5" in command
    assert "--remote-user" in command
    assert "aruppel" in command
    assert "--remote-host" in command
    assert "maestro.pasteur.fr" in command

    config_arg = command.split("--config", 1)[1].split()[0].strip("'\"")
    runtime_config_path = Path(config_arg)
    runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    assert runtime_config == {
        "input_dir": str(input_dir),
        "frames": "0,3",
        "nuclei": {
            "input": "nucleus_3dt.tif",
            "do_3d": True,
            "anisotropy": 2.5,
            "diameter": 33,
            "size": 12,
            "gamma": 1.3,
        },
        "cells": {
            "input": "cell_3dt.tif",
            "size": 21,
            "gamma": 0.7,
        },
    }

    widget.deleteLater()
    app.processEvents()


def test_command_and_runtime_config_exclude_ssh_auth_material(monkeypatch, tmp_path):
    app, mod, widget = _make_widget(monkeypatch)
    launched: list[str] = []
    monkeypatch.setattr(mod, "launch_in_terminal", launched.append)

    script = tmp_path / "run_pipeline.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    config = tmp_path / "cellpose_full.json"
    config.write_text("{}", encoding="utf-8")
    input_dir = tmp_path / "0_input"
    output_dir = tmp_path / "1_cellpose"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "nucleus_3dt.tif").write_bytes(b"nucleus")
    (input_dir / "cell_3dt.tif").write_bytes(b"cell")

    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ssh-agent.sock")
    widget.pipeline_script_path = script
    _set_text(widget.input_dir_edit, str(input_dir))
    _set_text(widget.output_dir_edit, str(output_dir))
    _set_text(widget.config_path_edit, str(config))

    widget._on_run_terminal()

    assert len(launched) == 1
    command = launched[0]
    config_arg = command.split("--config", 1)[1].split()[0].strip("'\"")
    runtime_config_text = Path(config_arg).read_text(encoding="utf-8")
    state_text = json.dumps(widget.get_state(), sort_keys=True)

    _assert_no_auth_material(command)
    _assert_no_auth_material(runtime_config_text)
    _assert_no_auth_material(state_text)

    widget.deleteLater()
    app.processEvents()
