"""author_config — write catalog.csv + config.toml, ready for run()."""
from __future__ import annotations

from cellflow.aggregate_quantification.config import NlsConfig, load_config
from cellflow.aggregate_quantification.pipeline import author_config


def _record(tmp_path, pid="p1"):
    pdir = tmp_path / "study" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    return {"id": pid, "condition": "ctrl", "date": "2026-06-22",
            "position_path": pdir, "notes": ""}


def test_writes_both_files_into_out_dir(tmp_path):
    out = tmp_path / "study"
    records = [_record(tmp_path)]
    config_path = author_config(out, records, quantities=["contacts"])
    assert config_path == out / "config.toml"
    assert (out / "catalog.csv").is_file()
    cfg = load_config(config_path)
    assert cfg.catalog == (out / "catalog.csv").resolve()
    assert cfg.quantities == ("contacts",)


def test_creates_missing_out_dir(tmp_path):
    out = tmp_path / "fresh"
    author_config(out, [_record(tmp_path)], quantities=["contacts"])
    assert (out / "config.toml").is_file()


def test_threads_knobs_into_config(tmp_path):
    out = tmp_path / "study"
    config_path = author_config(
        out, [_record(tmp_path)],
        quantities=["contacts"],
        params={"pixel_size_um": 0.25, "shuffles": 1000},
        nls=NlsConfig(enabled=True, method="auto"),
        render_plots=True, plot_formats=["png"],
    )
    cfg = load_config(config_path)
    assert cfg.params == {"pixel_size_um": 0.25, "shuffles": 1000}
    assert cfg.nls is not None and cfg.nls.enabled is True
    assert cfg.render_plots is True and cfg.plot_formats == ("png",)
