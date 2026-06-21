"""The TOML run-config: the hand-authored 'author once, then run' knob file."""
from __future__ import annotations

from pathlib import Path

import pytest

from cellflow.aggregate_quantification.config import RunConfig, load_config


def _write(tmp: Path, text: str, name: str = "config.toml") -> Path:
    path = tmp / name
    path.write_text(text)
    return path


def test_load_minimal_config_defaults(tmp_path):
    """A bare config naming only the catalog gets sensible defaults: every
    quantity and a flat ``export`` dir beside the config."""
    cfg_path = _write(tmp_path, 'catalog = "catalog.csv"\n')

    cfg = load_config(cfg_path)

    assert isinstance(cfg, RunConfig)
    assert cfg.catalog == (tmp_path / "catalog.csv").resolve()
    assert cfg.export_dir == (tmp_path / "export").resolve()
    assert cfg.quantities == ()  # empty = run every available quantifier
    assert cfg.params == {}
    # Plot rendering is off by default; the run stays Iris-only (no engine dep).
    assert cfg.render_plots is False
    assert cfg.plot_formats == ("png", "svg")


def test_plots_table_parsed(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        catalog = "cat.csv"

        [plots]
        render = true
        formats = ["pdf", "png"]
        """,
    )

    cfg = load_config(cfg_path)

    assert cfg.render_plots is True
    assert cfg.plot_formats == ("pdf", "png")


def test_quantities_and_params_parsed(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        catalog = "cat.csv"
        quantities = ["cell_shape", "neighbor_count"]

        [params]
        pixel_size_um = 0.65
        frame_interval_min = 10
        """,
    )

    cfg = load_config(cfg_path)

    assert cfg.quantities == ("cell_shape", "neighbor_count")
    assert cfg.params == {"pixel_size_um": 0.65, "frame_interval_min": 10}


def test_relative_paths_resolve_against_config_dir(tmp_path):
    sub = tmp_path / "proj"
    sub.mkdir()
    cfg_path = _write(
        sub,
        """
        catalog = "data/catalog.csv"
        export_dir = "out"
        """,
    )

    cfg = load_config(cfg_path)

    assert cfg.catalog == (sub / "data/catalog.csv").resolve()
    assert cfg.export_dir == (sub / "out").resolve()


def test_absolute_paths_kept(tmp_path):
    abs_cat = tmp_path / "elsewhere" / "catalog.csv"
    cfg_path = _write(tmp_path, f'catalog = "{abs_cat}"\n')

    cfg = load_config(cfg_path)

    assert cfg.catalog == abs_cat.resolve()


def test_missing_catalog_key_raises(tmp_path):
    cfg_path = _write(tmp_path, 'quantities = ["cell_shape"]\n')

    with pytest.raises(ValueError, match="catalog"):
        load_config(cfg_path)


def test_unknown_quantity_raises(tmp_path):
    cfg_path = _write(
        tmp_path, 'catalog = "c.csv"\nquantities = ["cell_shape", "bogus_metric"]\n'
    )

    with pytest.raises(ValueError, match="bogus_metric"):
        load_config(cfg_path)
