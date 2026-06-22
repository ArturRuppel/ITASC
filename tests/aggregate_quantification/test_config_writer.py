"""write_config — the inverse of load_config (round-trips through TOML)."""
from __future__ import annotations

from cellflow.aggregate_quantification.config import (
    NlsConfig,
    load_config,
    write_config,
)


def test_round_trip_minimal(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, catalog="catalog.csv", quantities=["contacts"])
    cfg = load_config(path)
    assert cfg.catalog == (tmp_path / "catalog.csv").resolve()
    assert cfg.export_dir == (tmp_path / "export").resolve()
    assert cfg.curation == (tmp_path / "curation.csv").resolve()
    assert cfg.quantities == ("contacts",)
    assert cfg.nls is None
    assert cfg.render_plots is False


def test_params_drop_unset_keys(tmp_path):
    path = tmp_path / "config.toml"
    write_config(
        path,
        catalog="catalog.csv",
        params={"pixel_size_um": 0.25, "time_interval_s": None,
                "fov_area_mm2": None, "shuffles": 1000},
    )
    cfg = load_config(path)
    assert cfg.params == {"pixel_size_um": 0.25, "shuffles": 1000}


def test_nls_table_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    write_config(
        path,
        catalog="catalog.csv",
        nls=NlsConfig(enabled=True, image="0_input/NLS_zavg.tif",
                      method="fixed", threshold=12.5),
    )
    cfg = load_config(path)
    assert cfg.nls == NlsConfig(enabled=True, image="0_input/NLS_zavg.tif",
                                method="fixed", threshold=12.5)


def test_plots_table_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    write_config(path, catalog="catalog.csv", render_plots=True,
                 plot_formats=["png", "pdf"])
    cfg = load_config(path)
    assert cfg.render_plots is True
    assert cfg.plot_formats == ("png", "pdf")


def test_quantities_empty_means_all(tmp_path):
    """No quantities key -> load_config reads () -> 'every quantifier'."""
    path = tmp_path / "config.toml"
    write_config(path, catalog="catalog.csv", quantities=[])
    assert "quantities" not in path.read_text()
    assert load_config(path).quantities == ()


def test_string_escaping(tmp_path):
    """Backslashes / quotes in a path survive the round trip."""
    path = tmp_path / "config.toml"
    write_config(path, catalog=r'weird"\name.csv')
    assert load_config(path).catalog.name == r'weird"\name.csv'


def test_returns_written_path(tmp_path):
    path = tmp_path / "config.toml"
    assert write_config(path, catalog="catalog.csv") == path
