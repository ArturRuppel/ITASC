import math

import numpy as np
import pandas as pd
import pytest

from cellflow.aggregate_quantification.plotting import (
    DISTRIBUTION_PLOTS,
    UNCLASSIFIED,
    PlotSpec,
    PositionSource,
    StyleSpec,
    aggregate,
    build_figure,
    pool_object_tables,
    write_csv,
)


def _shape_table(cell_ids, areas, frame=0):
    n = len(cell_ids)
    return {
        "frame": np.full(n, frame, dtype=np.int64),
        "cell_id": np.asarray(cell_ids, dtype=np.int64),
        "area": np.asarray(areas, dtype=float),
    }


def _sources():
    """3 positions: p1/p2 in condition A, p3 in B; p3 has no contacts join."""
    return [
        PositionSource(
            metadata={"condition": "A", "date": "d1", "position_id": "p1"},
            table=_shape_table([1, 2, 3], [10.0, 20.0, 30.0]),
            join_table={
                "frame": np.array([0, 0, 0]),
                "cell_id": np.array([1, 2, 3]),
                "class_label": np.array(["pos", "neg", "pos"], dtype=object),
            },
            join_columns=("class_label",),
        ),
        PositionSource(
            metadata={"condition": "A", "date": "d1", "position_id": "p2"},
            table=_shape_table([1, 2], [40.0, 50.0]),
        ),
        PositionSource(
            metadata={"condition": "B", "date": "d2", "position_id": "p3"},
            table=_shape_table([1, 2], [100.0, 200.0]),
        ),
    ]


def test_pool_prepends_metadata_and_joins_class_per_position():
    df = pool_object_tables(_sources())

    assert len(df) == 7  # 3 + 2 + 2 cells
    assert set(["condition", "date", "position_id", "frame", "cell_id", "area"]) <= set(df.columns)
    # p1 carried a class join; p2/p3 did not -> those rows are "unclassified".
    p1 = df[df["position_id"] == "p1"].sort_values("cell_id")
    assert p1["class_label"].tolist() == ["pos", "neg", "pos"]
    assert set(df[df["position_id"] == "p2"]["class_label"]) == {UNCLASSIFIED}
    assert set(df[df["position_id"] == "p3"]["class_label"]) == {UNCLASSIFIED}


def test_pool_empty_sources_is_empty_frame():
    assert pool_object_tables([]).empty


def test_aggregate_per_position_count_is_mean_cells_per_tissue():
    df = pool_object_tables(_sources())
    out = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="position", stat="count")
    ).set_index("condition")

    # Condition A: positions p1=3, p2=2 cells -> mean 2.5, n=2 positions, sd(3,2).
    assert out.loc["A", "value"] == pytest.approx(2.5)
    assert out.loc["A", "n"] == 2
    assert out.loc["A", "error"] == pytest.approx(np.std([3, 2], ddof=1))
    # Condition B: a single position (2 cells) -> mean 2, n=1, sd undefined.
    assert out.loc["B", "value"] == pytest.approx(2.0)
    assert out.loc["B", "n"] == 1
    assert math.isnan(out.loc["B", "error"])


def test_aggregate_pooled_cell_mean_and_count():
    df = pool_object_tables(_sources())
    means = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="cell", stat="mean")
    ).set_index("condition")
    # Condition A pooled cells: [10,20,30,40,50] -> mean 30 over 5 cells.
    assert means.loc["A", "value"] == pytest.approx(30.0)
    assert means.loc["A", "n"] == 5
    assert means.loc["A", "error"] == pytest.approx(np.std([10, 20, 30, 40, 50], ddof=1))
    assert means.loc["B", "value"] == pytest.approx(150.0)

    counts = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="cell", stat="count")
    ).set_index("condition")
    assert counts.loc["A", "value"] == 5
    assert counts.loc["B", "value"] == 2


def test_aggregate_empty_returns_schema():
    out = aggregate(pd.DataFrame(), PlotSpec(value="area", group_by=("condition",)))
    assert list(out.columns) == ["condition", "n", "value", "error"]
    assert out.empty


@pytest.mark.parametrize("plot", ["hist", "box", "violin", "strip", "swarm", "bar", "line"])
def test_build_figure_renders_each_plot_type_headless(tmp_path, plot):
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot=plot)
    fig = build_figure(df, spec)
    # Savable without a display (Agg canvas attached in build_figure).
    out = tmp_path / f"{plot}.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_build_figure_empty_is_safe():
    fig = build_figure(pd.DataFrame(), PlotSpec(value="area"))
    assert fig is not None


def test_distribution_plots_are_the_seaborn_family():
    # strip/swarm are the new scatter members; bar/line stay custom matplotlib.
    assert DISTRIBUTION_PLOTS == ("hist", "box", "violin", "strip", "swarm")


def test_style_defaults_reproduce_auto_title_and_size():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", plot="hist")
    fig = build_figure(df, spec)  # default StyleSpec
    ax = fig.axes[0]
    assert ax.get_title() == "area"  # auto title = value column
    assert fig.get_size_inches().tolist() == [6.0, 4.0]


def test_each_style_field_measurably_changes_the_figure():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="bar")

    base = build_figure(df, spec, StyleSpec())
    base_ax = base.axes[0]

    # Title override.
    titled = build_figure(df, spec, StyleSpec(title="Custom"))
    assert titled.axes[0].get_title() == "Custom"

    # Figure dimensions.
    sized = build_figure(df, spec, StyleSpec(width=9.0, height=3.0))
    assert sized.get_size_inches().tolist() == [9.0, 3.0]

    # Font size drives the title font.
    big = build_figure(df, spec, StyleSpec(font_size=24.0))
    assert big.axes[0].title.get_fontsize() > base_ax.title.get_fontsize()

    # Palette changes the bar colors.
    other = build_figure(df, spec, StyleSpec(palette="Set1"))
    base_color = base_ax.patches[0].get_facecolor()
    other_color = other.axes[0].patches[0].get_facecolor()
    assert base_color != other_color

    # Axis label overrides.
    labelled = build_figure(df, spec, StyleSpec(xlabel="XX", ylabel="YY"))
    assert labelled.axes[0].get_xlabel() == "XX"
    assert labelled.axes[0].get_ylabel() == "YY"


def test_legend_can_be_turned_off():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="line")
    with_legend = build_figure(df, spec, StyleSpec(legend=True))
    without = build_figure(df, spec, StyleSpec(legend=False))
    assert with_legend.axes[0].get_legend() is not None
    assert without.axes[0].get_legend() is None


def test_bar_still_honors_position_level_aggregation():
    # The pseudoreplication guard must survive: per-position count for condition A
    # is mean cells/tissue (2.5), not the pooled 5 cells.
    df = pool_object_tables(_sources())
    spec = PlotSpec(
        value="area", group_by=("condition",), plot="bar", level="position", stat="count"
    )
    fig = build_figure(df, spec)
    summary = aggregate(df, spec).set_index("condition")
    heights = [p.get_height() for p in fig.axes[0].patches]
    assert pytest.approx(summary.loc["A", "value"]) == 2.5
    assert any(h == pytest.approx(2.5) for h in heights)


def test_plotspec_rejects_bad_enums():
    with pytest.raises(ValueError):
        PlotSpec(value="area", plot="pie")


def test_write_csv_round_trips_and_forces_suffix(tmp_path):
    df = pool_object_tables(_sources())
    written = write_csv(df, tmp_path / "pooled")  # no suffix
    assert written.name == "pooled.csv"
    reloaded = pd.read_csv(written)
    assert len(reloaded) == len(df)
    assert "area" in reloaded.columns
