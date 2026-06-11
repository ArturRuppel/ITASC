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


def test_pool_joins_cell_id_only_label_broadcasts_across_frames():
    """An NLS-style join table keyed on ``cell_id`` alone (one row per track)
    broadcasts its label across every frame of that cell."""
    source = PositionSource(
        metadata={"condition": "A", "date": "d1", "position_id": "p1"},
        table={
            "frame": np.array([0, 1, 0, 1], dtype=np.int64),
            "cell_id": np.array([1, 1, 2, 2], dtype=np.int64),
            "area": np.array([10.0, 11.0, 20.0, 21.0]),
        },
        join_table={
            "cell_id": np.array([1, 2]),
            "class_label": np.array(["pos", "neg"], dtype=object),
        },
        join_columns=("class_label",),
    )
    df = pool_object_tables([source]).sort_values(["cell_id", "frame"])
    assert df[df["cell_id"] == 1]["class_label"].tolist() == ["pos", "pos"]
    assert df[df["cell_id"] == 2]["class_label"].tolist() == ["neg", "neg"]


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


def _box_df():
    """A two-group frame whose group A carries a single high outlier, so the
    box-plot knobs have something to act on."""
    return pd.DataFrame({
        "condition": ["A"] * 8 + ["B"] * 8,
        "area": [10, 11, 12, 13, 14, 15, 16, 100, 20, 21, 22, 23, 24, 25, 26, 27.0],
    })


def _flier_count(ax) -> int:
    """Outlier markers a box plot drew — fliers are marker-only Line2D artists."""
    return sum(
        len(line.get_ydata())
        for line in ax.lines
        if line.get_marker() not in ("", "none", None)
    )


def test_box_outliers_can_be_hidden():
    df = _box_df()
    spec = PlotSpec(value="area", group_by=("condition",), plot="box")
    shown = build_figure(df, spec, StyleSpec(box_showfliers=True))
    hidden = build_figure(df, spec, StyleSpec(box_showfliers=False))
    assert _flier_count(shown.axes[0]) > 0
    assert _flier_count(hidden.axes[0]) == 0


def test_box_wider_whiskers_swallow_outliers():
    # Stretching the whiskers to the full range leaves nothing flagged as an outlier.
    df = _box_df()
    spec = PlotSpec(value="area", group_by=("condition",), plot="box")
    tukey = build_figure(df, spec, StyleSpec(box_whis=1.5))
    full = build_figure(df, spec, StyleSpec(box_whis=100.0))
    assert _flier_count(tukey.axes[0]) > _flier_count(full.axes[0])


def test_box_notch_changes_box_geometry():
    # A notched box is a richer polygon (the median pinch adds vertices).
    df = _box_df()
    spec = PlotSpec(value="area", group_by=("condition",), plot="box")
    plain = build_figure(df, spec, StyleSpec(box_notch=False))
    notched = build_figure(df, spec, StyleSpec(box_notch=True))
    plain_verts = max(len(p.get_path().vertices) for p in plain.axes[0].patches)
    notched_verts = max(len(p.get_path().vertices) for p in notched.axes[0].patches)
    assert notched_verts > plain_verts


def test_box_defaults_match_tukey_look():
    style = StyleSpec()
    assert style.box_whis == 1.5
    assert style.box_showfliers is True
    assert style.box_notch is False


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


def test_axis_limits_applied_when_set():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, build_figure,
    )
    import pandas as pd
    df = pd.DataFrame({"condition": ["A", "A", "B"], "frame": [0, 1, 0],
                       "cell_id": [1, 2, 1], "position_id": ["p", "p", "p"],
                       "area": [10.0, 20.0, 30.0]})
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="strip")
    style = StyleSpec(xmin=None, xmax=None, ymin=5.0, ymax=40.0)
    ax = build_figure(df, spec, style).axes[0]
    assert ax.get_ylim() == (5.0, 40.0)


def test_axis_limits_default_to_auto():
    from cellflow.aggregate_quantification.plotting import StyleSpec
    s = StyleSpec()
    assert s.xmin is None and s.xmax is None and s.ymin is None and s.ymax is None


def test_pickable_points_strip_is_one_per_finite_row():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, pickable_points,
    )
    import numpy as np, pandas as pd
    df = pd.DataFrame({"condition": ["A", "A", "B"], "area": [1.0, np.nan, 3.0]})
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="strip")
    pts = pickable_points(df, spec, StyleSpec())
    assert {p.row_index for p in pts} == {0, 2}            # NaN row dropped
    assert {p.category for p in pts} == {"A", "B"}
    assert next(p for p in pts if p.row_index == 0).value == 1.0


def test_pickable_points_box_is_outliers_only():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, pickable_points,
    )
    import pandas as pd
    vals = [10, 11, 12, 13, 12, 11, 10, 12, 11, 200]      # 200 is the flier
    df = pd.DataFrame({"condition": ["A"] * len(vals), "area": [float(v) for v in vals]})
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="box")
    pts = pickable_points(df, spec, StyleSpec(box_whis=1.5))
    assert [p.row_index for p in pts] == [9]


def test_pickable_points_hist_is_empty():
    from cellflow.aggregate_quantification.plotting import (
        PlotSpec, StyleSpec, pickable_points,
    )
    import pandas as pd
    df = pd.DataFrame({"area": [1.0, 2.0]})
    spec = PlotSpec(value="area", group_by=(), level="cell", plot="hist")
    assert pickable_points(df, spec, StyleSpec()) == []
