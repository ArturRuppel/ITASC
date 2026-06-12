import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cellflow.aggregate_quantification.plotting import (
    DISTRIBUTION_PLOTS,
    PLOT_OPTIONS,
    UNCLASSIFIED,
    PlotSpec,
    PositionSource,
    StyleSpec,
    adaptive_bin_edges,
    aggregate,
    build_figure,
    effective_barrier,
    pickable_points,
    plot_options,
    pool_object_tables,
    potential_landscape,
    potential_table,
    style_from_dict,
    style_to_dict,
    summary_table,
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


def _multiframe_sources():
    """Genuine tracks: each cell spans several frames, so frame-pooling and
    track-reduction give *different* answers. Two dates, two positions each.

    date d1 / position p1 (condition A):
        cell 1 over 3 frames: [10, 14, 12]  -> track mean 12
        cell 2 over 1 frame:  [30]          -> track mean 30
    date d1 / position p2 (condition A):
        cell 1 over 2 frames: [40, 60]      -> track mean 50
    date d2 / position p3 (condition A):
        cell 1 over 4 frames: [100,100,100,100] -> track mean 100
    """
    def tbl(frames, cells, areas):
        return {
            "frame": np.asarray(frames, dtype=np.int64),
            "cell_id": np.asarray(cells, dtype=np.int64),
            "area": np.asarray(areas, dtype=float),
        }

    return [
        PositionSource(
            metadata={"condition": "A", "date": "d1", "position_id": "p1"},
            table=tbl([0, 1, 2, 0], [1, 1, 1, 2], [10.0, 14.0, 12.0, 30.0]),
        ),
        PositionSource(
            metadata={"condition": "A", "date": "d1", "position_id": "p2"},
            table=tbl([0, 1], [1, 1], [40.0, 60.0]),
        ),
        PositionSource(
            metadata={"condition": "A", "date": "d2", "position_id": "p3"},
            table=tbl([0, 1, 2, 3], [1, 1, 1, 1], [100.0, 100.0, 100.0, 100.0]),
        ),
    ]


def test_summary_table_describes_per_unit_values():
    """``summary_table`` reports n/mean/median/sd/sem/min/max over the same
    independent units a distribution draws — tracks, not cell-frames."""
    df = pool_object_tables(_multiframe_sources())
    out = summary_table(
        df, PlotSpec(value="area", group_by=("condition",), level="cell")
    ).set_index("condition")
    units = [12.0, 30.0, 50.0, 100.0]  # the four track means
    assert out.loc["A", "n"] == 4  # tracks, not the 10 frame rows
    assert out.loc["A", "mean"] == pytest.approx(np.mean(units))
    assert out.loc["A", "median"] == pytest.approx(np.median(units))
    assert out.loc["A", "sd"] == pytest.approx(np.std(units, ddof=1))
    assert out.loc["A", "sem"] == pytest.approx(np.std(units, ddof=1) / np.sqrt(4))
    assert out.loc["A", "min"] == 12.0
    assert out.loc["A", "max"] == 100.0


def test_summary_table_single_unit_has_nan_spread():
    df = pool_object_tables(_multiframe_sources())
    # Per date: d2 has a single position/track → its sd / sem are undefined.
    out = summary_table(
        df, PlotSpec(value="area", group_by=("date",), level="date")
    ).set_index("date")
    assert out.loc["d2", "n"] == 1
    assert math.isnan(out.loc["d2", "sd"])
    assert math.isnan(out.loc["d2", "sem"])


def test_group_by_a_nesting_key_does_not_collide():
    """Grouping by ``position_id`` (also a nesting key) must not raise — the
    group axis and the reduction axis are de-duplicated, not inserted twice."""
    df = pool_object_tables(_multiframe_sources())
    spec = PlotSpec(value="area", group_by=("position_id",), level="cell", stat="mean")
    out = aggregate(df, spec).set_index("position_id")
    # p1 has two tracks (means 12, 30) → n=2; p2/p3 one track each.
    assert out.loc["p1", "n"] == 2
    assert out.loc["p2", "n"] == 1
    # Count and summary over the same grouping also survive.
    assert not summary_table(df, spec).empty
    assert not aggregate(df, replace(spec, stat="count")).empty


def test_summary_table_missing_value_is_empty_schema():
    df = pool_object_tables(_multiframe_sources())
    out = summary_table(df, PlotSpec(value="not_a_column", group_by=("condition",)))
    assert list(out.columns) == ["condition", "n", "mean", "median", "sd", "sem", "min", "max"]
    assert out.empty


def test_cell_level_reduces_frames_to_one_value_per_track():
    """``level="cell"`` is per *track*: a cell over N frames is one datapoint, its
    own per-frame mean — never N independent points."""
    df = pool_object_tables(_multiframe_sources())
    out = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="cell", stat="mean")
    ).set_index("condition")
    # 4 tracks: means [12, 30, 50, 100]. n counts tracks (4), not the 10 frames.
    assert out.loc["A", "n"] == 4
    assert out.loc["A", "value"] == pytest.approx(np.mean([12.0, 30.0, 50.0, 100.0]))
    assert out.loc["A", "error"] == pytest.approx(np.std([12, 30, 50, 100], ddof=1))


def test_position_level_equal_weights_tracks_then_positions():
    """Each track collapses to one value, each position to the mean of its tracks,
    then positions are the units — a crowded position is not up-weighted."""
    df = pool_object_tables(_multiframe_sources())
    out = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="position", stat="mean")
    ).set_index("condition")
    # p1 tracks [12, 30] -> 21 ; p2 [50] -> 50 ; p3 [100] -> 100. n = 3 positions.
    assert out.loc["A", "n"] == 3
    assert out.loc["A", "value"] == pytest.approx(np.mean([21.0, 50.0, 100.0]))
    assert out.loc["A", "error"] == pytest.approx(np.std([21, 50, 100], ddof=1))


def test_date_level_is_the_biological_replicate_unit():
    """``level="date"`` climbs frame→track→position→date: n is the number of
    independent replicates, the most conservative comparison unit."""
    df = pool_object_tables(_multiframe_sources())
    out = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="date", stat="mean")
    ).set_index("condition")
    # d1: positions [21, 50] -> 35.5 ; d2: [100] -> 100. n = 2 dates.
    assert out.loc["A", "n"] == 2
    assert out.loc["A", "value"] == pytest.approx(np.mean([35.5, 100.0]))


def test_count_counts_cells_not_cell_frames():
    """A cell tracked over many frames is one cell. Pooled cell count and
    cells-per-position both count distinct tracks, never rows."""
    df = pool_object_tables(_multiframe_sources())
    pooled = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="cell", stat="count")
    ).set_index("condition")
    # 4 distinct tracks across A (not the 10 frame rows).
    assert pooled.loc["A", "value"] == 4

    per_pos = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="position", stat="count")
    ).set_index("condition")
    # cells/position: p1=2, p2=1, p3=1 -> mean 4/3 over 3 positions.
    assert per_pos.loc["A", "n"] == 3
    assert per_pos.loc["A", "value"] == pytest.approx(np.mean([2, 1, 1]))


def test_median_stat_follows_through_both_collapses():
    """``stat="median"`` takes medians at every step, including the within-track
    collapse, not just the final cross-unit summary."""
    df = pool_object_tables(_multiframe_sources())
    out = aggregate(
        df, PlotSpec(value="area", group_by=("condition",), level="cell", stat="median")
    ).set_index("condition")
    # Track medians: cell1@p1 median(10,14,12)=12 ; 30 ; median(40,60)=50 ; 100.
    # Cross-track median of [12,30,50,100] = 40.
    assert out.loc["A", "value"] == pytest.approx(np.median([12.0, 30.0, 50.0, 100.0]))


def test_distribution_points_are_per_unit_not_per_frame():
    """Strip plot exposes one pickable point per track, and each maps back to a
    representative source row whose identity columns load that cell."""
    df = pool_object_tables(_multiframe_sources())
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="strip")
    pts = pickable_points(df, spec, StyleSpec())
    assert len(pts) == 4  # 4 tracks, not 10 frames
    # Each row_index addresses a real pooled row (identity resolves for loading).
    for p in pts:
        assert 0 <= p.row_index < len(df)


def test_pick_resolves_to_a_frame_matching_the_plotted_value():
    """The picked point loads a frame that *looks like* the plotted point: the
    representative row's own value tracks the unit's plotted (reduced) value, not
    an arbitrary first frame.

    Regression: a track whose first frame is tiny but whose mean is the largest in
    the plot used to load that tiny first frame — so clicking the "biggest" point
    loaded a small cell."""
    # cell 1's first frame (10) is the smallest moment of its life, but its mean
    # area (≈340) is the largest unit in the plot.
    def tbl(frames, cells, areas):
        return {
            "frame": np.asarray(frames, dtype=np.int64),
            "cell_id": np.asarray(cells, dtype=np.int64),
            "area": np.asarray(areas, dtype=float),
        }

    sources = [
        PositionSource(
            metadata={"condition": "A", "date": "d1", "position_id": "p1"},
            table=tbl(
                [0, 1, 2, 3, 4, 0, 1, 2],
                [1, 1, 1, 1, 1, 2, 2, 2],
                [10.0, 400.0, 420.0, 430.0, 440.0, 50.0, 55.0, 60.0],
            ),
        ),
    ]
    df = pool_object_tables(sources)
    spec = PlotSpec(value="area", group_by=(), level="cell", plot="strip", stat="mean")
    pts = pickable_points(df, spec, StyleSpec())
    biggest = max(pts, key=lambda p: p.value)
    loaded_area = float(df.iloc[biggest.row_index]["area"])
    # The frame it loads has an area close to the plotted mean — never the tiny
    # first frame (10) the old "first row" rule resolved to.
    assert loaded_area == pytest.approx(biggest.value, abs=100.0)
    assert loaded_area > 300.0


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


@pytest.mark.parametrize("plot", ["hist", "box", "violin", "strip", "swarm", "bar", "line"])
def test_build_figure_missing_value_column_is_safe(plot):
    # A stale build can advertise a column it doesn't carry (e.g. a per-track
    # ``msd_*`` fit added after the .h5 was written). It must render a "No data"
    # placeholder, not KeyError deep in the aggregation.
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="msd_D_um2_per_s", group_by=("condition",), plot=plot)
    fig = build_figure(df, spec)
    assert fig.axes[0].get_title() == "No data in scope"


def test_build_figure_count_needs_no_value_column():
    # ``count`` tallies tracks, so a missing value column is fine for bar/line.
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="msd_D_um2_per_s", group_by=("condition",), plot="bar", stat="count")
    fig = build_figure(df, spec)
    assert fig.axes[0].get_title() != "No data in scope"


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


# ----------------------------------------------------- expanded style fields


def test_spines_and_border_width():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="box")
    ax = build_figure(df, spec, StyleSpec(spines=("left", "bottom"), spine_width=2.5)).axes[0]
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.spines["left"].get_visible() is True
    assert ax.spines["left"].get_linewidth() == 2.5


def test_log_scale_per_axis():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", plot="hist")
    assert build_figure(df, spec, StyleSpec(xlog=True)).axes[0].get_xscale() == "log"
    assert build_figure(df, spec, StyleSpec(ylog=True)).axes[0].get_yscale() == "log"


def test_tick_controls():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="box")
    ax = build_figure(df, spec, StyleSpec(tick_label_size=18.0, tick_length=9.0)).axes[0]
    assert ax.get_xticklabels()[0].get_fontsize() == 18.0
    # The major x ticks were drawn at the requested length.
    assert any(t.tick1line.get_markersize() == 9.0 for t in ax.xaxis.get_major_ticks())


def test_dpi_and_facecolor():
    df = pool_object_tables(_sources())
    fig = build_figure(df, PlotSpec(value="area", plot="hist"),
                       StyleSpec(dpi=200, facecolor="#eeeeee"))
    assert fig.get_dpi() == 200
    assert fig.axes[0].get_facecolor()[:3] == pytest.approx((0.933, 0.933, 0.933), abs=1e-2)


def test_color_override_recolours_one_group():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="bar")
    ax = build_figure(df, spec, StyleSpec(color_overrides=(("A", "#123456"),))).axes[0]
    # #123456 → (0.071, 0.204, 0.337); the A bar takes the override exactly.
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx((0.0706, 0.2039, 0.3373), abs=1e-3)


def test_alpha_applies_to_histogram():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="hist")
    ax = build_figure(df, spec, StyleSpec(alpha=0.2)).axes[0]
    assert ax.patches[0].get_facecolor()[3] == pytest.approx(0.2)


def test_markers_toggle_and_line_width_on_line():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="line")
    on = build_figure(df, spec, StyleSpec(markers=True)).axes[0]
    off = build_figure(df, spec, StyleSpec(markers=False)).axes[0]
    assert on.lines[0].get_marker() == "o"
    assert off.lines[0].get_marker() in ("", "none", None)
    wide = build_figure(df, spec, StyleSpec(line_width=4.0)).axes[0]
    assert wide.lines[0].get_linewidth() == 4.0


def test_grid_and_legend_extras():
    df = pool_object_tables(_sources())
    spec = PlotSpec(value="area", group_by=("condition",), plot="line")
    gridded = build_figure(df, spec, StyleSpec(grid=True, grid_axis="y", grid_linestyle=":"))
    assert gridded.axes[0].yaxis.get_gridlines()[0].get_visible()
    legend = build_figure(
        df, spec, StyleSpec(legend=True, legend_title="Cond", legend_ncol=2),
    ).axes[0].get_legend()
    assert legend.get_title().get_text() == "Cond"
    assert legend._ncols == 2


# --------------------------------------------------------- capability map


def test_plot_options_covers_every_plot_type():
    from cellflow.aggregate_quantification.plotting import _PLOTS
    assert set(PLOT_OPTIONS) == set(_PLOTS)


@pytest.mark.parametrize("plot,expected", [
    ("box", {"box_whis", "box_showfliers", "box_notch"}),
    ("hist", {"bins", "hist_element", "hist_cumulative"}),
    ("potential", {"bins", "adaptive_bins", "markers", "marker_size"}),
    ("swarm", set()),
])
def test_plot_options_per_type(plot, expected):
    assert set(plot_options(plot)) == expected


def test_plot_options_unknown_is_empty():
    assert plot_options("does-not-exist") == ()


# ----------------------------------------------------------- style themes


def test_style_theme_round_trips_through_json():
    import json
    style = StyleSpec(
        dpi=150, alpha=0.3, xlog=True, spines=("left", "bottom"), font_family="serif",
        color_overrides=(("A", "#ff0000"), ("B", "#00ff00")), tick_label_size=14.0,
    )
    restored = style_from_dict(json.loads(json.dumps(style_to_dict(style))))
    assert restored == style
    # tuple-typed fields survive the list round-trip JSON imposes
    assert isinstance(restored.spines, tuple)
    assert isinstance(restored.color_overrides, tuple)


def test_style_from_dict_tolerates_unknown_and_missing_keys():
    style = style_from_dict({"dpi": 222, "totally_unknown_key": 9})
    assert style.dpi == 222  # known key honoured
    assert style.palette == "tab10"  # missing key falls back to default


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
    import numpy as np
    import pandas as pd
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


# -------------------------------------------- pickable points (nearest-in-pixels)
from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR  # noqa: E402


def _tagged(ax):
    """The drawn point collections carrying source rows, with their row arrays.

    The collections are not armed for matplotlib picking; the panel hit-tests
    their offsets in pixel space (nearest wins). The contract here is just that
    each collection carries a row per offset, aligned to it."""
    return [
        (c, list(getattr(c, _PICK_ROWS_ATTR)))
        for c in ax.collections
        if hasattr(c, _PICK_ROWS_ATTR)
    ]


@pytest.mark.parametrize("plot", ["strip", "swarm"])
def test_point_collections_carry_exact_source_rows(plot):
    # Each drawn point stamps its positional .iloc row, in input-row order per
    # category, aligned to its offsets — so a click maps straight to a row.
    df = pd.DataFrame({
        "condition": ["A", "A", "B"],
        "cell_id": [10, 20, 30],
        "area": [1.0, 2.0, 3.0],
    })
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot=plot)
    fig = build_figure(df, spec, StyleSpec())
    tagged = _tagged(fig.axes[0])
    assert tagged  # at least one collection carries rows
    assert sorted(r for _, rows in tagged for r in rows) == [0, 1, 2]
    # offsets line up with rows positionally: same count per collection.
    for col, rows in tagged:
        assert len(np.asarray(col.get_offsets())) == len(rows)


def test_equal_valued_points_keep_distinct_rows():
    # Two identical values in one category must not collapse: native tagging
    # keeps a row per marker (the heuristic this replaces collided them).
    df = pd.DataFrame({
        "condition": ["A", "A"],
        "cell_id": [10, 20],
        "area": [42.0, 42.0],
    })
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="strip")
    fig = build_figure(df, spec, StyleSpec())
    rows = [r for _, rs in _tagged(fig.axes[0]) for r in rs]
    assert sorted(rows) == [0, 1]


def test_box_outlier_overlay_is_pickable_at_the_flier():
    # The box plot exposes only its Tukey fliers, via a transparent overlay
    # scatter sitting on each flier and carrying that row.
    vals = [10, 11, 12, 13, 12, 11, 10, 12, 11, 200]      # 200 is the flier
    df = pd.DataFrame({
        "condition": ["A"] * len(vals),
        "cell_id": list(range(len(vals))),
        "area": [float(v) for v in vals],
    })
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="box")
    fig = build_figure(df, spec, StyleSpec(box_whis=1.5))
    tagged = _tagged(fig.axes[0])
    assert len(tagged) == 1
    col, rows = tagged[0]
    assert rows == [9]
    assert float(np.asarray(col.get_offsets())[0][1]) == 200.0


def test_hidden_fliers_arm_no_box_overlay():
    df = pd.DataFrame({
        "condition": ["A"] * 5,
        "cell_id": list(range(5)),
        "area": [10.0, 11.0, 12.0, 13.0, 200.0],
    })
    spec = PlotSpec(value="area", group_by=("condition",), level="cell", plot="box")
    fig = build_figure(df, spec, StyleSpec(box_showfliers=False))
    assert _tagged(fig.axes[0]) == []


# --------------------------------------------------------------- potential mode


def test_potential_landscape_is_neg_log_p_over_occupied_bins():
    # Three samples in one bin, one in another: P = 3/4 and 1/4 → U = −ln P.
    values = np.array([0.1, 0.1, 0.1, 0.9])
    centers, u, counts = potential_landscape(values, bins=2, value_range=(0.0, 1.0))
    assert counts.tolist() == [3, 1]
    assert u[0] == pytest.approx(-math.log(0.75))
    assert u[1] == pytest.approx(-math.log(0.25))
    assert centers.tolist() == [0.25, 0.75]


def test_potential_landscape_drops_empty_bins():
    # Middle bin is empty → dropped (U → ∞), so only two occupied bins remain.
    values = np.array([0.0, 0.0, 1.0])
    centers, u, counts = potential_landscape(values, bins=3, value_range=(0.0, 1.0))
    assert counts.tolist() == [2, 1]
    assert len(centers) == 2 and np.isfinite(u).all()


def test_potential_landscape_empty_input_returns_empty():
    centers, u, counts = potential_landscape(np.array([np.nan, np.nan]), bins=10)
    assert centers.size == 0 and u.size == 0 and counts.size == 0


def test_effective_barrier_bimodal_is_positive_and_finite():
    rng = np.random.RandomState(0)
    values = np.concatenate([rng.normal(-3, 0.4, 4000), rng.normal(3, 0.4, 4000)])
    centers, u, _ = potential_landscape(values, bins=41, value_range=(-5, 5))
    barrier = effective_barrier(centers, u)
    # The well minima sit near ±3; the rarely-visited centre (x=0) is the barrier.
    assert np.isfinite(barrier) and barrier > 1.0


def test_effective_barrier_nan_when_zero_not_bracketed():
    # All samples positive → 0 is left of the occupied range → undefined barrier.
    values = np.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5])
    centers, u, _ = potential_landscape(values, bins=6)
    assert math.isnan(effective_barrier(centers, u))


def test_potential_pools_raw_samples_not_units():
    # A table whose nesting keys (position_id) would collapse a distribution plot
    # to one value per position. The potential mode must ignore that and bin every
    # raw sample, so its curve has many occupied bins, not one point per position.
    rng = np.random.RandomState(1)
    df = pd.DataFrame({
        "position_id": ["p1"] * 500 + ["p2"] * 500,
        "signed_length": np.concatenate([rng.normal(-2, 0.5, 500), rng.normal(2, 0.5, 500)]),
    })
    spec = PlotSpec(value="signed_length", plot="potential", bins=30)
    table = potential_table(df, spec)
    assert table["group"].nunique() == 1  # ungrouped → "all"
    assert len(table) > 10  # a full curve, not 2 per-position points


def test_potential_table_one_block_per_group_with_barrier():
    rng = np.random.RandomState(2)
    both = np.concatenate([rng.normal(-2, 0.4, 1500), rng.normal(2, 0.4, 1500)])
    df = pd.DataFrame({
        "condition": ["A"] * 3000 + ["B"] * 3000,
        "signed_length": np.concatenate([both, both]),
    })
    spec = PlotSpec(value="signed_length", plot="potential", group_by=("condition",), bins=31)
    table = potential_table(df, spec)
    assert set(table.columns) == {"group", "center", "U", "counts", "delta_e_eff"}
    assert set(table["group"]) == {"A", "B"}
    # Each group spans both wells → a finite, repeated-down-the-block barrier.
    for _, block in table.groupby("group"):
        assert block["delta_e_eff"].nunique() == 1
        assert np.isfinite(block["delta_e_eff"].iloc[0])


def test_build_figure_potential_draws_one_curve_per_group_with_barrier_label():
    rng = np.random.RandomState(3)
    both = np.concatenate([rng.normal(-2, 0.4, 1500), rng.normal(2, 0.4, 1500)])
    df = pd.DataFrame({
        "condition": ["A"] * 3000 + ["B"] * 3000,
        "signed_length": np.concatenate([both, both]),
    })
    spec = PlotSpec(value="signed_length", plot="potential", group_by=("condition",), bins=31)
    fig = build_figure(df, spec, StyleSpec())
    ax = fig.axes[0]
    assert len(ax.lines) >= 2  # one curve per group (+ the x=0 marker line)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(label.startswith("A (ΔE_eff=") for label in labels)
    assert ax.get_ylabel() == "U = −ln P  [kT]"


def test_build_figure_potential_missing_value_is_placeholder():
    df = pd.DataFrame({"condition": ["A", "B"]})
    spec = PlotSpec(value="signed_length", plot="potential", bins=10)
    fig = build_figure(df, spec, StyleSpec())
    assert fig.axes[0].get_title() == "No data in scope"


def test_pickable_points_potential_is_empty():
    df = pd.DataFrame({"signed_length": [1.0, -1.0]})
    spec = PlotSpec(value="signed_length", plot="potential")
    assert pickable_points(df, spec, StyleSpec()) == []


# ----------------------------------------------------------- adaptive binning


def test_adaptive_bin_edges_are_tighter_near_zero():
    edges = adaptive_bin_edges(-10.0, 10.0, 20)
    assert len(edges) == 21
    # Endpoints preserved, strictly increasing.
    assert edges[0] == -10.0 and edges[-1] == 10.0
    assert np.all(np.diff(edges) > 0)
    widths = np.diff(edges)
    # The central bins (straddling 0) are the narrowest; the outermost the widest.
    central = widths[len(widths) // 2 - 1]
    assert central == widths.min()
    assert widths[0] > central and widths[-1] > central


def test_adaptive_bin_edges_symmetric_about_zero_for_symmetric_range():
    edges = adaptive_bin_edges(-5.0, 5.0, 10)
    np.testing.assert_allclose(edges, -edges[::-1], atol=1e-9)


def test_adaptive_bin_edges_low_sharpness_approaches_uniform():
    adaptive = adaptive_bin_edges(-4.0, 4.0, 16, sharpness=1e-6)
    uniform = np.linspace(-4.0, 4.0, 17)
    np.testing.assert_allclose(adaptive, uniform, atol=1e-3)


def test_adaptive_bin_edges_degenerate_range_is_safe():
    edges = adaptive_bin_edges(2.0, 2.0, 5)  # zero-width range
    assert len(edges) == 6 and np.all(np.diff(edges) > 0)


def test_potential_adaptive_mode_resolves_more_bins_near_zero():
    # A coordinate concentrated near 0 with sparse tails: adaptive binning puts
    # more occupied bins in |x|<1 than uniform binning at the same bin count.
    rng = np.random.RandomState(7)
    values = np.concatenate([rng.normal(0, 0.3, 4000), rng.uniform(-8, 8, 400)])
    df = pd.DataFrame({"signed_length": values})
    near = lambda tbl: int((tbl["center"].abs() < 1.0).sum())
    uni = potential_table(df, PlotSpec(value="signed_length", plot="potential", bins=30, bin_mode="uniform"))
    ada = potential_table(df, PlotSpec(value="signed_length", plot="potential", bins=30, bin_mode="adaptive"))
    assert near(ada) > near(uni)


def test_build_figure_potential_adaptive_renders():
    rng = np.random.RandomState(8)
    df = pd.DataFrame({"signed_length": np.concatenate([rng.normal(-2, 0.5, 2000), rng.normal(2, 0.5, 2000)])})
    spec = PlotSpec(value="signed_length", plot="potential", bins=31, bin_mode="adaptive")
    fig = build_figure(df, spec, StyleSpec())
    assert len(fig.axes[0].lines) >= 1


def test_plotspec_rejects_unknown_bin_mode():
    with pytest.raises(ValueError):
        PlotSpec(value="x", plot="potential", bin_mode="logarithmic")


# --------------------------------------------------------- plotted_table (one CSV export)
def _drawn_point_count(ax) -> int:
    """Total scatter points drawn across the axes' collections."""
    total = 0
    for coll in ax.collections:
        offs = np.asarray(coll.get_offsets(), dtype=float)
        if offs.ndim == 2:
            total += offs.shape[0]
    return total


def test_plotted_table_distribution_is_per_unit_values():
    from cellflow.aggregate_quantification.plotting import plotted_table

    df = pool_object_tables(_multiframe_sources())
    spec = PlotSpec(value="area", plot="strip", level="cell", group_by=("condition",))
    table = plotted_table(df, spec)
    # One row per cell track (frames collapsed), carrying the group + value only.
    assert list(table.columns) == ["condition", "area"]
    assert len(table) == 4  # four distinct cell tracks
    assert set(np.round(sorted(table["area"]), 0)) == {12.0, 30.0, 50.0, 100.0}


def test_plotted_table_bar_is_the_aggregate():
    from cellflow.aggregate_quantification.plotting import aggregate, plotted_table

    df = pool_object_tables(_multiframe_sources())
    spec = PlotSpec(value="area", plot="bar", group_by=("condition",))
    pd.testing.assert_frame_equal(plotted_table(df, spec), aggregate(df, spec))


def test_plotted_table_line_is_per_frame_series():
    from cellflow.aggregate_quantification.plotting import plotted_table

    df = pool_object_tables(_multiframe_sources())
    spec = PlotSpec(value="area", plot="line", stat="mean")
    table = plotted_table(df, spec)
    assert list(table.columns) == ["group", "frame", "value"]
    # frames 0..3 are present across the pooled positions.
    assert sorted(table["frame"].unique()) == [0, 1, 2, 3]


def test_plotted_table_potential_is_the_curve():
    from cellflow.aggregate_quantification.plotting import plotted_table, potential_table

    df = pd.DataFrame({"signed_length": np.linspace(-3, 3, 500)})
    spec = PlotSpec(value="signed_length", plot="potential", bins=20)
    pd.testing.assert_frame_equal(plotted_table(df, spec), potential_table(df, spec))


# ------------------------------------------------------------- swarm overflow fallback
def test_swarm_overflow_draws_every_point_without_warning(recwarn):
    # A column crowded enough that a swarm cannot place all markers: the backend
    # must fall back (auto-size, then stripplot) so no datapoint is dropped.
    n = 600
    df = pd.DataFrame({
        "condition": ["A"] * n,
        "position_id": [f"p{i}" for i in range(n)],
        "cell_id": list(range(n)),
        "frame": [0] * n,
        "area": np.linspace(0.0, 1.0, n),
    })
    spec = PlotSpec(value="area", plot="swarm", level="cell")
    style = StyleSpec(width=2.0, height=2.0)
    fig = build_figure(df, spec, style)
    ax = fig.axes[0]
    # Every per-cell point is drawn (no silent omission).
    assert _drawn_point_count(ax) == n
    # No "cannot be placed" overflow warning escapes the backend.
    assert not any("cannot be placed" in str(w.message) for w in recwarn.list)
