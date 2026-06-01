from __future__ import annotations

import pandas as pd

from cellflow.tracking_ultrack.track_quality import (
    compute_track_scores,
    quality_order,
)


def _tracks_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["id", "t", "track_id"])


def test_compute_track_scores_sums_node_probs_and_edge_weights() -> None:
    # Track 1: nodes 10 (t0) -> 11 (t1) -> 12 (t2).
    tracks_df = _tracks_df(
        [
            (10, 0, 1),
            (11, 1, 1),
            (12, 2, 1),
        ]
    )
    node_probs = {10: 0.5, 11: 0.25, 12: 0.75}
    links = {(10, 11): (0.6, "REAL"), (11, 12): (0.4, "REAL")}

    scores = compute_track_scores(tracks_df, node_probs, links)

    # nodes: 0.5 + 0.25 + 0.75 = 1.5 ; edges: 0.6 + 0.4 = 1.0
    assert scores == {1: 2.5}


def test_null_node_prob_and_null_weight_render_as_one() -> None:
    tracks_df = _tracks_df([(1, 0, 7), (2, 1, 7)])
    node_probs = {1: None, 2: 0.5}
    links = {(1, 2): (None, "REAL")}

    scores = compute_track_scores(tracks_df, node_probs, links)

    # node 1 NULL -> 1.0, node 2 -> 0.5, edge NULL -> 1.0
    assert scores == {7: 2.5}


def test_fake_link_contributes_zero() -> None:
    tracks_df = _tracks_df([(1, 0, 3), (2, 1, 3)])
    node_probs = {1: 0.5, 2: 0.5}
    links = {(1, 2): (0.9, "FAKE")}

    scores = compute_track_scores(tracks_df, node_probs, links)

    # FAKE edge ignored -> only the two node probs.
    assert scores == {3: 1.0}


def test_missing_link_contributes_zero() -> None:
    tracks_df = _tracks_df([(1, 0, 4), (2, 1, 4)])
    node_probs = {1: 0.3, 2: 0.3}

    scores = compute_track_scores(tracks_df, node_probs, links={})

    assert scores == {4: 0.6}


def test_nodes_summed_in_temporal_order_regardless_of_row_order() -> None:
    # Rows out of temporal order; edges keyed earlier->later frame.
    tracks_df = _tracks_df([(12, 2, 1), (10, 0, 1), (11, 1, 1)])
    node_probs = {10: 0.1, 11: 0.1, 12: 0.1}
    links = {(10, 11): (1.0, "REAL"), (11, 12): (1.0, "REAL")}

    scores = compute_track_scores(tracks_df, node_probs, links)

    assert scores == {1: 2.3}


def test_quality_order_sorts_by_score_desc_then_id_asc() -> None:
    scores = {3: 1.0, 1: 5.0, 2: 5.0, 4: 2.0}

    # 1 and 2 tie at 5.0 -> id asc; then 4 (2.0); then 3 (1.0).
    assert quality_order(scores) == [1, 2, 4, 3]


def test_quality_order_empty() -> None:
    assert quality_order({}) == []
