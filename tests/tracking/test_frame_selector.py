import numpy as np

import cellflow.tracking.frame_selector as frame_selector
from cellflow.database.hypotheses import (
    HypothesisRecord,
    SeededWatershedParams,
    write_hypothesis_sweep_h5,
)
from cellflow.tracking.frame_selector import (
    compute_frame_stats,
    load_hypothesis_frame_stats,
    select_top_k_paths,
)


def _frame(area: int, label: int = 1, shape=(20, 20)) -> np.ndarray:
    labels = np.zeros((1,) + shape, dtype=np.uint32)
    labels[0, 2:4, 2:2 + area // 2] = label
    return labels


def _volume(areas: list[int], label: int = 1, shape=(20, 20)) -> np.ndarray:
    labels = np.zeros((len(areas),) + shape, dtype=np.uint32)
    for z, area in enumerate(areas):
        labels[z, 2:4, 2:2 + area // 2] = label
    return labels


def _block(row: int, col: int, height: int, width: int, label: int = 1) -> np.ndarray:
    labels = np.zeros((1, 20, 20), dtype=np.uint32)
    labels[0, row:row + height, col:col + width] = label
    return labels


def test_select_top_k_paths_prefers_smooth_full_frame_sequence():
    candidates = [
        [
            compute_frame_stats(_frame(20), t=0, p=0),
            compute_frame_stats(_frame(20), t=0, p=1),
        ],
        [
            compute_frame_stats(_frame(22), t=1, p=0),
            compute_frame_stats(_frame(38), t=1, p=1),
        ],
        [
            compute_frame_stats(_frame(24), t=2, p=0),
            compute_frame_stats(_frame(40), t=2, p=1),
        ],
    ]

    paths = select_top_k_paths(candidates, k=2)

    assert [state.p for state in paths[0].states] == [0, 0, 0]
    assert len(paths) == 2
    assert paths[0].score < paths[1].score


def test_select_top_k_paths_penalizes_missing_seeded_cells():
    frame_with_two_cells = np.zeros((1, 20, 20), dtype=np.uint32)
    frame_with_two_cells[0, 2:6, 2:6] = 1
    frame_with_two_cells[0, 10:14, 10:14] = 2
    frame_missing_one_cell = frame_with_two_cells.copy()
    frame_missing_one_cell[frame_missing_one_cell == 2] = 0

    candidates = [
        [compute_frame_stats(frame_with_two_cells, t=0, p=0)],
        [
            compute_frame_stats(frame_with_two_cells, t=1, p=0),
            compute_frame_stats(frame_missing_one_cell, t=1, p=1),
        ],
    ]

    paths = select_top_k_paths(candidates, k=2)

    assert [state.p for state in paths[0].states] == [0, 0]
    assert [state.p for state in paths[1].states] == [0, 1]
    assert paths[1].transitions[0].missing_count == 1


def test_load_hypothesis_frame_stats_groups_h5_candidates_by_time(tmp_path):
    h5_path = tmp_path / "hypotheses.h5"
    params0 = SeededWatershedParams(foreground_threshold=0.1)
    params1 = SeededWatershedParams(foreground_threshold=0.2)
    records = [
        HypothesisRecord(t=0, p=0, labels=_frame(20, label=1), params=params0),
        HypothesisRecord(t=0, p=1, labels=_frame(30, label=1), params=params1),
        HypothesisRecord(t=1, p=0, labels=_frame(22, label=1), params=params0),
        HypothesisRecord(t=1, p=1, labels=_frame(32, label=1), params=params1),
    ]
    write_hypothesis_sweep_h5(h5_path, records, overwrite=True)

    grouped = load_hypothesis_frame_stats(h5_path)

    assert [[(state.p, state.z) for state in states] for states in grouped] == [[(0, 0), (1, 0)], [(0, 0), (1, 0)]]
    assert grouped[0][0].t == 0
    assert grouped[1][1].foreground_area == 32


def test_load_hypothesis_frame_stats_treats_z_slices_as_alternatives(tmp_path):
    h5_path = tmp_path / "hypotheses.h5"
    params = SeededWatershedParams(foreground_threshold=0.1)
    records = [
        HypothesisRecord(t=0, p=0, labels=_volume([20, 30]), params=params),
        HypothesisRecord(t=1, p=0, labels=_volume([22, 32]), params=params),
    ]
    write_hypothesis_sweep_h5(h5_path, records, overwrite=True)

    grouped = load_hypothesis_frame_stats(h5_path)

    assert [[(state.p, state.z) for state in states] for states in grouped] == [[(0, 0), (0, 1)], [(0, 0), (0, 1)]]
    assert [state.foreground_area for state in grouped[0]] == [20, 30]


def test_select_top_k_paths_uses_shape_continuity_when_area_matches():
    candidates = [
        [
            compute_frame_stats(_block(2, 2, 4, 4), t=0, p=0),
        ],
        [
            compute_frame_stats(_block(3, 3, 4, 4), t=1, p=0),
            compute_frame_stats(_block(3, 3, 1, 16), t=1, p=1),
        ],
    ]

    paths = select_top_k_paths(candidates, k=2)

    assert [state.p for state in paths[0].states] == [0, 0]
    assert paths[0].transitions[0].shape_cost < paths[1].transitions[0].shape_cost


def test_select_top_k_paths_can_bound_active_beam_width():
    candidates = [
        [
            compute_frame_stats(_frame(20), t=0, p=0),
            compute_frame_stats(_frame(30), t=0, p=1),
        ],
        [
            compute_frame_stats(_frame(22), t=1, p=0),
            compute_frame_stats(_frame(32), t=1, p=1),
        ],
    ]

    paths = select_top_k_paths(candidates, k=2, beam_width=1)

    assert len(paths) == 1


def test_select_top_k_paths_reuses_transition_scores_for_duplicate_path_endings(monkeypatch):
    candidates = [
        [compute_frame_stats(_frame(20 + p * 2), t=0, p=p) for p in range(3)],
        [compute_frame_stats(_frame(22 + p * 2), t=1, p=p) for p in range(3)],
        [compute_frame_stats(_frame(24 + p * 2), t=2, p=p) for p in range(3)],
    ]
    original_score_transition = frame_selector.score_transition
    scored_pairs: list[tuple[int, int]] = []

    def counting_score_transition(previous, current, weights):
        scored_pairs.append((id(previous), id(current)))
        return original_score_transition(previous, current, weights)

    monkeypatch.setattr(frame_selector, "score_transition", counting_score_transition)

    select_top_k_paths(candidates, k=5, beam_width=20)

    assert len(scored_pairs) == len(set(scored_pairs))
