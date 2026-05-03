"""Consensus-label movie helpers for cell-boundary hypothesis sweeps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class ConsensusMember:
    p: int
    compactness: float
    foreground_threshold: float
    basin: str


@dataclass(frozen=True, slots=True)
class CompactnessGroup:
    compactness: float
    members: tuple[ConsensusMember, ...]


@dataclass(frozen=True, slots=True)
class ConsensusMovie:
    labels: np.ndarray
    support: np.ndarray
    thresholds: np.ndarray


def vote_consensus_labels(
    labels: np.ndarray,
    *,
    vote_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel majority labels and their vote support.

    ``labels`` is an integer array with shape ``(n_votes, y, x)``. Label IDs are
    treated as categories, not numeric intensities.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (n_votes, y, x), got {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("Expected at least one vote plane")
    if not 0.0 <= vote_threshold <= 1.0:
        raise ValueError("vote_threshold must be between 0 and 1")

    sorted_stack = np.sort(stack, axis=0)
    best_label = sorted_stack[0].copy()
    current_label = sorted_stack[0].copy()
    best_count = np.ones(sorted_stack.shape[1:], dtype=np.uint16)
    current_count = np.ones(sorted_stack.shape[1:], dtype=np.uint16)

    for plane in sorted_stack[1:]:
        same_label = plane == current_label
        current_count = np.where(same_label, current_count + 1, 1)
        current_label = np.where(same_label, current_label, plane)
        better = current_count > best_count
        best_count = np.where(better, current_count, best_count)
        best_label = np.where(better, current_label, best_label)

    support = (best_count.astype(np.float32) / float(stack.shape[0])).astype(np.float32)
    consensus = np.where(support >= vote_threshold, best_label, 0).astype(stack.dtype, copy=False)
    return consensus, support


def collapse_z_by_label_presence(labels: np.ndarray) -> np.ndarray:
    """Collapse a z-stack by non-background label presence.

    Background z-slices do not vote unless every z-slice is background. If
    multiple non-background labels occur at a pixel, the most frequent label
    wins with the same deterministic tie behavior as ``vote_consensus_labels``.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (z, y, x), got {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("Expected at least one z-slice")

    sentinel = np.iinfo(stack.dtype).max if np.issubdtype(stack.dtype, np.integer) else -1
    foreground_votes = np.where(stack > 0, stack, sentinel)
    collapsed, _support = vote_consensus_labels(foreground_votes, vote_threshold=0.0)
    return np.where(collapsed == sentinel, 0, collapsed).astype(stack.dtype, copy=False)


def vote_label_footprints(
    labels: np.ndarray,
    *,
    vote_threshold: float = 0.0,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vote across 2D label footprints while ignoring background as a competitor."""
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (n_votes, y, x), got {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("Expected at least one footprint vote")
    if not 0.0 <= vote_threshold <= 1.0:
        raise ValueError("vote_threshold must be between 0 and 1")
    if weights is None:
        vote_weights = np.ones(stack.shape[0], dtype=np.float32)
    else:
        vote_weights = np.asarray(weights, dtype=np.float32)
        if vote_weights.shape != (stack.shape[0],):
            raise ValueError(f"Expected {stack.shape[0]} weights, got shape {vote_weights.shape}")
        if np.any(vote_weights < 0):
            raise ValueError("weights must be non-negative")
        if not np.any(vote_weights > 0):
            raise ValueError("At least one weight must be positive")

    labels_out = np.zeros(stack.shape[1:], dtype=stack.dtype)
    best_count = np.zeros(stack.shape[1:], dtype=np.float32)
    for label in np.unique(stack):
        if label == 0:
            continue
        count = np.sum(
            (stack == label) * vote_weights[:, np.newaxis, np.newaxis],
            axis=0,
            dtype=np.float32,
        )
        better = count > best_count
        labels_out = np.where(better, label, labels_out)
        best_count = np.where(better, count, best_count)

    support_out = (best_count.astype(np.float32) / float(np.sum(vote_weights))).astype(
        np.float32,
        copy=False,
    )
    labels_out = np.where(support_out >= vote_threshold, labels_out, 0).astype(
        stack.dtype,
        copy=False,
    )
    return labels_out, support_out


def apply_vote_thresholds(
    labels: np.ndarray,
    support: np.ndarray,
    thresholds: float | np.ndarray,
) -> np.ndarray:
    """Set labels below scalar or per-frame support thresholds to background."""
    label_stack = np.asarray(labels)
    support_stack = np.asarray(support, dtype=np.float32)
    if label_stack.shape != support_stack.shape:
        raise ValueError(f"Expected support shape {label_stack.shape}, got {support_stack.shape}")

    threshold_arr = np.asarray(thresholds, dtype=np.float32)
    if threshold_arr.ndim == 0:
        threshold_view = threshold_arr
    elif threshold_arr.ndim == 1 and label_stack.ndim == 3:
        if threshold_arr.shape[0] != label_stack.shape[0]:
            raise ValueError(
                f"Expected {label_stack.shape[0]} frame thresholds, got {threshold_arr.shape[0]}"
            )
        threshold_view = threshold_arr[:, np.newaxis, np.newaxis]
    elif threshold_arr.shape == label_stack.shape:
        threshold_view = threshold_arr
    else:
        raise ValueError(
            "thresholds must be scalar, one value per frame, or match the label shape"
        )
    return np.where(support_stack >= threshold_view, label_stack, 0).astype(
        label_stack.dtype,
        copy=False,
    )


def resolve_vote_thresholds(
    support: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str = "fixed",
    vote_threshold: float = 0.5,
    percentile: float = 60.0,
    min_threshold: float = 0.35,
    max_threshold: float = 0.65,
) -> np.ndarray:
    """Return one support threshold per frame."""
    support_stack = np.asarray(support, dtype=np.float32)
    label_stack = np.asarray(labels)
    if support_stack.ndim != 3:
        raise ValueError(f"Expected support with shape (t, y, x), got {support_stack.shape}")
    if label_stack.shape != support_stack.shape:
        raise ValueError(f"Expected labels shape {support_stack.shape}, got {label_stack.shape}")
    if mode == "fixed":
        _validate_threshold(vote_threshold, "vote_threshold")
        return np.full(support_stack.shape[0], vote_threshold, dtype=np.float32)
    if mode != "percentile":
        raise ValueError(f"Unknown threshold mode: {mode}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    _validate_threshold(min_threshold, "min_threshold")
    _validate_threshold(max_threshold, "max_threshold")
    if min_threshold > max_threshold:
        raise ValueError("min_threshold must be <= max_threshold")

    thresholds = np.zeros(support_stack.shape[0], dtype=np.float32)
    for t in range(support_stack.shape[0]):
        values = support_stack[t][label_stack[t] > 0]
        if values.size == 0:
            threshold = vote_threshold
        else:
            threshold = float(np.percentile(values, percentile))
        thresholds[t] = np.float32(np.clip(threshold, min_threshold, max_threshold))
    return thresholds


def smooth_consensus_labels(
    labels: np.ndarray,
    support: np.ndarray,
    *,
    vote_threshold: float = 0.5,
    weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
) -> tuple[np.ndarray, np.ndarray]:
    """Temporally smooth top-voted consensus labels with a 3-frame window."""
    label_stack = np.asarray(labels)
    support_stack = np.asarray(support, dtype=np.float32)
    if label_stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (t, y, x), got {label_stack.shape}")
    if support_stack.shape != label_stack.shape:
        raise ValueError(
            f"Expected support shape {label_stack.shape}, got {support_stack.shape}"
        )
    if len(weights) != 3:
        raise ValueError("weights must contain previous/current/next weights")
    if not 0.0 <= vote_threshold <= 1.0:
        raise ValueError("vote_threshold must be between 0 and 1")

    output = np.zeros_like(label_stack)
    output_support = np.zeros(label_stack.shape, dtype=np.float32)
    base_weights = np.asarray(weights, dtype=np.float32)

    for t in range(label_stack.shape[0]):
        window_labels = []
        window_scores = []
        active_weights = []
        for offset, weight in zip((-1, 0, 1), base_weights):
            idx = t + offset
            if 0 <= idx < label_stack.shape[0]:
                window_labels.append(label_stack[idx])
                window_scores.append(support_stack[idx])
                active_weights.append(float(weight))

        norm = float(sum(active_weights))
        candidate_labels = np.stack(window_labels, axis=0)
        candidate_scores = (
            np.asarray(active_weights, dtype=np.float32)[:, np.newaxis, np.newaxis]
            * np.stack(window_scores, axis=0)
            / norm
        )

        best_label = candidate_labels[0].copy()
        best_score = _score_candidate_label(candidate_labels, candidate_scores, best_label)
        for candidate in candidate_labels[1:]:
            score = _score_candidate_label(candidate_labels, candidate_scores, candidate)
            better = score > best_score
            best_label = np.where(better, candidate, best_label)
            best_score = np.where(better, score, best_score)

        output_support[t] = best_score.astype(np.float32, copy=False)
        output[t] = np.where(best_score >= vote_threshold, best_label, 0)

    return output, output_support


def load_compactness_groups(path: str | Path) -> list[CompactnessGroup]:
    """Return hypothesis parameter groups keyed by watershed compactness."""
    groups: dict[float, list[ConsensusMember]] = {}
    with h5py.File(Path(path), "r") as h5:
        root = h5["hypotheses"]
        first_t = sorted(k for k in root.keys() if k.startswith("t"))[0]
        for p_key in sorted(k for k in root[first_t].keys() if k.startswith("p")):
            p = int(p_key[1:])
            attrs = root[first_t][p_key].attrs
            compactness = float(attrs.get("compactness", 0.0))
            member = ConsensusMember(
                p=p,
                compactness=compactness,
                foreground_threshold=float(attrs.get("foreground_threshold", 0.0)),
                basin=str(attrs.get("basin", "")),
            )
            groups.setdefault(compactness, []).append(member)

    return [
        CompactnessGroup(
            compactness=compactness,
            members=tuple(
                sorted(
                    members,
                    key=lambda member: (member.foreground_threshold, member.basin, member.p),
                )
            ),
        )
        for compactness, members in sorted(groups.items())
    ]


def build_consensus_movie(
    path: str | Path,
    group: CompactnessGroup,
    *,
    vote_threshold: float = 0.5,
    smooth_temporally: bool = True,
    temporal_weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
    threshold_mode: str = "fixed",
    threshold_percentile: float = 60.0,
    min_vote_threshold: float = 0.35,
    max_vote_threshold: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one consensus label movie for a compactness group."""
    movie = build_consensus_movie_with_thresholds(
        path,
        group,
        vote_threshold=vote_threshold,
        smooth_temporally=smooth_temporally,
        temporal_weights=temporal_weights,
        threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
        min_vote_threshold=min_vote_threshold,
        max_vote_threshold=max_vote_threshold,
    )
    return movie.labels, movie.support


def build_consensus_movie_with_thresholds(
    path: str | Path,
    group: CompactnessGroup,
    *,
    vote_threshold: float = 0.5,
    smooth_temporally: bool = True,
    temporal_weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
    threshold_mode: str = "fixed",
    threshold_percentile: float = 60.0,
    min_vote_threshold: float = 0.35,
    max_vote_threshold: float = 0.65,
) -> ConsensusMovie:
    """Build one consensus label movie and return per-frame thresholds."""
    raw_labels = []
    raw_support = []
    with h5py.File(Path(path), "r") as h5:
        root = h5["hypotheses"]
        for t_key in sorted(k for k in root.keys() if k.startswith("t")):
            vote_planes = []
            for member in group.members:
                labels = root[t_key][f"p{member.p:03d}"]["labels"][:]
                vote_planes.extend(labels[z] for z in range(labels.shape[0]))
            consensus, support = vote_consensus_labels(
                np.stack(vote_planes, axis=0),
                vote_threshold=0.0,
            )
            raw_labels.append(consensus)
            raw_support.append(support)

    label_movie = np.stack(raw_labels, axis=0)
    support_movie = np.stack(raw_support, axis=0)
    if smooth_temporally:
        label_movie, support_movie = smooth_consensus_labels(
            label_movie,
            support_movie,
            vote_threshold=0.0,
            weights=temporal_weights,
        )
    thresholds = resolve_vote_thresholds(
        support_movie,
        label_movie,
        mode=threshold_mode,
        vote_threshold=vote_threshold,
        percentile=threshold_percentile,
        min_threshold=min_vote_threshold,
        max_threshold=max_vote_threshold,
    )
    thresholded_labels = apply_vote_thresholds(label_movie, support_movie, thresholds)
    return ConsensusMovie(
        labels=thresholded_labels,
        support=support_movie,
        thresholds=thresholds,
    )


def _score_candidate_label(
    candidate_labels: np.ndarray,
    candidate_scores: np.ndarray,
    label: np.ndarray,
) -> np.ndarray:
    return np.sum(np.where(candidate_labels == label, candidate_scores, 0.0), axis=0)


def _validate_threshold(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
