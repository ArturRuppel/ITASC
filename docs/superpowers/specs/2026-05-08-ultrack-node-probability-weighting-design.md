# Ultrack Node Probability Weighting Design

## Goal

Make Ultrack database node-probability scoring explicit and add shape circularity as an enabled-by-default scoring component. The generated `data.db` should continue to rank candidates by segmentation quality, but circular nuclei should receive an additional preference because they are usually better nucleus candidates.

## Current Behavior

During Ultrack database generation, `build_ultrack_database()` calls `write_seed_prior_node_probs()`. That function stores each non-validated node's `NodeDB.node_prob` as:

```python
drop_frac ** cfg.quality_exponent + cfg.seed_weight * best_affinity
```

`drop_frac` is the existing signal-based segmentation quality proxy. `quality_exponent` is already exposed in the database-generation UI, but the weight of that quality term is implicit and fixed at `1.0`.

## Proposed Behavior

Node probability becomes an explicit weighted sum:

```python
node_prob = (
    cfg.quality_weight * (drop_frac ** cfg.quality_exponent)
    + cfg.circularity_weight * circularity
    + cfg.seed_weight * best_affinity
)
```

Defaults:

```python
quality_weight = 1.0
circularity_weight = 0.25
quality_exponent = 8.0
```

`quality_weight=1.0` keeps the existing signal-quality contribution dominant. `circularity_weight=0.25` enables circularity by default without making shape dominate the score. Users can set circularity weight to `0.0` to disable the shape preference.

## Circularity Metric

Circularity is computed from the node mask as:

```python
4 * pi * area / perimeter**2
```

The implementation should return `0.0` for empty masks or zero-perimeter masks and clip the final value to `[0.0, 1.0]`. The metric should operate on the node's cropped boolean mask, so it does not need the full-frame image.

## UI

The Ultrack Database Generation section should expose three related controls:

- `Quality Weight:` default `1.0`
- `Quality Exp:` default `8.0`
- `Circularity Weight:` default `0.25`

These controls should be persisted in `get_state()` and restored in `set_state()`. The terminal DB-generation script path must pass the same config values as the in-process path.

## Testing

Coverage should include:

- Unit tests for circularity scoring, including circular masks scoring higher than elongated masks.
- Unit tests for the weighted node-probability formula.
- A regression test that `circularity_weight=0.0` disables the circularity contribution.
- Widget tests that the new controls are exposed, persisted, restored, and passed through to `TrackingConfig`.
- DB build orchestration tests should continue to prove scoring happens between segmentation/injection and linking.

## Out Of Scope

This change does not alter Ultrack linking, solver weights, DB browser visualization, validated-node injection, or the contour/foreground generation stage.
