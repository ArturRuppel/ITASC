# Ultrack Anchor IoU Linker Experiment Findings

Date: 2026-04-30

## Context

We reran the 50-frame middle-anchor Ultrack experiment after fixing
`resolve_with_validation()` so configured linking modes are honored. The
specific question was whether the custom IoU linker improves behavior right
before and right after an anchored GT frame, where we want nearly no splits, no
merges, and unchanged foreground coverage.

Input data:

- Nucleus directory:
  `/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/2_nucleus`
- Hypotheses: `hypotheses.h5`
- GT/reference labels: `tracked_labels.tif`
- Frames compared: 50
- Anchor: middle frame, `t=25`

## Runs

Compared runs:

- `middle_anchor_cell_dedup`
  - Anchored solve with default linker.
- `middle_anchor_adjacent_suppression`
  - Anchored solve with default linker plus adjacent fragment suppression.
- `middle_anchor_iou_linker_after_reseed_fix`
  - Anchored solve with `--linking-mode iou`.
- `middle_anchor_iou_linker_adjacent_suppression`
  - Anchored solve with `--linking-mode iou --suppress-anchor-adjacent-fragments`.

The IoU linker run created 86,623 IoU-weighted edges. Anchor matching was the
same across the anchored runs: 112 / 123 GT labels matched at `t=25`, with 11
unmatched labels. This means the anchor frame itself is capped by unmatched GT
objects before linker behavior enters the problem.

## Aggregate Results

| Run | Tracks | Avg length | Median length | Global binary IoU | Mean frame IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchored default | 808 | 9.18 | 6 | 0.9703 | 0.9701 |
| anchored default + suppression | 797 | 9.28 | 6 | 0.9702 | 0.9700 |
| anchored IoU | 1107 | 6.59 | 3 | 0.9655 | 0.9653 |
| anchored IoU + suppression | 1089 | 6.67 | 3 | 0.9656 | 0.9654 |

The IoU linker worsened global tracking continuity. It increased the total
track count by about 35-37% relative to the anchored default runs and cut the
median track length from 6 frames to 3 frames.

## Anchor-Adjacent Frames

Metrics below compare predicted labels to GT at the frames around the anchor.
Coverage is foreground pixel count ratio, `pred_fg / gt_fg`.

| Run | Frame | GT objects | Pred objects | Coverage | FG IoU | Splits | Merges | Missing | Extra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchored default | 24 | 123 | 138 | 0.9884 | 0.9749 | 17 | 2 | 2 | 1 |
| anchored default | 25 | 123 | 112 | 0.9355 | 0.9338 | 0 | 0 | 11 | 0 |
| anchored default | 26 | 124 | 139 | 0.9890 | 0.9781 | 18 | 2 | 2 | 0 |
| anchored default + suppression | 24 | 123 | 132 | 0.9781 | 0.9646 | 12 | 2 | 2 | 1 |
| anchored default + suppression | 25 | 123 | 112 | 0.9355 | 0.9338 | 0 | 0 | 11 | 0 |
| anchored default + suppression | 26 | 124 | 131 | 0.9888 | 0.9779 | 12 | 2 | 2 | 0 |
| anchored IoU | 24 | 123 | 135 | 0.9755 | 0.9673 | 15 | 1 | 4 | 0 |
| anchored IoU | 25 | 123 | 112 | 0.9355 | 0.9338 | 0 | 0 | 11 | 0 |
| anchored IoU | 26 | 124 | 129 | 0.9649 | 0.9553 | 9 | 0 | 5 | 0 |
| anchored IoU + suppression | 24 | 123 | 126 | 0.9734 | 0.9652 | 8 | 2 | 3 | 0 |
| anchored IoU + suppression | 25 | 123 | 112 | 0.9355 | 0.9338 | 0 | 0 | 11 | 0 |
| anchored IoU + suppression | 26 | 124 | 126 | 0.9661 | 0.9565 | 7 | 0 | 5 | 0 |

## Interpretation

The IoU linker improves the local split/merge pattern near the anchor, but not
enough to satisfy the target behavior.

- At `t=26`, IoU reduces splits from 18 to 9 and merges from 2 to 0.
- With adjacent suppression, IoU reduces `t=26` splits further to 7 and keeps
  merges at 0.
- However, the coverage drop is substantial: `t=26` coverage falls from about
  0.989 with anchored default to about 0.965 with IoU, or 0.966 with IoU plus
  suppression.
- The global result is worse: many more tracks and shorter median track length.

So pure IoU linking is moving in the right direction for local split/merge
counts, but it is too conservative or too brittle for foreground coverage and
track continuity.

## Conclusion

Pure IoU linking is not an acceptable replacement for the default linker in
this experiment. It reduces local splits and merges, but the price is missing
foreground and much heavier track fragmentation.

The next experiment should use a blended linker score rather than pure IoU. A
reasonable sweep is:

- `--linking-mode iou --iou-weight 0.6`
- `--linking-mode iou --iou-weight 0.75`
- optionally `--suppress-anchor-adjacent-fragments` for each

The success criterion should prioritize all three anchor-adjacent goals
together:

- splits and merges near zero at `t=24` and `t=26`
- coverage close to the anchored default, roughly 0.988-0.989
- no large increase in total tracks or drop in median track length

## Follow-up: Signal-based Node Quality

After the IoU-linker results, we tested whether the original nucleus
fluorescence signal can provide useful node evidence. The working intuition was
that a good nucleus candidate should have high signal inside the mask and a
sharp drop just outside the mask.

The most useful simple metric was:

```text
drop_frac = fraction of 1-pixel outer-ring pixels below the node's inside median
```

This signal was useful in diagnostics: for many anchor-adjacent failure cases,
the best whole-object candidate had better `drop_frac` than the individual
selected overlapping fragments. However, wiring plain `drop_frac` into
`NodeDB.node_prob` made fragmentation much worse, because the Ultrack objective
adds positive node rewards independently. A fragmented explanation can beat the
whole object by summing several decent fragment scores.

The important correction was to reshape the reward:

```text
node_prob = drop_frac^8
```

Before the full solve, we checked known overlap-competition failures using:

```text
drop_frac(whole)^p > sum(drop_frac(selected overlapping competitors)^p)
```

`p=1` almost never won against the selected fragment set. `p=8` flipped most
failures, and `p=16` improved slightly but looked more brittle.

General-solver comparison:

| Run | Tracks | Avg length | Median length | Global binary IoU | Mean frame IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline general | 816 | 9.20 | 6 | 0.9713 | 0.9711 |
| `drop_frac` node prob | 1246 | 6.39 | 3 | 0.9740 | 0.9737 |
| `drop_frac^8` node prob | 494 | 11.92 | 7 | 0.9759 | 0.9756 |

Anchor-adjacent comparison for the general solver:

| Run | Frame | Pred objects | Coverage | FG IoU | Splits | Merges | Missing | Extra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline general | 24 | 150 | 0.9828 | 0.9746 | 30 | 3 | 2 | 1 |
| `drop_frac` node prob | 24 | 159 | 0.9882 | 0.9799 | 37 | 4 | 2 | 1 |
| `drop_frac^8` node prob | 24 | 118 | 0.9887 | 0.9804 | 5 | 7 | 2 | 1 |
| baseline general | 25 | 148 | 0.9845 | 0.9799 | 29 | 2 | 3 | 0 |
| `drop_frac` node prob | 25 | 156 | 0.9870 | 0.9819 | 38 | 4 | 3 | 0 |
| `drop_frac^8` node prob | 25 | 121 | 0.9881 | 0.9830 | 7 | 5 | 3 | 0 |
| baseline general | 26 | 150 | 0.9830 | 0.9776 | 29 | 5 | 2 | 0 |
| `drop_frac` node prob | 26 | 161 | 0.9869 | 0.9799 | 39 | 6 | 2 | 0 |
| `drop_frac^8` node prob | 26 | 122 | 0.9872 | 0.9802 | 9 | 9 | 2 | 0 |

Conclusion: the signal prior is real, but it must be shaped so that whole
objects can beat sets of fragments. `drop_frac^8` sharply reduces split
fragmentation and improves foreground IoU, but increases merges. This is not
yet production behavior; keep it as an experiment result until the objective is
settled.

## Next Experiment: Seed-local Node Prior

The next hypothesis is that validated or seeded tracks can provide local node
evidence without imposing a risky global morphology prior. For a candidate
node, reward similarity to nearby seed nodes in space, time, and size. The
reward should decay with spatiotemporal distance from the seed and should use
the best seed affinity, not a sum over seeds.

Planned experimental score:

```text
base_quality = drop_frac^8

node_prob = base_quality * (1 + seed_weight * best_seed_affinity)

best_seed_affinity =
    max over seeds [
        size_similarity(node, seed)
      * spatial_decay(node, seed)
      * temporal_decay(node, seed)
    ]
```

Initial definitions:

```text
size_similarity = exp(-abs(log(area_node / area_seed)) / sigma_area)
spatial_decay   = exp(-(centroid_distance / sigma_space)^2)
temporal_decay  = exp(-abs(dt) / tau_time)
```

For the first contained experiment, use the GT labels at `t=25` as seed nodes
but run the general solver with no anchored constraints. Compare the result
against `baseline_general` and `drop_frac^8` in a local window around the seed.

### Seed-local Prior Result

Experiment directory:

```text
ultrack_anchor_experiment/standard_cell_dedup_dropfrac_p8_seed_t25_nodeprob
```

Parameters:

```text
seed_frame = 25
quality_power = 8
seed_weight = 1.0
max_dt = 5
sigma_space = 25 px
sigma_area = 0.5
tau_time = 2
```

The score was:

```text
node_prob = drop_frac^8 * (1 + seed_weight * best_seed_affinity)
```

The seed affinity reached 1.0 for exact seed-local matches, but was sparse
overall:

```text
affinity p50 = 0.000000
affinity p90 = 0.082127
affinity p99 = 0.552313
affinity max = 1.000000
score max = 2.000000
```

Global comparison:

| Run | Tracks | Avg length | Median length | Global binary IoU | Mean frame IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline general | 816 | 9.20 | 6 | 0.9713 | 0.9711 |
| `drop_frac^8` node prob | 494 | 11.92 | 7 | 0.9759 | 0.9756 |
| `drop_frac^8` + seed prior | 488 | 12.03 | 7 | 0.9756 | 0.9753 |

Local window around the seed:

| Run | Frames | Mean pred objects | Mean coverage | Mean FG IoU | Splits | Merges | Missing | Extra |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline general | 22..28 | 149.0 | 0.9796 | 0.9742 | 208 | 28 | 23 | 3 |
| `drop_frac^8` node prob | 22..28 | 121.3 | 0.9852 | 0.9791 | 54 | 47 | 21 | 3 |
| `drop_frac^8` + seed prior | 22..28 | 119.0 | 0.9843 | 0.9784 | 34 | 43 | 23 | 3 |
| baseline general | 24..26 | 149.3 | 0.9834 | 0.9774 | 88 | 10 | 7 | 1 |
| `drop_frac^8` node prob | 24..26 | 120.3 | 0.9880 | 0.9812 | 21 | 21 | 7 | 1 |
| `drop_frac^8` + seed prior | 24..26 | 119.3 | 0.9853 | 0.9788 | 12 | 16 | 9 | 1 |

Interpretation: the seed-local prior further reduces split fragmentation near
the seed frame, but the first parameterization is not a net improvement over
plain `drop_frac^8`. It trades away some coverage and foreground IoU and
slightly increases missing objects. This suggests the seed prior is affecting
the solver, but the current formulation is too willing to concentrate reward
near seed-compatible candidates at the expense of foreground recovery.

### Additive Seed-local Prior Result

The next test used the seed prior as an independent additive reward:

```text
node_prob = drop_frac^8 + seed_weight * best_seed_affinity
```

Experiment directory:

```text
ultrack_anchor_experiment/standard_cell_dedup_dropfrac_p8_addseed_t25_w05_nodeprob
```

Parameters were the same as the multiplicative test except:

```text
seed_weight = 0.5
```

The additive bonus was moderate relative to the `drop_frac^8` base:

```text
bonus p50 = 0.000000
bonus p90 = 0.041063
bonus p99 = 0.276157
bonus max = 0.500000
score max = 1.500000
```

Global comparison:

| Run | Tracks | Avg length | Median length | Global binary IoU | Mean frame IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline general | 816 | 9.20 | 6 | 0.9713 | 0.9711 |
| `drop_frac^8` node prob | 494 | 11.92 | 7 | 0.9759 | 0.9756 |
| `drop_frac^8` + multiplicative seed prior | 488 | 12.03 | 7 | 0.9756 | 0.9753 |
| `drop_frac^8` + additive seed prior | 483 | 12.14 | 7 | 0.9758 | 0.9756 |

Local window around the seed:

| Run | Frames | Mean pred objects | Mean coverage | Mean FG IoU | Splits | Merges | Missing | Extra |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline general | 22..28 | 149.0 | 0.9796 | 0.9742 | 208 | 28 | 23 | 3 |
| `drop_frac^8` node prob | 22..28 | 121.3 | 0.9852 | 0.9791 | 54 | 47 | 21 | 3 |
| `drop_frac^8` + multiplicative seed prior | 22..28 | 119.0 | 0.9843 | 0.9784 | 34 | 43 | 23 | 3 |
| `drop_frac^8` + additive seed prior | 22..28 | 120.0 | 0.9858 | 0.9801 | 37 | 40 | 22 | 2 |
| baseline general | 24..26 | 149.3 | 0.9834 | 0.9774 | 88 | 10 | 7 | 1 |
| `drop_frac^8` node prob | 24..26 | 120.3 | 0.9880 | 0.9812 | 21 | 21 | 7 | 1 |
| `drop_frac^8` + multiplicative seed prior | 24..26 | 119.3 | 0.9853 | 0.9788 | 12 | 16 | 9 | 1 |
| `drop_frac^8` + additive seed prior | 24..26 | 121.3 | 0.9883 | 0.9822 | 16 | 15 | 8 | 0 |

Interpretation: the additive seed prior is the better formulation. Compared
with plain `drop_frac^8`, it improves the seed-local window while preserving
coverage: fewer splits, fewer merges, slightly higher foreground IoU, and no
extra objects in frames `24..26`. The global result is roughly neutral, which
is expected because the seed prior only affects a local time window.

### Selected-node Penalty Result

We also tested a more explicit linear surrogate for the `A > B + C` problem:
subtract a constant penalty for every selected node. This keeps the ILP linear
and directly makes fragmented explanations pay an additional cardinality cost.

Important implementation detail: Ultrack applies the configured tracking
`link_function` to `node_prob` as well as link weights. With the current config,
plain `drop_frac` in `NodeDB.node_prob` is seen by the objective as
approximately `drop_frac^4`. The one-off solver patch therefore tested:

```text
objective node term = drop_frac^4 - 0.2 * selected_node
```

Experiment directory:

```text
ultrack_anchor_experiment/standard_cell_dedup_dropfrac_penalty_l02_nodeprob
```

Global comparison:

| Run | Tracks | Avg length | Median length | Global binary IoU | Mean frame IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline general | 816 | 9.20 | 6 | 0.9713 | 0.9711 |
| `drop_frac` node prob | 1246 | 6.39 | 3 | 0.9740 | 0.9737 |
| `drop_frac^8` node prob | 494 | 11.92 | 7 | 0.9759 | 0.9756 |
| `drop_frac^8` + additive seed prior | 483 | 12.14 | 7 | 0.9758 | 0.9756 |
| `drop_frac^4 - 0.2` selected-node penalty | 750 | 9.19 | 5 | 0.9717 | 0.9715 |

Local window around the seed:

| Run | Frames | Mean pred objects | Mean coverage | Mean FG IoU | Splits | Merges | Missing | Extra |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline general | 22..28 | 149.0 | 0.9796 | 0.9742 | 208 | 28 | 23 | 3 |
| `drop_frac` node prob | 22..28 | 158.6 | 0.9842 | 0.9779 | 264 | 37 | 18 | 3 |
| `drop_frac^8` node prob | 22..28 | 121.3 | 0.9852 | 0.9791 | 54 | 47 | 21 | 3 |
| `drop_frac^8` + additive seed prior | 22..28 | 120.0 | 0.9858 | 0.9801 | 37 | 40 | 22 | 2 |
| `drop_frac^4 - 0.2` selected-node penalty | 22..28 | 136.7 | 0.9771 | 0.9715 | 141 | 34 | 26 | 2 |
| baseline general | 24..26 | 149.3 | 0.9834 | 0.9774 | 88 | 10 | 7 | 1 |
| `drop_frac` node prob | 24..26 | 158.7 | 0.9874 | 0.9806 | 114 | 14 | 7 | 1 |
| `drop_frac^8` node prob | 24..26 | 120.3 | 0.9880 | 0.9812 | 21 | 21 | 7 | 1 |
| `drop_frac^8` + additive seed prior | 24..26 | 121.3 | 0.9883 | 0.9822 | 16 | 15 | 8 | 0 |
| `drop_frac^4 - 0.2` selected-node penalty | 24..26 | 138.7 | 0.9799 | 0.9738 | 63 | 13 | 10 | 0 |

Interpretation: the selected-node penalty does reduce the worst fragmentation
of plain `drop_frac`, but it behaves like a blunt global suppression term. It
does not recover the split reduction or foreground quality of the high-power
quality prior, and it increases missing objects. A tuned penalty might improve,
but this one-off result does not look better than the current
`drop_frac^8` plus additive seed prior direction.
