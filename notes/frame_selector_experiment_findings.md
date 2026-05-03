# Frame Selector Experiment Findings

Date: 2026-05-01

## Input

- Cell workflow hypotheses:
  `/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/hypotheses.h5`
- Size: 358.7 MB
- Timepoints: 50
- 2D candidate states: 22,000
- Candidate shape example: `(8, 512, 512)` per parameter per frame

## Runs

- `frame_selector_experiment_20260501_beam100_optimized`
  - `--top-k 20 --beam-width 100 --export-top 3`
  - Wall time: 58.86 s
  - Max RSS: 592,920 KB
  - Wrote `ranked_paths.csv`, `transition_scores.csv`, `summary.json`, and top three TIFF stacks.
- `frame_selector_experiment_20260501_beam200_optimized`
  - `--top-k 20 --beam-width 200 --export-top 0`
  - Wall time: 68.03 s
  - Max RSS: 544,864 KB
  - Wrote CSV/JSON outputs only.

## Result

Beam 100 and beam 200 produced the same top five paths and identical best path:

- Best score: `85.75055886`
- Best parameter path: `p=10` for all 50 frames
- Selected parameter attributes:
  `{"method": "seeded_watershed", "basin": "flow_mag", "foreground_threshold": 0.1, "compactness": 0.1}`
- Best z path:
  `4 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 1 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 3 3 3 3 3 3 3 1 0 0 0 1 0 0 0 0`

Top-three transition summaries from the beam-200 run:

| Rank | Score | Mean transition | Max transition | Missing total | Extra total | Parameter switches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 85.750559 | 1.750011 | 10.186233 | 12 | 9 | 0 |
| 2 | 85.756785 | 1.750138 | 10.186233 | 12 | 9 | 0 |
| 3 | 85.757747 | 1.750158 | 10.186233 | 12 | 9 | 0 |

## Interpretation

The internal-consistency selector strongly prefers the same cell-boundary
parameter (`p=10`) for every frame. The remaining ambiguity is almost entirely
which z-slice to use per frame. Because beam 100 and beam 200 match exactly at
the top, beam 100 is sufficient for this dataset unless the scoring model
changes.

## Consensus Movie Follow-Up

After visual/metric concerns with selecting one hypothesis path, we ran a
consensus-voting experiment that treats each compactness value independently.
For each compactness group, every frame receives 40 votes per pixel: five
foreground thresholds times eight z-slices. The output label is the most common
grounded cell ID where its temporally smoothed support is at least `0.5`;
otherwise the pixel is background.

Run:

- Output directory:
  `/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/consensus_movie_experiment_20260501_threshold050`
- Command:
  `python scripts/experiment_consensus_movie.py <3_cell/hypotheses.h5> --output-dir <output-dir>`
- Wall time: 1:44.41
- Max RSS: 601,756 KB
- Outputs: one `*_labels.tif` and one `*_support_u8.tif` per compactness,
  plus `summary.csv` and `summary.json`.

Summary trend:

| Compactness | Mean support | Foreground fraction |
| ---: | ---: | ---: |
| 0.00 | 0.726 | 0.3231 |
| 0.01 | 0.737 | 0.3522 |
| 0.02 | 0.740 | 0.3603 |
| 0.03 | 0.742 | 0.3662 |
| 0.04 | 0.743 | 0.3707 |
| 0.05 | 0.744 | 0.3746 |
| 0.06 | 0.745 | 0.3779 |
| 0.07 | 0.746 | 0.3808 |
| 0.08 | 0.747 | 0.3836 |
| 0.09 | 0.748 | 0.3860 |
| 0.10 | 0.748 | 0.3884 |

This experiment no longer chooses a single z/threshold path. It instead
produces a consensus movie per compactness value and a support map showing
where the hypothesis ensemble agrees or disagrees.

## Dynamic Consensus Threshold Follow-Up

We added a dynamic threshold mode for consensus movies:

- `--threshold-mode fixed` keeps the original single support threshold.
- `--threshold-mode percentile` chooses one threshold per frame from the
  foreground candidate support distribution.
- The per-frame thresholds are written to `thresholds.csv`.

Probe runs on compactness `0.1` showed that percentile `60` with a wide clamp
was fairly conservative: thresholds ranged from `0.708333` to `0.800000`, and
foreground fraction dropped to `0.209002`. Percentile `40` was less aggressive
and kept useful threshold variation.

Full dynamic run:

- Output directory:
  `/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/consensus_movie_experiment_20260501_percentile40_dynamic`
- Command:
  `python scripts/experiment_consensus_movie.py <3_cell/hypotheses.h5> --output-dir <output-dir> --threshold-mode percentile --threshold-percentile 40 --min-vote-threshold 0.35 --max-vote-threshold 0.95`
- Wall time: 1:44.74
- Max RSS: 600,880 KB
- Selected label movie:
  `/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/consensus_labels_percentile40_dynamic.tif`

Dynamic threshold summary:

| Compactness | Mean support | Foreground fraction | Threshold min | Threshold mean | Threshold max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.725 | 0.2787 | 0.483 | 0.607 | 0.650 |
| 0.01 | 0.737 | 0.2926 | 0.483 | 0.624 | 0.667 |
| 0.02 | 0.740 | 0.2971 | 0.483 | 0.626 | 0.667 |
| 0.03 | 0.742 | 0.3002 | 0.483 | 0.628 | 0.667 |
| 0.04 | 0.743 | 0.3031 | 0.483 | 0.628 | 0.667 |
| 0.05 | 0.744 | 0.3055 | 0.483 | 0.629 | 0.667 |
| 0.06 | 0.745 | 0.3072 | 0.475 | 0.629 | 0.667 |
| 0.07 | 0.746 | 0.3092 | 0.467 | 0.629 | 0.667 |
| 0.08 | 0.747 | 0.3107 | 0.467 | 0.629 | 0.667 |
| 0.09 | 0.748 | 0.3120 | 0.467 | 0.630 | 0.667 |
| 0.10 | 0.749 | 0.3135 | 0.467 | 0.629 | 0.667 |
