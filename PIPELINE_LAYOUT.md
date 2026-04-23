# CellFlow Pipeline Directory Layout

Each position runs independently. Replace `pos00` with `pos01`, `pos02`, etc.

```
<project_root>/
├── project.json
├── pipeline_schema.json
│
└── pos00/
    ├── pipeline_manifest.json
    ├── pipeline.log
    │
    ├── 0_input/                               ← step 0: raw import
    │   ├── nucleus_4d.tif                     (T, Z, H, W) uint16
    │   ├── cell_4d.tif                        (T, Z, H, W) uint16
    │   ├── nucleus_zavg.tif                   (T, H, W)    uint16
    │   ├── cell_zavg.tif                      (T, H, W)    uint16
    │   └── z_shift.csv
    │
    ├── 1_cellpose/                            ← step 1: Cellpose cluster outputs
    │   ├── nucleus_dp_4d.tif                  (T, C, Z, H, W) float32
    │   ├── nucleus_prob_4d.tif                (T, Z, H, W)    float32
    │   ├── cell_dp_4d.tif                     (T, Z, 2, H, W) float32
    │   ├── cell_prob_4d.tif                   (T, Z, H, W)    float32
    │   ├── cell_dp_zavg.tif                   (T, 2, H, W)    float32
    │   └── cell_prob_zavg.tif                 (T, H, W)       float32
    │
    ├── 2_nucleus_ultrack/                     ← step 2: nucleus Ultrack
    │   ├── data.db                            Ultrack SQLite database
    │   ├── tracks.csv
    │   ├── tracked_labels.tif                 (T, Z, H, W) uint32
    │   ├── nuclear_labels_2d.tif              (T, H, W)    uint32
    │   ├── hypotheses_manifest.json
    │   └── labelmaps/labelmap_*.tif
    │
    ├── 3_correction/                          ← step 3: manual correction
    │   └── nuclear_labels_corrected.tif       (T, H, W)    int32
    │
    ├── 4_cell_ultrack/                        ← step 4: cell Ultrack
    │   ├── data.db
    │   ├── tracks.csv
    │   ├── tracked_labels.tif                 (T, Z, H, W) uint32
    │   ├── cell_labels_2d.tif                 (T, H, W)    uint32
    │   ├── hypotheses_manifest.json
    │   └── labelmaps/labelmap_*.tif
    │
    └── 5_analysis/                            ← step 5: analysis
        ├── graph.h5
        └── topology.npz
```

## Stage → Directory mapping

| Stage key            | Output directory          | Widget tab     |
|----------------------|---------------------------|----------------|
| `raw_import`         | `0_input/`                | Data Prep      |
| `cellpose_cluster`   | `1_cellpose/`             | Cellpose       |
| `nucleus_ultrack`    | `2_nucleus_ultrack/`      | Ultrack        |
| `correction`         | `3_correction/`           | Correction     |
| `cell_ultrack`       | `4_cell_ultrack/`         | Cell Seg       |
| `analysis`           | `5_analysis/`             | Edge Analysis  |

## Notes

- `nucleus_dp_4d.tif` uses `(T, C, Z, H, W)` with `C=3` in full-3D Cellpose mode
  and `C=2` in slice-wise mode.
- `nuclear_labels_2d.tif` is the input to the correction step.
- `nuclear_labels_corrected.tif` is the input to the cell Ultrack step.
