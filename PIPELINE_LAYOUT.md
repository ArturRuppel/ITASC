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
    ├── 0_input/                               ← step 0: NDTiff export
    │   ├── nucleus/
    │   │   ├── nucleus_3d_t000.tif            (Z, H, W)    uint16
    │   │   ├── nucleus_3d_t001.tif
    │   │   ├── ...
    │   │   └── nucleus_zavg.tif               (T, H, W)    uint16
    │   └── cell/
    │       └── cell_zavg.tif                  (T, H, W)    uint16
    │
    ├── 1_cellpose/
    │   ├── nucleus/                           ← step 1a: Cellpose 3D nuclei
    │   │   ├── nucleus_3d_t000_dp.tif         (3, Z, H, W) float32
    │   │   ├── nucleus_3d_t000_prob.tif       (Z, H, W)    float32
    │   │   └── ...
    │   └── cell/                              ← step 1b: Cellpose 2D cells
    │       ├── cell_dp.tif                    (T, 2, H, W) float32
    │       └── cell_prob.tif                  (T, H, W)    float32
    │
    ├── 2_ultrack/                             ← steps 2: contours + tracking
    │   ├── foreground.tif                     (T, H, W)    float32
    │   ├── contours.tif                       (T, H, W)    float32
    │   ├── data.db                            Ultrack SQLite database
    │   ├── tracks.csv
    │   ├── tracked_labels.tif                 (T, Z, H, W) uint32   ← 3D Ultrack output
    │   └── nuclear_labels_2d.tif              (T, H, W)    uint32   ← max-projection of tracked_labels
    │
    ├── 3_correction/                          ← step 3: manual correction + LapTrack loop
    │   └── nuclear_labels_corrected.tif       (T, H, W)    int32
    │
    ├── 4_cell_segmentation/                   ← step 4: nucleus-anchored cell seg
    │   ├── cell_labels_raw.tif                (T, H, W)    int32
    │   └── cell_labels.tif                    (T, H, W)    int32
    │
    └── 5_analysis/                            ← step 6: edge analysis
        ├── graph.h5
        └── topology.npz
```

## Stage → Directory mapping

| Stage key            | Output directory          | Widget tab     |
|----------------------|---------------------------|----------------|
| `raw_import`         | `0_input/`                | Data Prep      |
| `cellpose_nucleus`   | `1_cellpose/nucleus/`     | Cellpose       |
| `cellpose_cell`      | `1_cellpose/cell/`        | Cellpose       |
| `contours`           | `2_ultrack/`              | Ultrack        |
| `tracking`           | `2_ultrack/`              | Ultrack        |
| `correction`         | `3_correction/`           | Correction     |
| `cell_segmentation`  | `4_cell_segmentation/`    | Cell Seg       |
| `graph_extraction`   | `5_analysis/`             | Edge Analysis  |
| `topology_analysis`  | `5_analysis/`             | Edge Analysis  |

## Notes

- `nuclear_labels_2d.tif` is produced by the Ultrack stage as a max-projection
  of `tracked_labels.tif` along Z. It is the input to the correction step.
- `nuclear_labels_corrected.tif` is the final output of the correction loop
  (manual edits + optional LapTrack retracking). It is the input to cell seg.
- The LapTrack "Tracking" tab has no pipeline stage key — it is part of the
  correction loop and does not write its own output directory.
- ForSys results are embedded in the `5_analysis/` HDF5 outputs; no separate
  stage directory is planned.
