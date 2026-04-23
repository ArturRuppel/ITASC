# Nucleus Workflow UI Specification (v2)

## 1. Hypothesis Generation Section
This section manages the creation of segmentation candidates stored in `2_nucleus/hypotheses.h5`.

### Shared Controls (Above Tabs)
- **Seed Source**: `Peak local max` (default), `Active Layer`, `Disk (Corrected)`.
- **Seed Distance**: Spinbox (1-500px), active if `Peak local max` is selected.
- **Overwrite existing**: Checkbox. If unchecked, skip parameter sets already present in the HDF5 database.

### Tab 1: Single Hypothesis ("Tuning")
- **Controls**: Single SpinBoxes for **Threshold (%)**, **Compactness**, and **Smooth Sigma**.
- **Action: Preview**: Run watershed on current frame/Z. Update `Preview: Nucleus` labels layer.
- **Action: Save to DB**: Calculate for the full volume and add as a new `p###` entry in `hypotheses.h5`.
- **Action: Use as Tracked**: Copies the current preview labels into the `2_nucleus/tracked_labels.h5` file for the current frame.

### Tab 2: Parameter Sweep ("Batch")
- **Controls**: Min/Max/Step SpinBoxes for **Threshold**, **Compactness**, and **Smooth Sigma**.
- **Action: Run Batch Sweep**: Generate all permutations locally within napari.
- **Action: Run in Terminal**: Generates a command-line string to run the sweep in an external process (useful for large datasets).

---

## 2. Database Browser (Seeding)
Used to inspect the pool and pick the "Ground Truth" for the starting frame.

- **Hypothesis Slider**: Browse `p000` to `pNNN` from the HDF5.
- **Metadata Display**: Label showing the parameters used for the selected `p` index.
- **Action: Set as Tracking Seed**: Copy the selected labels at the current frame into the tracking result.

---

## 3. Automated Search (Greedy Propagator)
- **IoU Threshold**: Spinbox (0.0 to 1.0). Minimum overlap to consider a link.
- **Max Distance (µm)**: Spinbox. Prefilters candidates by centroid distance before calculating IoU.
- **Action: Propagate Next**: Link current frame to the best candidates in frame `t+1`.
- **Action: Propagate All**: Run the search forward until the end of the movie.
- **Action: Stop**: Abort running propagation.

---

## 4. Manual Correction Integration
- **Action: Correct Current Frame**: 
    - Jumps to the **Global Correction Section**.
    - Targets the `tracked_labels` layer.
    - Enables the interactive tool.

---

## 5. Global Correction Section
*Located below the Cell Workflow section.*

- **Activate/Deactivate Toggle**: High-level switch for mouse callbacks.
- **Shortcuts Reference**: List of Merge/Split/Erase commands.
- **Cell Inspector**: 
    - **Cell ID Spinbox**: Jump to a specific cell.
    - **Lifetime Info**: Shows frames where the ID exists.
    - **Go Button**: Centers the viewer on the cell.
