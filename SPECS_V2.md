# CellFlow v2: Specification & Workflow

## Project Goal
Transform CellFlow into a fast, interactive, hypothesis-driven tracking tool. Replace the global optimization (Ultrack) with a greedy local search through a pre-calculated database of segmentation hypotheses.

## The New Workflow

### 1. Data Preparation (Legacy)
- Import raw stacks, perform Z-alignment, and generate Z-projections.
- Output: `0_input/` artifacts.

### 2. Cellpose Output (External)
- Cellpose flow and probability maps are calculated externally (e.g., on a cluster).
- Output: `1_cellpose/` artifacts.
- Plugin role: Informational panel only.

### 3. Nucleus Workflow
- **A. Hypothesis Generation:** Generate a pool of segmentation candidates by sweeping parameters (thresholds, etc.) and save them to a database (`hypotheses.h5`).
- **B. Seeding:** Select a starting frame and pick the best initial hypothesis.
- **C. Manual Correction:** Use the correction widget to refine the seed frame or any subsequent frame.
- **D. Automated Search:** Use a greedy search engine (IoU-based) to automatically find the "best fit" candidate for the next frame from the hypothesis database.

### 4. Cell Workflow
- **A. Seeded Hypothesis Generation:** Create a cell hypothesis database using **Seeded Watershed**, where the corrected nuclear labels from the previous step serve as the seeds.
- **B. Search & Selection:** Similar to the nucleus workflow, pick/correct a layer and use the automated search to propagate labels through the cell hypothesis pool.

### 5. Analysis
- Calculate topology, mechanics, and export statistics based on the final tracked labels.

## Architectural Mandates
- **Clean Slate:** All legacy Ultrack code and complex relational databases are removed.
- **HDF5-Centric:** Use HDF5 for storing both the hypothesis pools and the final tracked results.
- **UI-Driven:** The interface is the "steering wheel" for the search engine. Propagation should be fast enough for interactive use.

## Package Structure
- `cellflow.core`: Project and path management.
- `cellflow.database`: HDF5 IO for hypotheses and tracked labels.
- `cellflow.segmentation`: Nucleus and Seeded-Cell hypothesis generators.
- `cellflow.tracking`: The greedy search engine (IoU + KDTree).
- `cellflow.correction`: Tools for manual label refinement.
- `cellflow.napari`: The unified workflow-based UI.
