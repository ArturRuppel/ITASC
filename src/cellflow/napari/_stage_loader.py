"""Load a pipeline stage's on-disk output(s) into a napari viewer.

The catalog status rail's dots are clickable: clicking a stage's dot loads that
stage's canonical output(s) for the position into the viewer. This is *per-stage*
loading (four stages), coarser than the Pipeline Files panel's per-file loading —
it reuses the same ``tifffile.imread`` → ``add_labels`` / ``add_image`` logic. The
image-stage targets prefer a committed ``*_labels.tif`` over its working file so a
finalized position loads its stable output.

Qt-free: the viewer is used purely by its duck-typed ``layers`` / ``add_labels`` /
``add_image`` surface, so this is testable with a stub viewer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._stage_status import (
    STAGE_CELL,
    STAGE_CELLPOSE,
    STAGE_CONTACTS,
    STAGE_NUCLEUS,
)


@dataclass(frozen=True)
class LoadTarget:
    """One file to load, and how to present it."""

    path: Path
    as_labels: bool
    colormap: str = "gray"


def stage_load_targets(pos_dir: Path | str, stage: str) -> list[LoadTarget]:
    """Canonical file(s) a stage loads for *pos_dir* (existence not checked here).

    Cellpose loads its divergence maps (foreground → grey, contours → magma) for
    both channels. Nucleus / Cell load the committed ``*_labels.tif`` when present,
    else the working stage file, as labels. Contacts has no raw-image output (the
    ``.h5`` is opened by the Visualize Contacts tool), so it loads nothing.
    """
    paths = NucleusArtifactPaths(Path(pos_dir))
    if stage == STAGE_CELLPOSE:
        return [
            LoadTarget(paths.nucleus_foreground, as_labels=False, colormap="gray"),
            LoadTarget(paths.nucleus_contours, as_labels=False, colormap="magma"),
            LoadTarget(paths.cell_foreground, as_labels=False, colormap="gray"),
            LoadTarget(paths.cell_contours, as_labels=False, colormap="magma"),
        ]
    if stage == STAGE_NUCLEUS:
        committed = paths.nucleus_labels
        chosen = committed if committed.is_file() else paths.tracked
        return [LoadTarget(chosen, as_labels=True)]
    if stage == STAGE_CELL:
        committed = paths.cell_labels
        chosen = committed if committed.is_file() else paths.cell_tracked
        return [LoadTarget(chosen, as_labels=True)]
    if stage == STAGE_CONTACTS:
        return []
    return []


def _layer_name(pos_dir: Path, path: Path) -> str:
    """A per-position layer name so two positions' files don't collide."""
    return f"{Path(pos_dir).name}:{path.stem}"


def load_stage(viewer, pos_dir: Path | str | None, stage: str) -> list[str]:
    """Load *stage*'s existing outputs for *pos_dir* into *viewer*.

    Returns the layer names loaded (empty when the position has no canonical root,
    no viewer, or nothing on disk yet). An already-present layer of the same name
    has its data replaced rather than duplicated.
    """
    if viewer is None or pos_dir is None:
        return []
    import tifffile

    loaded: list[str] = []
    for target in stage_load_targets(pos_dir, stage):
        if not target.path.is_file():
            continue
        data = tifffile.imread(str(target.path))
        name = _layer_name(Path(pos_dir), target.path)
        if name in viewer.layers:
            try:
                viewer.layers[name].data = data
                loaded.append(name)
                continue
            except Exception:  # noqa: BLE001 - fall back to a fresh layer
                viewer.layers.remove(viewer.layers[name])
        if target.as_labels:
            viewer.add_labels(data, name=name)
        else:
            viewer.add_image(data, name=name, colormap=target.colormap)
        loaded.append(name)
    return loaded
