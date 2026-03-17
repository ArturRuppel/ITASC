"""TrackMate XML parser for nuclear tracking data."""
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackMateData:
    """Parsed TrackMate data."""

    # Per-frame positions: frame -> list of (spot_id, y, x)
    spots_by_frame: Dict[int, List[Tuple[int, float, float]]]

    # Track assignments: spot_id -> track_id
    spot_to_track: Dict[int, int]

    # Image dimensions (H, W) in pixels
    image_shape: Optional[Tuple[int, int]] = None

    # Calibration
    pixel_size_x: Optional[float] = None
    pixel_size_y: Optional[float] = None
    time_interval: Optional[float] = None
    spatial_units: str = ""
    time_units: str = ""

    # Number of frames in the original image
    n_frames: Optional[int] = None

    # Detector/tracker settings (informational)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def n_spots(self) -> int:
        return sum(len(spots) for spots in self.spots_by_frame.values())

    @property
    def n_tracks(self) -> int:
        return len(set(self.spot_to_track.values()))

    @property
    def frame_indices(self) -> List[int]:
        return sorted(self.spots_by_frame.keys())

    def to_positions_array(self) -> np.ndarray:
        """Convert to (N, 3) array with columns (frame, y, x) in pixel coordinates."""
        rows = []
        for frame in sorted(self.spots_by_frame.keys()):
            for spot_id, y, x in self.spots_by_frame[frame]:
                rows.append([frame, y, x])
        return np.array(rows)

    def to_positions_array_with_track_ids(self) -> np.ndarray:
        """Convert to (N, 4) array with columns (frame, y, x, track_id).

        Spots without a track assignment get track_id = -1.
        """
        rows = []
        for frame in sorted(self.spots_by_frame.keys()):
            for spot_id, y, x in self.spots_by_frame[frame]:
                track_id = self.spot_to_track.get(spot_id, -1)
                rows.append([frame, y, x, track_id])
        return np.array(rows)


def parse_trackmate_xml(path: Union[str, Path]) -> TrackMateData:
    """Parse a TrackMate XML file.

    Args:
        path: Path to the TrackMate XML file.

    Returns:
        TrackMateData with spots, tracks, and metadata.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    model = root.find("Model")
    if model is None:
        raise ValueError("No <Model> element found in TrackMate XML")

    spatial_units = model.get("spatialunits", "")
    time_units = model.get("timeunits", "")

    spots_by_frame = _parse_spots(model)
    spot_to_track = _parse_tracks(model)
    image_shape, pixel_size_x, pixel_size_y, time_interval, n_frames, metadata = (
        _parse_settings(root)
    )

    # Convert spot positions from physical to pixel coordinates if calibrated
    if pixel_size_x is not None and pixel_size_y is not None:
        for frame, spots in spots_by_frame.items():
            spots_by_frame[frame] = [
                (sid, y / pixel_size_y, x / pixel_size_x) for sid, y, x in spots
            ]

    n_spots = sum(len(s) for s in spots_by_frame.values())
    n_tracks = len(set(spot_to_track.values()))
    logger.info(
        f"Parsed TrackMate XML: {n_spots} spots, {n_tracks} tracks, "
        f"{len(spots_by_frame)} frames"
    )

    return TrackMateData(
        spots_by_frame=spots_by_frame,
        spot_to_track=spot_to_track,
        image_shape=image_shape,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        time_interval=time_interval,
        spatial_units=spatial_units,
        time_units=time_units,
        n_frames=n_frames,
        metadata=metadata,
    )


def _parse_spots(model: ET.Element) -> Dict[int, List[Tuple[int, float, float]]]:
    """Parse <AllSpots> section.

    Returns dict: frame -> list of (spot_id, y, x) in physical coordinates.
    """
    spots_by_frame: Dict[int, List[Tuple[int, float, float]]] = {}

    all_spots = model.find("AllSpots")
    if all_spots is None:
        raise ValueError("No <AllSpots> element found in TrackMate XML")

    for sif in all_spots.findall("SpotsInFrame"):
        frame = int(sif.get("frame"))
        frame_spots = []
        for spot in sif.findall("Spot"):
            spot_id = int(spot.get("ID"))
            # TrackMate uses X, Y in physical units
            x = float(spot.get("POSITION_X"))
            y = float(spot.get("POSITION_Y"))
            frame_spots.append((spot_id, y, x))
        spots_by_frame[frame] = frame_spots

    return spots_by_frame


def _parse_tracks(model: ET.Element) -> Dict[int, int]:
    """Parse <AllTracks> and <FilteredTracks> sections.

    Returns dict: spot_id -> track_id.
    Only includes tracks listed in FilteredTracks (if present).
    """
    all_tracks = model.find("AllTracks")
    if all_tracks is None:
        return {}

    # Check which tracks are filtered (accepted)
    filtered_tracks = model.find("FilteredTracks")
    if filtered_tracks is not None:
        accepted_ids = {
            int(t.get("TRACK_ID")) for t in filtered_tracks.findall("TrackID")
        }
    else:
        accepted_ids = None  # Accept all

    # Build spot_id -> track_id mapping from edges
    spot_to_track: Dict[int, int] = {}
    for track in all_tracks.findall("Track"):
        track_id = int(track.get("TRACK_ID"))
        if accepted_ids is not None and track_id not in accepted_ids:
            continue

        # Collect all spot IDs from edges in this track
        track_spots = set()
        for edge in track.findall("Edge"):
            track_spots.add(int(edge.get("SPOT_SOURCE_ID")))
            track_spots.add(int(edge.get("SPOT_TARGET_ID")))

        for spot_id in track_spots:
            spot_to_track[spot_id] = track_id

    return spot_to_track


def _parse_settings(
    root: ET.Element,
) -> Tuple[
    Optional[Tuple[int, int]],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[int],
    Dict[str, str],
]:
    """Parse <Settings> section for image data and detector/tracker info.

    Returns (image_shape, pixel_size_x, pixel_size_y, time_interval, n_frames, metadata).
    """
    image_shape = None
    pixel_size_x = None
    pixel_size_y = None
    time_interval = None
    n_frames = None
    metadata: Dict[str, str] = {}

    settings = root.find("Settings")
    if settings is None:
        return image_shape, pixel_size_x, pixel_size_y, time_interval, n_frames, metadata

    image_data = settings.find("ImageData")
    if image_data is not None:
        width = image_data.get("width")
        height = image_data.get("height")
        if width is not None and height is not None:
            image_shape = (int(height), int(width))  # (H, W)

        pw = image_data.get("pixelwidth")
        ph = image_data.get("pixelheight")
        if pw is not None:
            pixel_size_x = float(pw)
        if ph is not None:
            pixel_size_y = float(ph)

        ti = image_data.get("timeinterval")
        if ti is not None:
            time_interval = float(ti)

        nf = image_data.get("nframes")
        if nf is not None:
            n_frames = int(nf)

        metadata["filename"] = image_data.get("filename", "")
        metadata["folder"] = image_data.get("folder", "")

    detector = settings.find("DetectorSettings")
    if detector is not None:
        metadata["detector"] = detector.get("DETECTOR_NAME", "")

    tracker = settings.find("TrackerSettings")
    if tracker is not None:
        metadata["tracker"] = tracker.get("TRACKER_NAME", "")

    return image_shape, pixel_size_x, pixel_size_y, time_interval, n_frames, metadata
