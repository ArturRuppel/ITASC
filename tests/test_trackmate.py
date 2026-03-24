"""Tests for TrackMate XML parser."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from napariTissueFlow.core.trackmate import TrackMateData, parse_trackmate_xml

SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<TrackMate version="7.0.0">
  <Log>test log</Log>
  <Model spatialunits="micron" timeunits="sec">
    <FeatureDeclarations>
      <SpotFeatures>
        <Feature feature="POSITION_X" name="X" dimension="POSITION" isint="false" />
        <Feature feature="POSITION_Y" name="Y" dimension="POSITION" isint="false" />
      </SpotFeatures>
    </FeatureDeclarations>
    <AllSpots nspots="7">
      <SpotsInFrame frame="0">
        <Spot ID="0" POSITION_X="10.0" POSITION_Y="20.0" POSITION_Z="0.0" FRAME="0" VISIBILITY="1" RADIUS="5.0" QUALITY="1.0" />
        <Spot ID="1" POSITION_X="30.0" POSITION_Y="40.0" POSITION_Z="0.0" FRAME="0" VISIBILITY="1" RADIUS="5.0" QUALITY="1.0" />
        <Spot ID="2" POSITION_X="50.0" POSITION_Y="60.0" POSITION_Z="0.0" FRAME="0" VISIBILITY="1" RADIUS="5.0" QUALITY="1.0" />
      </SpotsInFrame>
      <SpotsInFrame frame="1">
        <Spot ID="3" POSITION_X="11.0" POSITION_Y="21.0" POSITION_Z="0.0" FRAME="1" VISIBILITY="1" RADIUS="5.0" QUALITY="1.0" />
        <Spot ID="4" POSITION_X="31.0" POSITION_Y="41.0" POSITION_Z="0.0" FRAME="1" VISIBILITY="1" RADIUS="5.0" QUALITY="1.0" />
        <Spot ID="5" POSITION_X="51.0" POSITION_Y="61.0" POSITION_Z="0.0" FRAME="1" VISIBILITY="1" RADIUS="5.0" QUALITY="1.0" />
      </SpotsInFrame>
      <SpotsInFrame frame="2">
        <Spot ID="6" POSITION_X="12.0" POSITION_Y="22.0" POSITION_Z="0.0" FRAME="2" VISIBILITY="1" RADIUS="5.0" QUALITY="1.0" />
      </SpotsInFrame>
    </AllSpots>
    <AllTracks>
      <Track name="Track_0" TRACK_ID="0" NUMBER_SPOTS="3" NUMBER_SPLITS="0" NUMBER_MERGES="0">
        <Edge SPOT_SOURCE_ID="0" SPOT_TARGET_ID="3" LINK_COST="1.0" />
        <Edge SPOT_SOURCE_ID="3" SPOT_TARGET_ID="6" LINK_COST="1.0" />
      </Track>
      <Track name="Track_1" TRACK_ID="1" NUMBER_SPOTS="2" NUMBER_SPLITS="0" NUMBER_MERGES="0">
        <Edge SPOT_SOURCE_ID="1" SPOT_TARGET_ID="4" LINK_COST="1.0" />
      </Track>
      <Track name="Track_2" TRACK_ID="2" NUMBER_SPOTS="2" NUMBER_SPLITS="0" NUMBER_MERGES="0">
        <Edge SPOT_SOURCE_ID="2" SPOT_TARGET_ID="5" LINK_COST="1.0" />
      </Track>
    </AllTracks>
    <FilteredTracks>
      <TrackID TRACK_ID="0" />
      <TrackID TRACK_ID="1" />
    </FilteredTracks>
  </Model>
  <Settings>
    <ImageData filename="test.tif" folder="/tmp/" width="200" height="100" nslices="1" nframes="3" pixelwidth="0.5" pixelheight="0.5" voxeldepth="1.0" timeinterval="2.0" />
    <DetectorSettings DETECTOR_NAME="LOG_DETECTOR" />
    <TrackerSettings TRACKER_NAME="SPARSE_LAP_TRACKER" />
  </Settings>
</TrackMate>
"""


@pytest.fixture
def sample_xml_path(tmp_path):
    p = tmp_path / "tracks.xml"
    p.write_text(SAMPLE_XML)
    return p


class TestParseSpots:
    def test_spot_count(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.n_spots == 7

    def test_frames_present(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.frame_indices == [0, 1, 2]

    def test_spots_per_frame(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert len(data.spots_by_frame[0]) == 3
        assert len(data.spots_by_frame[1]) == 3
        assert len(data.spots_by_frame[2]) == 1

    def test_positions_in_pixel_coords(self, sample_xml_path):
        """Positions should be converted from physical to pixel coordinates."""
        data = parse_trackmate_xml(sample_xml_path)
        # pixel_size = 0.5, so physical 10.0 -> pixel 20.0
        spot_0 = data.spots_by_frame[0][0]  # ID=0, X=10.0, Y=20.0 physical
        assert spot_0[0] == 0  # spot_id
        assert spot_0[1] == pytest.approx(40.0)  # y = 20.0 / 0.5
        assert spot_0[2] == pytest.approx(20.0)  # x = 10.0 / 0.5


class TestParseTracks:
    def test_filtered_tracks(self, sample_xml_path):
        """Only filtered tracks (0 and 1) should be included, not track 2."""
        data = parse_trackmate_xml(sample_xml_path)
        track_ids = set(data.spot_to_track.values())
        assert 0 in track_ids
        assert 1 in track_ids
        assert 2 not in track_ids

    def test_track_assignment(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        # Track 0: spots 0, 3, 6
        assert data.spot_to_track[0] == 0
        assert data.spot_to_track[3] == 0
        assert data.spot_to_track[6] == 0
        # Track 1: spots 1, 4
        assert data.spot_to_track[1] == 1
        assert data.spot_to_track[4] == 1

    def test_untracked_spot(self, sample_xml_path):
        """Spot 5 belongs to filtered-out track 2, so it has no track."""
        data = parse_trackmate_xml(sample_xml_path)
        assert 5 not in data.spot_to_track

    def test_n_tracks(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.n_tracks == 2


class TestParseMetadata:
    def test_image_shape(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.image_shape == (100, 200)  # (H, W)

    def test_calibration(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.pixel_size_x == 0.5
        assert data.pixel_size_y == 0.5
        assert data.time_interval == 2.0

    def test_units(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.spatial_units == "micron"
        assert data.time_units == "sec"

    def test_n_frames(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.n_frames == 3

    def test_detector_tracker(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        assert data.metadata["detector"] == "LOG_DETECTOR"
        assert data.metadata["tracker"] == "SPARSE_LAP_TRACKER"


class TestPositionsArray:
    def test_to_positions_array(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        arr = data.to_positions_array()
        assert arr.shape == (7, 3)  # 7 spots, (frame, y, x)
        # All frame 0 rows should have frame=0
        frame0 = arr[arr[:, 0] == 0]
        assert len(frame0) == 3

    def test_to_positions_with_track_ids(self, sample_xml_path):
        data = parse_trackmate_xml(sample_xml_path)
        arr = data.to_positions_array_with_track_ids()
        assert arr.shape == (7, 4)  # 7 spots, (frame, y, x, track_id)
        # Spot with no track (spot 5, which is in track 2 but filtered out)
        # should have track_id = -1
        # Find spot 5's row: frame=1, and it's the one with untracked spot
        frame1 = arr[arr[:, 0] == 1]
        track_ids_frame1 = set(frame1[:, 3].astype(int))
        assert -1 in track_ids_frame1


class TestWithSampleData:
    """Test with the real sample XML if available."""

    @pytest.fixture
    def real_xml_path(self):
        p = Path(__file__).parent.parent / "sample_data" / "nuclear_tracks.xml"
        if not p.exists():
            pytest.skip("Sample data not available")
        return p

    def test_parse_real_xml(self, real_xml_path):
        data = parse_trackmate_xml(real_xml_path)
        assert data.n_spots > 0
        assert data.n_tracks > 0
        assert data.image_shape is not None
        assert data.pixel_size_x is not None

    def test_real_xml_positions_array(self, real_xml_path):
        data = parse_trackmate_xml(real_xml_path)
        arr = data.to_positions_array()
        assert arr.ndim == 2
        assert arr.shape[1] == 3
        # All positions should be non-negative (pixel coords)
        assert np.all(arr[:, 1:] >= 0)
