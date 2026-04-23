from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ultrack" / "src"))

from cellflow.ultrack.config import TrackingConfig
from cellflow.ultrack.pruning import (
    _circularity,
    _node_from_pickle_value,
    prune_circularity_filtered_candidates,
)
from cellflow.ultrack.stages.tracking import run_segmentation


# ---------------------------------------------------------------------------
# _circularity unit tests
# ---------------------------------------------------------------------------

def test_circularity_perfect_circle():
    r = 50
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = (x*x + y*y) <= r*r
    c = _circularity(int(mask.sum()), mask)
    assert 0.85 < c <= 1.0


def test_circularity_square_is_lower_than_circle():
    square = np.ones((20, 20), dtype=bool)
    r = 15
    y, x = np.ogrid[-r:r+1, -r:r+1]
    circle = (x*x + y*y) <= r*r
    assert _circularity(int(circle.sum()), circle) > _circularity(int(square.sum()), square)


def test_circularity_zero_perimeter_returns_zero():
    mask = np.ones((1, 1), dtype=bool)
    assert _circularity(1, mask) == 0.0


def test_node_from_pickle_value_accepts_decoded_objects():
    node = types.SimpleNamespace(mask=np.ones((2, 2), dtype=bool))
    assert _node_from_pickle_value(node) is node


def test_node_from_pickle_value_accepts_pickled_bytes():
    node = types.SimpleNamespace(mask=np.ones((2, 2), dtype=bool))
    restored = _node_from_pickle_value(pickle.dumps(node))
    assert np.array_equal(restored.mask, node.mask)


# ---------------------------------------------------------------------------
# prune_circularity_filtered_candidates integration test
# ---------------------------------------------------------------------------

def _make_mask(circularity_approx: str) -> bytes:
    """Return a pickled fake Node with a mask whose circularity matches description."""
    if circularity_approx == "circle":
        r = 20
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = (x*x + y*y) <= r*r
    else:  # "line" — very elongated, low circularity
        mask = np.zeros((3, 50), dtype=bool)
        mask[1, :] = True

    node = types.SimpleNamespace(mask=mask)
    return pickle.dumps(node)


def _setup_fake_ultrack(monkeypatch, tmp_path):
    fake_ultrack = types.ModuleType("ultrack")
    fake_ultrack.__path__ = []
    fake_core = types.ModuleType("ultrack.core")
    fake_core.__path__ = []
    fake_linking_pkg = types.ModuleType("ultrack.core.linking")
    fake_linking_pkg.__path__ = []
    fake_db = types.ModuleType("ultrack.core.database")
    fake_linking_utils = types.ModuleType("ultrack.core.linking.utils")

    captured = {}

    def fake_clear_linking_data(database_path):
        captured["database_path"] = database_path

    fake_linking_utils.clear_linking_data = fake_clear_linking_data
    fake_config = types.ModuleType("ultrack.config")
    fake_config.__package__ = "ultrack"

    class FakeMainConfig:
        def __init__(self, **kwargs):
            self.data_config = types.SimpleNamespace(
                database_path=f"sqlite:///{tmp_path / 'data.db'}"
            )

    fake_config.MainConfig = FakeMainConfig

    monkeypatch.setitem(sys.modules, "ultrack", fake_ultrack)
    monkeypatch.setitem(sys.modules, "ultrack.core", fake_core)
    monkeypatch.setitem(sys.modules, "ultrack.core.linking", fake_linking_pkg)
    monkeypatch.setitem(sys.modules, "ultrack.core.linking.utils", fake_linking_utils)
    monkeypatch.setitem(sys.modules, "ultrack.config", fake_config)

    return fake_db, fake_config, captured


def test_prune_circularity_removes_low_circularity_nodes(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("skimage")

    from sqlalchemy import BigInteger, Column, Integer, LargeBinary, create_engine, select
    from sqlalchemy.orm import Session, declarative_base

    base = declarative_base()

    class NodeDB(base):  # type: ignore[misc,valid-type]
        __tablename__ = "nodes"
        t = Column(Integer, primary_key=True)
        id = Column(BigInteger, primary_key=True)
        area = Column(Integer)
        pickle = Column(LargeBinary)

    class OverlapDB(base):  # type: ignore[misc,valid-type]
        __tablename__ = "overlaps"
        id = Column(Integer, primary_key=True, autoincrement=True)
        node_id = Column(BigInteger)
        ancestor_id = Column(BigInteger)

    engine = create_engine(f"sqlite:///{tmp_path / 'data.db'}")
    base.metadata.create_all(engine)

    r = 20
    y, x = np.ogrid[-r:r+1, -r:r+1]
    circle_mask = (x*x + y*y) <= r*r
    circle_area = int(circle_mask.sum())

    line_mask = np.zeros((3, 50), dtype=bool)
    line_mask[1, :] = True
    line_area = int(line_mask.sum())

    with Session(engine) as session:
        session.add_all([
            NodeDB(t=0, id=1, area=circle_area, pickle=pickle.dumps(types.SimpleNamespace(mask=circle_mask))),
            NodeDB(t=0, id=2, area=line_area,   pickle=pickle.dumps(types.SimpleNamespace(mask=line_mask))),
            OverlapDB(node_id=1, ancestor_id=2),
            OverlapDB(node_id=2, ancestor_id=1),
        ])
        session.commit()

    fake_db, fake_config, captured = _setup_fake_ultrack(monkeypatch, tmp_path)
    fake_db.NodeDB = NodeDB
    fake_db.OverlapDB = OverlapDB
    monkeypatch.setitem(sys.modules, "ultrack.core.database", fake_db)

    removed = prune_circularity_filtered_candidates(
        tmp_path,
        TrackingConfig(min_circularity=0.5),
    )

    assert removed == 1
    assert "database_path" in captured

    with Session(engine) as session:
        remaining = session.scalars(select(NodeDB.id).order_by(NodeDB.id)).all()
        remaining_overlaps = session.execute(
            select(OverlapDB.node_id, OverlapDB.ancestor_id).order_by(OverlapDB.id)
        ).all()

    assert remaining == [1]
    assert remaining_overlaps == []


def test_prune_circularity_skips_when_disabled(tmp_path, monkeypatch):
    """min_circularity=0 must return 0 without touching the DB."""
    cfg = TrackingConfig(min_circularity=0.0)
    removed = prune_circularity_filtered_candidates(tmp_path, cfg)
    assert removed == 0


def test_prune_circularity_parallel(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("skimage")

    from sqlalchemy import BigInteger, Column, Integer, LargeBinary, create_engine
    from sqlalchemy.orm import Session, declarative_base

    base = declarative_base()

    class NodeDB(base):  # type: ignore[misc,valid-type]
        __tablename__ = "nodes"
        t = Column(Integer, primary_key=True)
        id = Column(BigInteger, primary_key=True)
        area = Column(Integer)
        pickle = Column(LargeBinary)

    class OverlapDB(base):  # type: ignore[misc,valid-type]
        __tablename__ = "overlaps"
        id = Column(Integer, primary_key=True, autoincrement=True)
        node_id = Column(BigInteger)
        ancestor_id = Column(BigInteger)

    engine = create_engine(f"sqlite:///{tmp_path / 'data.db'}")
    base.metadata.create_all(engine)

    r = 20
    y, x = np.ogrid[-r:r+1, -r:r+1]
    circle_mask = (x*x + y*y) <= r*r
    circle_area = int(circle_mask.sum())

    line_mask = np.zeros((3, 50), dtype=bool)
    line_mask[1, :] = True
    line_area = int(line_mask.sum())

    with Session(engine) as session:
        session.add_all([
            NodeDB(t=0, id=1, area=circle_area, pickle=pickle.dumps(types.SimpleNamespace(mask=circle_mask))),
            NodeDB(t=0, id=2, area=line_area,   pickle=pickle.dumps(types.SimpleNamespace(mask=line_mask))),
            NodeDB(t=1, id=3, area=circle_area, pickle=pickle.dumps(types.SimpleNamespace(mask=circle_mask))),
            NodeDB(t=1, id=4, area=line_area,   pickle=pickle.dumps(types.SimpleNamespace(mask=line_mask))),
        ])
        session.commit()

    fake_db, fake_config, captured = _setup_fake_ultrack(monkeypatch, tmp_path)
    fake_db.NodeDB = NodeDB
    fake_db.OverlapDB = OverlapDB
    monkeypatch.setitem(sys.modules, "ultrack.core.database", fake_db)

    removed = prune_circularity_filtered_candidates(
        tmp_path,
        TrackingConfig(min_circularity=0.5),
        n_workers=2,
    )

    assert removed == 2


def test_run_segmentation_invokes_circularity_pruning(tmp_path, monkeypatch):
    import tifffile

    foreground = np.ones((1, 2, 2), dtype=np.float32)
    contours = np.zeros((1, 2, 2), dtype=np.float32)
    foreground_path = tmp_path / "foreground.tif"
    contours_path = tmp_path / "contours.tif"
    tifffile.imwrite(str(foreground_path), foreground)
    tifffile.imwrite(str(contours_path), contours)

    captured = {}

    def fake_segment(_fg, _ct, _cfg, overwrite=False):
        captured["segment_called"] = True

    def fake_prune(working_dir, cfg, *, n_workers=1):
        captured["prune_working_dir"] = working_dir
        captured["prune_cfg"] = cfg
        captured["prune_n_workers"] = n_workers
        return 0

    fake_ultrack = types.ModuleType("ultrack")
    fake_ultrack.__path__ = []
    fake_core = types.ModuleType("ultrack.core")
    fake_core.__path__ = []
    fake_seg_pkg = types.ModuleType("ultrack.core.segmentation")
    fake_seg_pkg.__path__ = []
    fake_db = types.ModuleType("ultrack.core.database")
    fake_db.clear_all_data = lambda _: None
    fake_seg = types.ModuleType("ultrack.core.segmentation.processing")
    fake_seg.segment = fake_segment

    monkeypatch.setitem(sys.modules, "ultrack", fake_ultrack)
    monkeypatch.setitem(sys.modules, "ultrack.core", fake_core)
    monkeypatch.setitem(sys.modules, "ultrack.core.database", fake_db)
    monkeypatch.setitem(sys.modules, "ultrack.core.segmentation", fake_seg_pkg)
    monkeypatch.setitem(sys.modules, "ultrack.core.segmentation.processing", fake_seg)

    fake_config = types.ModuleType("ultrack.config")
    fake_config.__package__ = "ultrack"

    class FakeMainConfig:
        def __init__(self, **kwargs):
            self.data_config = types.SimpleNamespace(database_path=tmp_path / "data.db")

    fake_config.MainConfig = FakeMainConfig
    monkeypatch.setitem(sys.modules, "ultrack.config", fake_config)
    monkeypatch.setattr(
        "cellflow.ultrack.stages.tracking.prune_circularity_filtered_candidates",
        fake_prune,
    )

    cfg = TrackingConfig(n_workers=1)
    progress = list(run_segmentation(foreground_path, contours_path, tmp_path, cfg, overwrite=True))

    assert captured["segment_called"] is True
    assert captured["prune_working_dir"] == tmp_path
    assert captured["prune_cfg"] == cfg
    assert captured["prune_n_workers"] == 1
    assert progress[-1][2] == "Segmentation done."
