"""Tests for cellflow.core — Phase 1 contracts."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellflow.core.logging import StageLogger, stage_logger
from cellflow.core.manifest import PipelineManifest, StageRecord
from cellflow.core.paths import (
    STAGE_DIRS,
    log_path,
    manifest_path,
    pos_dir,
    resolve_interface_path,
    schema_path,
    stage_dir,
)
from cellflow.core.protocol import StageProgress, ValidationResult
from cellflow.core.runner import config_hash, run_in_thread
from cellflow.core.schema import InterfaceSpec, PipelineSchema
from cellflow.core.validation import validate_inputs, validate_tiff_header


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------


class TestStageProgress:
    def test_namedtuple_fields(self):
        p = StageProgress(done=3, total=10, message="working")
        assert p.done == 3
        assert p.total == 10
        assert p.message == "working"

    def test_unpacking(self):
        done, total, msg = StageProgress(1, 5, "hello")
        assert done == 1 and total == 5 and msg == "hello"


class TestValidationResult:
    def test_ok(self):
        r = ValidationResult(ok=True, errors=[])
        assert r.ok is True
        assert r.errors == []

    def test_not_ok(self):
        r = ValidationResult(ok=False, errors=["missing file"])
        assert not r.ok


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


class TestPipelineSchema:
    def test_defaults(self):
        s = PipelineSchema()
        assert s.schema_version == "1.0"
        assert s.stages == []
        assert s.interfaces == {}

    def test_round_trip(self, tmp_path):
        s = PipelineSchema(
            stages=["raw_import", "tracking"],
            interfaces={
                "tracking.output.tracked_labels": InterfaceSpec(
                    path_template="pos{pos:02d}/3_tracking/tracked_labels.tif",
                    shape="THW",
                    dtype="uint32",
                )
            },
        )
        p = tmp_path / "pipeline_schema.json"
        s.save(p)
        loaded = PipelineSchema.load(p)
        assert loaded.stages == ["raw_import", "tracking"]
        spec = loaded.interfaces["tracking.output.tracked_labels"]
        assert spec.shape == "THW"
        assert spec.dtype == "uint32"

    def test_save_creates_parent_dir(self, tmp_path):
        s = PipelineSchema(stages=["raw_import"])
        dest = tmp_path / "new_dir" / "pipeline_schema.json"
        s.save(dest)
        assert dest.exists()

    def test_load_invalid_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(Exception):
            PipelineSchema.load(p)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class TestPipelineManifest:
    def test_empty_load_from_missing_file(self, tmp_path):
        m = PipelineManifest.load(tmp_path / "nonexistent.json")
        assert m.stages == {}

    def test_mark_running(self):
        m = PipelineManifest()
        m.mark_running("cellpose_nucleus")
        assert m.stages["cellpose_nucleus"].status == "running"

    def test_mark_complete(self):
        m = PipelineManifest()
        m.mark_complete("cellpose_nucleus", config_hash="abc123")
        rec = m.stages["cellpose_nucleus"]
        assert rec.status == "complete"
        assert rec.config_hash == "abc123"
        assert rec.finished_at is not None

    def test_mark_failed(self):
        m = PipelineManifest()
        m.mark_running("tracking")
        m.mark_failed("tracking", error="OOM")
        assert m.stages["tracking"].status == "failed"
        assert m.stages["tracking"].error == "OOM"

    def test_mark_stale(self):
        m = PipelineManifest()
        m.mark_complete("tracking", config_hash="xyz")
        m.mark_stale("tracking")
        assert m.stages["tracking"].status == "stale"
        assert m.stages["tracking"].config_hash == "xyz"

    def test_is_complete(self):
        m = PipelineManifest()
        assert not m.is_complete("foo")
        m.mark_complete("foo")
        assert m.is_complete("foo")

    def test_is_runnable(self):
        m = PipelineManifest()
        assert m.is_runnable("foo")  # pending by default
        m.mark_complete("foo")
        assert not m.is_runnable("foo")
        m.mark_stale("foo")
        assert m.is_runnable("foo")

    def test_atomic_save_round_trip(self, tmp_path):
        m = PipelineManifest()
        m.mark_complete("raw_import", config_hash="abc")
        p = tmp_path / "pos00" / "pipeline_manifest.json"
        m.save(p)
        assert p.exists()
        loaded = PipelineManifest.load(p)
        assert loaded.stages["raw_import"].status == "complete"

    def test_no_temp_files_left_after_save(self, tmp_path):
        m = PipelineManifest()
        m.mark_complete("raw_import")
        p = tmp_path / "manifest.json"
        m.save(p)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


class TestPaths:
    def test_pos_dir(self):
        assert pos_dir("/exp", 0) == Path("/exp/pos00")
        assert pos_dir("/exp", 12) == Path("/exp/pos12")

    def test_stage_dir_known(self):
        assert stage_dir("/exp", 0, "tracking") == Path("/exp/pos00/3_tracking")
        assert stage_dir("/exp", 0, "raw_import") == Path("/exp/pos00/0_raw")

    def test_stage_dir_unknown_falls_back_to_name(self):
        assert stage_dir("/exp", 0, "my_custom_stage") == Path(
            "/exp/pos00/my_custom_stage"
        )

    def test_manifest_path(self):
        assert manifest_path("/exp", 1) == Path("/exp/pos01/pipeline_manifest.json")

    def test_schema_path(self):
        assert schema_path("/exp") == Path("/exp/pipeline_schema.json")

    def test_log_path(self):
        assert log_path("/exp", 3) == Path("/exp/pos03/pipeline.log")

    def test_resolve_interface_path(self):
        template = "pos{pos:02d}/3_tracking/tracked_labels.tif"
        p = resolve_interface_path("/exp", 5, template)
        assert p == Path("/exp/pos05/3_tracking/tracked_labels.tif")

    def test_resolve_interface_path_with_stem(self):
        template = "pos{pos:02d}/1a_cellpose_nucleus/{stem}_dp.tif"
        p = resolve_interface_path("/exp", 2, template, stem="frame_001")
        assert p == Path("/exp/pos02/1a_cellpose_nucleus/frame_001_dp.tif")

    def test_stage_dirs_covers_all_expected_stages(self):
        expected = {
            "raw_import",
            "cellpose_nucleus",
            "cellpose_cell",
            "flow_watershed",
            "contours",
            "tracking",
            "graph_extraction",
            "topology_analysis",
        }
        assert expected <= set(STAGE_DIRS.keys())


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_file_returns_error(self, tmp_path):
        result = validate_inputs([tmp_path / "nonexistent.tif"])
        assert not result.ok
        assert len(result.errors) == 1

    def test_existing_file_passes(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        result = validate_inputs([p])
        assert result.ok

    def test_tiff_dtype_mismatch_returns_error(self, tmp_path):
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), np.zeros((4, 4), dtype=np.uint16))
        result = validate_inputs([], tiff_specs={p: {"dtype": "float32"}})
        assert not result.ok
        assert any("float32" in e for e in result.errors)

    def test_tiff_dtype_match_passes(self, tmp_path):
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), np.zeros((4, 4), dtype=np.float32))
        result = validate_inputs([], tiff_specs={p: {"dtype": "float32"}})
        assert result.ok

    def test_missing_tiff_reported_via_required_paths(self, tmp_path):
        p = tmp_path / "missing.tif"
        result = validate_inputs([p], tiff_specs={p: {"dtype": "uint8"}})
        assert not result.ok
        # reported once via required_paths; tiff_specs skips missing files
        assert sum(1 for e in result.errors if "not found" in e) == 1


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_deterministic(self):
        from pydantic import BaseModel

        class Cfg(BaseModel):
            lr: float = 0.01
            epochs: int = 5

        assert config_hash(Cfg()) == config_hash(Cfg())

    def test_different_configs_differ(self):
        from pydantic import BaseModel

        class Cfg(BaseModel):
            lr: float

        assert config_hash(Cfg(lr=0.01)) != config_hash(Cfg(lr=0.1))

    def test_dict_input(self):
        h1 = config_hash({"a": 1, "b": 2})
        h2 = config_hash({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys=True

    def test_length_16(self):
        assert len(config_hash({"x": 1})) == 16


class TestRunInThread:
    def _make_stage(self, items):
        def stage_fn(**kwargs):
            for i, msg in enumerate(items):
                yield StageProgress(i + 1, len(items), msg)

        return stage_fn

    def test_progress_forwarded(self, tmp_path):
        m = PipelineManifest()
        p = tmp_path / "manifest.json"
        stage_fn = self._make_stage(["step1", "step2", "step3"])
        results = list(run_in_thread(stage_fn, "test_stage", m, p))
        assert len(results) == 3
        assert results[0].message == "step1"

    def test_manifest_marked_complete(self, tmp_path):
        m = PipelineManifest()
        p = tmp_path / "manifest.json"
        list(run_in_thread(self._make_stage(["a"]), "s", m, p))
        assert m.stages["s"].status == "complete"
        assert PipelineManifest.load(p).stages["s"].status == "complete"

    def test_manifest_marked_failed_on_exception(self, tmp_path):
        def bad_stage(**kwargs):
            yield StageProgress(0, 1, "starting")
            raise RuntimeError("boom")

        m = PipelineManifest()
        p = tmp_path / "manifest.json"
        with pytest.raises(RuntimeError, match="boom"):
            list(run_in_thread(bad_stage, "bad", m, p))
        assert m.stages["bad"].status == "failed"
        assert "boom" in m.stages["bad"].error


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


class TestStageLogger:
    def test_writes_json_lines(self, tmp_path):
        log = tmp_path / "pipeline.log"
        with StageLogger(log, "raw_import") as sl:
            sl.info("Processing frame 1")
            sl.warning("Slow disk")

        lines = log.read_text().strip().splitlines()
        # __enter__ writes "Stage started", info, warning, __exit__ writes "Stage completed"
        assert len(lines) == 4
        for line in lines:
            obj = json.loads(line)
            assert obj["stage"] == "raw_import"
            assert "ts" in obj
            assert "level" in obj
            assert "message" in obj

    def test_also_writes_stage_log(self, tmp_path):
        pipeline_log = tmp_path / "pipeline.log"
        stage_log = tmp_path / "stage.log"
        with StageLogger(pipeline_log, "cellpose_nucleus", stage_log=stage_log):
            pass
        assert pipeline_log.exists()
        assert stage_log.exists()
        assert pipeline_log.read_text() == stage_log.read_text()

    def test_error_logged_on_exception(self, tmp_path):
        log = tmp_path / "pipeline.log"
        sl = StageLogger(log, "tracking")
        with pytest.raises(ValueError):
            with sl:
                raise ValueError("bad input")
        lines = [json.loads(l) for l in log.read_text().strip().splitlines()]
        levels = [e["level"] for e in lines]
        assert "ERROR" in levels

    def test_does_not_suppress_exceptions(self, tmp_path):
        log = tmp_path / "pipeline.log"
        with pytest.raises(RuntimeError):
            with StageLogger(log, "foo"):
                raise RuntimeError("propagated")

    def test_stage_logger_context_manager(self, tmp_path):
        log = tmp_path / "pipeline.log"
        with stage_logger(log, "graph_extraction") as sl:
            sl.info("hello")
        assert log.exists()

    def test_creates_parent_directories(self, tmp_path):
        log = tmp_path / "deep" / "nested" / "pipeline.log"
        with StageLogger(log, "foo") as sl:
            sl.info("test")
        assert log.exists()
