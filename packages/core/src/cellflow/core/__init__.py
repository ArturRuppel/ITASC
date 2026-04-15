"""cellflow.core — shared contracts, schemas, and utilities for all pipeline stages."""
from cellflow.core.logging import StageLogger, stage_logger
from cellflow.core.manifest import PipelineManifest, StageRecord, StageStatus
from cellflow.core.paths import (
    STAGE_DIRS,
    log_path,
    manifest_path,
    pos_dir,
    project_config_path,
    resolve_interface_path,
    schema_path,
    stage_dir,
)
from cellflow.core.protocol import StageProgress, StageProtocol, ValidationResult
from cellflow.core.runner import config_hash, run_in_thread
from cellflow.core.schema import InterfaceSpec, PipelineMetadata, PipelineSchema
from cellflow.core.validation import validate_inputs, validate_tiff_header

__all__ = [
    # protocol
    "StageProgress",
    "StageProtocol",
    "ValidationResult",
    # schema
    "PipelineSchema",
    "InterfaceSpec",
    "PipelineMetadata",
    # manifest
    "PipelineManifest",
    "StageRecord",
    "StageStatus",
    # paths
    "STAGE_DIRS",
    "pos_dir",
    "stage_dir",
    "manifest_path",
    "schema_path",
    "log_path",
    "project_config_path",
    "resolve_interface_path",
    # validation
    "validate_inputs",
    "validate_tiff_header",
    # runner
    "run_in_thread",
    "config_hash",
    # logging
    "StageLogger",
    "stage_logger",
]
