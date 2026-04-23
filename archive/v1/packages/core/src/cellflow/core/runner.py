"""Stage runner — manifest bookkeeping wrapper for stage generators.

This module is intentionally free of napari imports so it can be used in
headless / test environments.  The napari ``thread_worker`` decoration
belongs in ``cellflow.napari`` — not here.

Typical usage inside a napari widget::

    from napari.qt.threading import thread_worker
    from cellflow.core.runner import run_in_thread

    @thread_worker(connect={"yielded": _on_progress})
    def _worker():
        yield from run_in_thread(
            stage.run,
            stage_name="nucleus_ultrack",
            manifest=manifest,
            manifest_path=paths.manifest_path(root, pos),
            config=stage.config,
            root_dir=root,
            pos=pos,
            overwrite=False,
        )
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional

if TYPE_CHECKING:
    from cellflow.core.manifest import PipelineManifest
    from cellflow.core.protocol import StageProgress


def config_hash(config_obj: Any) -> str:
    """Return a 16-char hex digest of a config object.

    Supports Pydantic v2 models (via ``model_dump_json``), plain dicts,
    and any other JSON-serialisable object.
    """
    if hasattr(config_obj, "model_dump_json"):
        raw = config_obj.model_dump_json(exclude_none=False)
    else:
        raw = json.dumps(config_obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_in_thread(
    stage_fn: Callable[..., Generator["StageProgress", None, None]],
    stage_name: str,
    manifest: "PipelineManifest",
    manifest_path: Path,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Generator["StageProgress", None, None]:
    """Wrap a stage generator with manifest bookkeeping.

    Yields every :class:`~cellflow.core.protocol.StageProgress` the
    underlying generator produces, then writes the manifest:

    * ``running`` before the first yield
    * ``complete`` on clean exit
    * ``failed`` when the stage generator raises

    Parameters
    ----------
    stage_fn:
        A callable that returns a ``StageProgress`` generator.
    stage_name:
        Key used when updating the manifest.
    manifest:
        :class:`~cellflow.core.manifest.PipelineManifest` instance to mutate.
    manifest_path:
        File path for the manifest (passed to ``manifest.save()``).
    config:
        Optional Pydantic model or dict; its hash is stored in the manifest.
    **kwargs:
        Forwarded verbatim to *stage_fn*.
    """
    manifest.mark_running(stage_name)
    manifest.save(manifest_path)
    try:
        yield from stage_fn(**kwargs)
        chash = config_hash(config) if config is not None else None
        manifest.mark_complete(stage_name, config_hash=chash)
        manifest.save(manifest_path)
    except Exception as exc:
        manifest.mark_failed(stage_name, error=str(exc))
        manifest.save(manifest_path)
        raise
