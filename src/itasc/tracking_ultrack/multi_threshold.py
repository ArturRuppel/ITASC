"""Source-node provenance queries for Ultrack databases.

Reads the optional ``itasc_ultrack_source_nodes`` table that records
which merged node ids originated from which source segmentation. Used by
:mod:`itasc.tracking_ultrack.db_query` to scope hierarchy-cut-state
queries to a single source; returns empty results when the table is absent.
"""
from __future__ import annotations

from pathlib import Path

import sqlalchemy as sqla

SOURCE_NODE_TABLE = "itasc_ultrack_source_nodes"


def _source_table_exists(conn) -> bool:
    return (
        conn.execute(
            sqla.text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name=:table_name"
            ),
            {"table_name": SOURCE_NODE_TABLE},
        ).first()
        is not None
    )


def query_source_node_ids(db_path: str | Path, source_index: int) -> tuple[int, ...]:
    """Return merged node ids that originated from ``source_index``."""
    engine = sqla.create_engine(f"sqlite:///{Path(db_path)}")
    try:
        with engine.connect() as conn:
            if not _source_table_exists(conn):
                return ()
            rows = conn.execute(
                sqla.text(
                    f"SELECT node_id FROM {SOURCE_NODE_TABLE} "
                    "WHERE source_index=:source_index ORDER BY node_id"
                ),
                {"source_index": int(source_index)},
            ).all()
            return tuple(int(row[0]) for row in rows)
    finally:
        engine.dispose()
