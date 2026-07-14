"""Low-level HDF5 table round-trip for the dynamics store."""
from __future__ import annotations

import h5py
import numpy as np

from itasc.contact_analysis.dynamics.store import _read_table


def test_read_table_preserves_authored_column_order(tmp_path):
    """h5py iterates datasets alphabetically; _read_table must return columns in
    the order they were authored, recorded in the column_order attr."""
    # Deliberately non-alphabetical authoring order.
    authored = ["cell_id", "msd_D_um2_per_s", "alpha", "beta"]
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("table")
        for i, name in enumerate(authored):
            g.create_dataset(name, data=np.array([float(i)]))
        g.attrs["column_order"] = authored

    with h5py.File(path, "r") as h5:
        out = _read_table(h5["table"])

    assert list(out.keys()) == authored  # not sorted(authored)


def test_read_table_legacy_without_order_attr_still_reads(tmp_path):
    """A file written before column_order existed must still read (alphabetical
    fallback), not crash."""
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("table")
        g.create_dataset("b", data=np.array([1.0]))
        g.create_dataset("a", data=np.array([2.0]))

    with h5py.File(path, "r") as h5:
        out = _read_table(h5["table"])

    assert set(out.keys()) == {"a", "b"}
    assert out["a"][0] == 2.0 and out["b"][0] == 1.0
