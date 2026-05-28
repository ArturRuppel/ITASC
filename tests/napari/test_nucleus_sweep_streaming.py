from __future__ import annotations

import numpy as np

from cellflow.database.hypotheses import (
    HypothesisRecord,
    SeededWatershedParams,
    iter_write_hypothesis_sweep_h5,
    read_hypothesis_labels,
)


def test_hypothesis_sweep_writer_streams_records(tmp_path):
    consumed = []

    def records():
        for t in range(2):
            consumed.append(t)
            yield HypothesisRecord(
                t=t,
                p=0,
                labels=np.full((1, 3, 3), t + 1, dtype=np.uint32),
                params=SeededWatershedParams(),
            )

    progress = iter_write_hypothesis_sweep_h5(
        tmp_path / "hypotheses.h5",
        records(),
        overwrite=True,
    )

    assert consumed == []
    assert next(progress) == 1
    assert consumed == [0]
    assert np.array_equal(
        read_hypothesis_labels(tmp_path / "hypotheses.h5", 0, 0),
        np.full((1, 3, 3), 1, dtype=np.uint32),
    )
    assert list(progress) == [2]
