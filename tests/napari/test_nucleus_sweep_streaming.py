from pathlib import Path


def test_nucleus_sweep_streams_records_to_hdf5_writer():
    source = Path("src/cellflow/napari/nucleus_workflow_widget.py").read_text()
    start = source.index("    def _on_run_sweep(self) -> None:")
    end = source.index("    def _set_sweep_buttons_running", start)
    run_sweep_source = source[start:end]

    assert "iter_write_hypothesis_sweep_h5" in run_sweep_source
    assert "collected" not in run_sweep_source
    assert "records = []" not in run_sweep_source
