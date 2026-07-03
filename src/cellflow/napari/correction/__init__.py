"""Interactive label-correction subsystem for the CellFlow napari plugin.

Groups the correction widgets (:mod:`correction_widget`,
:mod:`nucleus_correction_widget`, :mod:`cell_correction_widget`) and their
``_correction_*`` helper modules. Importers use the fully-qualified
``cellflow.napari.correction.<module>`` path; this package intentionally
re-exports nothing so the module boundaries stay explicit.
"""
