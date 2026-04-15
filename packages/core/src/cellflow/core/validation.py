"""Input validation helpers used by every stage's ``validate_inputs()``."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


class ValidationResult(NamedTuple):
    ok: bool
    errors: List[str]


def validate_tiff_header(
    path: Path,
    expected_dtype: Optional[str] = None,
) -> List[str]:
    """Return a list of error strings (empty list means OK).

    Only the TIFF header is read — the pixel data is never loaded.
    """
    errors: List[str] = []
    if not path.exists():
        errors.append(f"File not found: {path}")
        return errors
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            if expected_dtype is not None and str(page.dtype) != expected_dtype:
                errors.append(
                    f"{path.name}: expected dtype {expected_dtype!r}, "
                    f"got {page.dtype!r}"
                )
    except Exception as exc:
        errors.append(f"{path.name}: failed to read TIFF header — {exc}")
    return errors


def validate_inputs(
    required_paths: List[Path],
    tiff_specs: Optional[Dict[Path, Dict[str, str]]] = None,
) -> ValidationResult:
    """Check file existence and optional TIFF header constraints.

    Parameters
    ----------
    required_paths:
        Every path in this list must exist.
    tiff_specs:
        Optional mapping of ``path → {"dtype": "<dtype>"}`` for files
        that must additionally pass a TIFF header check.  Only files
        that already exist are checked (missing files are reported via
        *required_paths* instead).
    """
    errors: List[str] = []

    for p in required_paths:
        if not p.exists():
            errors.append(f"Required file not found: {p}")

    if tiff_specs:
        for path, spec in tiff_specs.items():
            if path.exists():
                errors.extend(
                    validate_tiff_header(path, expected_dtype=spec.get("dtype"))
                )

    return ValidationResult(ok=len(errors) == 0, errors=errors)
