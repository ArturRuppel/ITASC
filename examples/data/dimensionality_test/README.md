# Dimensionality test fixtures

Small raw-intensity stacks for exercising the segmentation / divergence-map path on
all four input layouts. Regenerate with `python scripts/make_dimensionality_fixtures.py`.

**Provenance:** derived from `examples/data/full_example/pos00/0_input/{nucleus,cell}_3dt.tif`
(real `(T=10, Z=10, 256, 256)` uint16), central-cropped to 128×128 and trimmed to
T=3 / Z=3. Real cell content, deliberately tiny.

| file                | axes | shape             | feed with layout |
|---------------------|------|-------------------|------------------|
| `{ch}_2d.tif`       | YX   | (128, 128)        | **2D**           |
| `{ch}_2dt.tif`      | TYX  | (3, 128, 128)     | **2D+t**         |
| `{ch}_3d.tif`       | ZYX  | (3, 128, 128)     | **3D**           |
| `{ch}_3dt.tif`      | TZYX | (3, 3, 128, 128)  | **3D+t**         |

`{ch}` is `nucleus` or `cell`.

The `2dt` and `3d` files are **shape-identical** on disk (`(3, 128, 128)`);
`infer_layout_from_ndim(3)` returns `None`, so only the axes metadata (`TYX` vs
`ZYX`) tells them apart. Pick the matching layout in the Cellpose widget — that is
exactly the ambiguity the dimensionality check needs to cover.

**Path they exercise:** raw stack → `cellpose_runner.to_tzyx(arr, layout)` →
`run_{nucleus,cell}_stack` (with `do_3d=False` for the divergence pipeline) →
`write_outputs` (`{ch}_prob_3dt.tif`, `{ch}_dp_3dt.tif`, singleton T/Z preserved via
axes metadata) → `build_divergence_maps` → foreground + contour maps.
