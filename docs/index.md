```{include} ../README.md
:relative-docs: docs/
:relative-images:
:end-before: <!-- docs-home-start -->
```

```{include} ../README.md
:relative-docs: docs/
:relative-images:
:start-after: <!-- docs-home-end -->
:end-before: <!-- hero-start -->
```

<video autoplay loop muted playsinline width="100%"
       poster="_static/napari_timelapse_last.png">
  <source src="_static/napari_timelapse.mp4" type="video/mp4">
</video>

```{include} ../README.md
:relative-docs: docs/
:relative-images:
:start-after: <!-- hero-end -->
```

```{toctree}
:hidden:
:caption: Get started

Install <manual/install>
```

```{toctree}
:hidden:
:caption: Pick one tool

Run the whole pipeline <manual/full-app>
Segment and track sparse cells <manual/cellpose>
Track dense cells from maps <manual/tracking>
Quantify and pool positions <manual/aggregate>
Build on the Python API <manual/core>
```

```{toctree}
:hidden:
:caption: How it works

explanation/index
explanation/input-maps
explanation/nucleus-tracking
explanation/cell-segmentation
explanation/contact-analysis
```

```{toctree}
:hidden:
:caption: Reference

api/index
```
