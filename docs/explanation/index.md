# How ITASC works

ITASC takes a time-lapse of a cell monolayer and returns, for every cell, an outline
and an identity that hold across the whole recording: its shape in each frame, and
the fact that it is the same cell from the first frame to the last. From that it
measures which cells touch and when neighbours swap. This page states the problem and
the idea behind the solution; the later pages detail each step and build on it.

```{figure} ../_static/diagrams/diagram-overview.svg
:alt: The ITASC pipeline as five stages left to right, input maps, nucleus tracking, cell bodies, contacts and T1, aggregate, with a correction box attached beneath nucleus tracking and cell bodies.
:figclass: only-light
:width: 100%

The pipeline end to end: the input maps feed nucleus tracking, each cell body grows from
its tracked nucleus, and the contacts and pooled tables follow. Correction sits on the two
segmentation-and-tracking stages, where a person fixes what the automatics miss.
```
```{figure} ../_static/diagrams/diagram-overview-dark.svg
:alt: The ITASC pipeline as five stages left to right, input maps, nucleus tracking, cell bodies, contacts and T1, aggregate, with a correction box attached beneath nucleus tracking and cell bodies.
:figclass: only-dark
:width: 100%

The pipeline end to end: the input maps feed nucleus tracking, each cell body grows from
its tracked nucleus, and the contacts and pooled tables follow. Correction sits on the two
segmentation-and-tracking stages, where a person fixes what the automatics miss.
```

## The problem

The monolayers ITASC is built for are dense and highly motile. Cells are packed with
no clear gap between them, and they travel far between frames, sliding past one
another and trading neighbours. Two jobs have to be done on this, and they interfere.

**Segmentation** draws each cell's outline in each frame. Where two cells touch, the
boundary between them is uncertain.

**Tracking** matches each cell to itself in the next frame. When cells move far and
look alike, the match is uncertain as well.

The usual pipeline does the two in order: outline every frame, then link the outlines
into tracks. It fails because the outlines are never exactly right, and they are wrong
in different ways from frame to frame. Two touching cells read as one shape in one
frame and two in the next, so the linker has nothing consistent to match and the track
breaks. On a dense, moving sheet this happens in every frame, and the errors compound.

## Segmentation and tracking together

ITASC tracks with [Ultrack](https://github.com/royerlab/ultrack), which does not commit
to one outline per frame. For each frame it builds many candidate outlines: it
over-segments the image into small fragments, then lets them merge in nested ways, so a
cell can be a single fragment, a pair, or a larger clump. It then selects, across all
frames at once, the set of outlines and links that is most consistent in time, subject
to constraints derived from biology: cells move only a short distance between frames,
may divide into two but not merge, and may not share a pixel. Because the outline and
the track are scored in a single calculation, the selected outline is also the one that
tracks correctly. The correct split of two touching cells is recovered because only
that split follows cleanly through time.

## Two consequences

**Nuclei first.** Nuclei are compact and come apart more cleanly than cell bodies, so
ITASC tracks the nuclei first, then grows each cell body outward from its tracked
nucleus. The body takes the nucleus's identity, so a nucleus and its cell stay in step
across the recording.

**A person at the end.** No solver is perfect on dense, dividing cells, and one wrong
track spoils every measurement built on it. So each automatic result is handed to a
person to correct, with the candidate outlines Ultrack already built offered as
selectable alternatives.

## The parts

- **[Preparing the input maps](input-maps.md).** Turn the raw images into the two maps
  Ultrack needs: where the cells are, and where their boundaries lie.
- **[Outlining and tracking the nuclei](nucleus-tracking.md).** Build the candidates,
  solve for the consistent set, correct it.
- **[Growing the cell bodies](cell-segmentation.md).** Extend each cell out from its
  tracked nucleus.
- **[Measuring the result](contact-analysis.md).** Cell shapes, neighbours, and the
  moments two cells swap neighbours.
