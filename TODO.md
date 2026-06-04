# TODO

## Ultrack Database Preview Mode

- [x] Deactivating preview mode should clear all preview layers.
- [x] Changing preview parameters should not make the banner or surrounding UI flicker.
- [x] Fix layout movement caused by the preview banner flicker so repeated clicks on `+` / `-` controls remain possible.

## Plugin-Wide UI Styling

- [x] Improve disabled button styling across the whole plugin: disabled buttons should retain a visible background pill and clearly read as disabled.

## Follow-Up Issues

- [x] Investigate correction-widget retrack slowdown. Initial hypothesis: the baseline retracker still has dense per-frame LAP cost, but the recent centroid visualization refresh likely made retrack much slower by rebuilding the full-stack label colormap and centroid layer after each retrack; a synthetic 30-frame 512x512 stack with ~800 labels/frame took ~0.79s for one retrack frame versus ~11.07s for full centroid rebuild. Profile a real position and consider incremental or deferred visual refresh after retrack.
- [x] Investigate why searching with ZC and extending with AD is very slow now; this may be related to the recent refactor.
- [x] Fix database cancellation safety: starting database building and then canceling appears to corrupt the database, causing subsequent steps to throw errors.
- [x] Constrain the solver power parameter to integers because non-integer exponents throw errors with negative weights.
- [x] Rename and reorganize scoring controls: the quality exponent parameter is misnamed because the exponent applies to both quality weight and circularity weight. Put the quality and circularity weight terms next to each other, place the exponent below them, and label it "Scoring exponent".
- [x] Rename "Scoring" to "Node Scoring".
- [x] Prevent changing position while in correction mode.
- [x] Investigate whether node and edge scores can exceed 1, since values above 1 may interact badly with the power parameter.
- [x] Show database-building statistics for node and edge values so users can evaluate database quality.
- [x] Reorganize database browser summary statistics: they are currently chained into one string separated with `|`, and the text does not fit its container, causing top and bottom clipping.
- [x] Fix custom button icon disabled/inactive styling: inactive custom buttons get a darker background pill, which is good, but the icon itself gets brighter, which is bad.
- [x] Remove superseded correction-mode activation logic: the "activate correction mode" banner and the disabling of widgets have been superseded by completely hiding the label layers and the original widget, so the old activation/disable logic can be removed. This stale logic also causes a bug — when there is no data to correct yet, the widget stays in an "activated" but frozen state.
- [x] Fix shift+arrow track navigation: shift+arrow should recenter both the lineage viewer and the main viewer, exactly as if the track had been clicked. It should also not scan for the next track in the current frame, but simply move to the next track in the list.
- [ ] Fix "reassign IDs to contiguous": it makes a mess with validated track numbers. It seems to work when all nodes are validated, though (duh).
