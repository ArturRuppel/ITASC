# TODO

## Ultrack Database Preview Mode

- [x] Deactivating preview mode should clear all preview layers.
- [x] Changing preview parameters should not make the banner or surrounding UI flicker.
- [x] Fix layout movement caused by the preview banner flicker so repeated clicks on `+` / `-` controls remain possible.

## Plugin-Wide UI Styling

- [x] Improve disabled button styling across the whole plugin: disabled buttons should retain a visible background pill and clearly read as disabled.

## Follow-Up Issues

- [ ] Investigate why searching with ZC and extending with AD is very slow now; this may be related to the recent refactor.
- [ ] Fix database cancellation safety: starting database building and then canceling appears to corrupt the database, causing subsequent steps to throw errors.
- [x] Constrain the solver power parameter to integers because non-integer exponents throw errors with negative weights.
- [x] Rename and reorganize scoring controls: the quality exponent parameter is misnamed because the exponent applies to both quality weight and circularity weight. Put the quality and circularity weight terms next to each other, place the exponent below them, and label it "Scoring exponent".
- [x] Rename "Scoring" to "Node Scoring".
- [x] Prevent changing position while in correction mode.
- [ ] Investigate whether node and edge scores can exceed 1, since values above 1 may interact badly with the power parameter.
- [ ] Show database-building statistics for node and edge values so users can evaluate database quality.
- [x] Fix custom button icon disabled/inactive styling: inactive custom buttons get a darker background pill, which is good, but the icon itself gets brighter, which is bad.
