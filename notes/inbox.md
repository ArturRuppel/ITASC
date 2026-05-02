# Notes Inbox

## 2026-05-02

- Changing `bassin` in the cell workflow causes very slow loading times, likely because large data needs to be loaded.
- Possible fixes:
  - lazy loading
  - add status updates / a loading bar so the user knows what is happening during load
- Foreground threshold in cell workflow hypothesis generation should likely be handled like the nucleus workflow:
  - provide the mask externally
  - skip the threshold sweep
  - figure out how to project the data into 2D afterwards in a way that stays smooth and sensible
- Add a "run in terminal" button for contour map creation.
- Change the contour maps widget data contract:
  - foreground masks are not an output of that step
- Add status labels for input and output data in the hypothesis generator.
- Add status labels for input and output data in the database browser.
