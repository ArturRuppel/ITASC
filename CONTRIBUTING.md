# Contributing to ITASC

ITASC is developed in the open, and contributions are welcome! This page covers
the three things people usually need: how to report a problem, how to get help,
and how to send a change.

## Report a bug or request a feature

Open an issue: <https://github.com/ArturRuppel/ITASC/issues>

A useful bug report says what you did, what you expected, and what happened
instead. Include:

- the ITASC version (`uv tool list`, or `pip show itasc`),
- your operating system,
- the full error text from napari's terminal window.

A small image, or one of the [`sample_data`](https://github.com/ArturRuppel/ITASC/tree/main/sample_data)
positions, that reproduces the problem helps most of all.

## Get help

For a usage question that is not clearly a bug, open an issue and label it
`question`. Read the [documentation](https://arturruppel.github.io/ITASC/) first:
the [install guide](https://arturruppel.github.io/ITASC/manual/install.html) and
the per-stage guides cover the common cases. For questions about the method
itself, or about whether ITASC suits your system, contact Artur Ruppel at
`artur@ruppel.pro`.

## Contribute a change

1. Fork the repository and branch from `main`.
2. Set up a development environment (below).
3. Make the change, with a test that fails before it and passes after.
4. Run the linter and the test suite locally.
5. Open a pull request describing what changed and why. Continuous integration
   runs the same checks on Linux, Windows, and macOS.

Keep pull requests focused: one concern per branch is easier to review and
faster to merge.

## Development setup

From a fresh clone, install the package with its development tools:

```bash
python -m pip install -e ".[dev]"
```

For a specific stage, add its extra: `.[dev,cellpose,tracking]` pulls in the
Cellpose and Ultrack solvers so their tests run. Then check the code the way CI
does:

```bash
python -m ruff check .     # lint
python -m pytest           # full test suite
```

The napari GUI tests need an OpenGL context. They run in CI on Linux (software
GL) and are skipped on Windows and macOS; to run only the headless tests
locally, `python -m pytest --ignore=tests/napari`.

## Code of conduct

Be respectful and constructive. Assume good faith, keep discussion on the
technical merits, and make this a project people are glad to take part in.
Report conduct concerns to `artur@ruppel.pro`.
