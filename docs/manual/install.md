# Install

ITASC is a napari plugin, so running it takes three things: a Python, an
isolated environment to keep that Python clean, and the ITASC package installed
into it. One tool, [uv](https://docs.astral.sh/uv/), does all three. You do not
need Python already, you do not need to know what an environment is, and nothing
here touches software you already have.

This page is written for a Windows machine with nothing installed yet. macOS and
Linux differ only in the first command, noted where it matters. It installs the
full pipeline, `itasc[all]`; installing a single stage instead changes one word
in the final command, listed under [Other distributions](#other-distributions).
The uv setup is the same for every one of them, so start here.

> **Already have a Python environment?** Run `uv pip install "itasc[all]"` (or
> `pip install "itasc[all]"`) into it and jump to [Launch and
> check](#launch-and-check). The rest of this page sets up that environment for
> you.

## Install uv

uv is a single small program that installs Pythons, builds environments, and
installs packages. Install it once and it manages the rest.

On **Windows**, open PowerShell: press the Start key, type `PowerShell`, and
press Enter. In the window that opens, paste this line and press Enter:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On **macOS or Linux**, open a terminal and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The installer edits your `PATH`, which the terminal only reads at startup. Close
this terminal window and open a new one, then confirm uv is there:

```bash
uv --version
```

A version number means uv is installed. A "command not found" means the terminal
is still the old one: close every terminal window and open a fresh one.

## Install ITASC

One command installs everything:

```bash
uv tool install napari --with "itasc[all]"
```

This downloads a Python for you, builds an isolated environment, installs napari
together with ITASC and every stage's engine (Cellpose-SAM for segmentation, the
Ultrack solver for tracking) into it, and puts a `napari` command on your
`PATH`. The `[all]` extra is what pulls in those engines; without it you get the
plugin but not the heavy machine-learning dependencies.

The download is large: the segmentation engine ships PyTorch, which is over a
gigabyte, so the first install runs for several minutes and is quiet while it
works. Let it finish.

The environment it builds is isolated: it will not disturb any other Python on
your machine, and removing ITASC later is one command (see [Update and
remove](#update-and-remove)).

> **Before the first release.** ITASC is not yet published on PyPI, so the
> command above cannot find it yet. Until the release, install the current
> version straight from the source repository, which produces the same result:
>
> ```bash
> uv tool install napari --with "itasc[all] @ git+https://github.com/ArturRuppel/ITASC.git"
> ```

## Launch and check

Start napari from the terminal:

```bash
napari
```

An empty napari window opens. In its menu bar, open **Plugins → ITASC → ITASC**.
The ITASC workflow widget docks on the right of the viewer. That widget appearing
is the check that the install worked.

> 📷 **Screenshot:** the napari window just after opening the plugin, the ITASC
> workflow widget docked on the right with its stage sections collapsed.

If `napari` reports "command not found", the terminal has not picked up the new
`PATH`: close it and open a fresh one. If napari opens but **ITASC** is missing
from the **Plugins** menu, the `--with "itasc[all]"` part did not take: rerun the
install command from the previous section.

With the plugin open, the [full-app guide](full-app.md) walks through the project
layout on disk and the five stages in order.

## GPU

The GPU matters only for the distributions that segment: `itasc[all]` and
`itasc-cellpose`. The tracking and aggregate stages run no deep-learning
inference, so this section does not apply to them.

ITASC runs on the CPU with no extra setup, and everything works there.
Segmentation with Cellpose-SAM is the one heavy step, and on an NVIDIA GPU it
runs many times faster. The GPU is used automatically whenever PyTorch detects a
CUDA-capable card, so on most NVIDIA machines the default install already uses
it.

If segmentation stays slow and you have an NVIDIA GPU, PyTorch has fallen back to
CPU and needs a CUDA build matched to your driver. Two guides give the exact
command for your setup:

- PyTorch's [Get Started](https://pytorch.org/get-started/locally/) selector,
  which builds the install command for your CUDA version.
- Cellpose's [GPU installation
  notes](https://cellpose.readthedocs.io/en/latest/installation.html#gpu-version-cuda-on-windows-and-linux),
  which cover the same for the segmentation engine specifically.

Install the PyTorch they specify into the same environment, and segmentation
picks up the GPU on the next run.

## Other distributions

The command above installs the whole pipeline. Each stage also ships on its own,
for running one job on its own data. The uv setup is identical; only the package
in the `--with` changes, along with the extras that carry its engine:

```bash
# Sparse cells: Cellpose-SAM segmentation, then laptrack linking
uv tool install napari --with "itasc-cellpose[cellpose,laptrack]"

# Foreground/contour maps to Ultrack tracks, plus correction
uv tool install napari --with "itasc-tracking[solve]"

# Tracked labels to contacts and T1 events (HDF5, CSV)
uv tool install napari --with "itasc-aggregate"
```

The extras hold the heavy engines, so you can drop one when you do not need it:
`itasc-tracking` without `[solve]` browses and corrects existing tracks but
cannot run the Ultrack solver, and `itasc-cellpose` without `[cellpose]` keeps
the tracking helpers but not the segmentation model. Each distribution's guide,
linked from the [overview](../index.md#what-it-does), says which extras
its widgets need. Launch and check works the same: open napari, then open that
distribution's widget from the **Plugins** menu.

Each command sets up the single `napari` tool with that one distribution, so
installing a second one replaces the first. To have every stage available at
once, use the full app, `itasc[all]`.

### itasc-core is a library

`itasc-core` is the shared substrate, not an app: you import it from your own
Python rather than launch it, so it belongs in a project rather than in a tool.
In a uv project, add it as a dependency:

```bash
uv add itasc-core
```

The [core guide](core.md) covers what it exposes.

## Update and remove

ITASC is registered as a uv tool under the name `napari`. Update it, engines and
all, with:

```bash
uv tool upgrade napari
```

Remove it and its entire environment, leaving the rest of your system untouched,
with:

```bash
uv tool uninstall napari
```
