# HPC Cellpose Widget Design

## Goal

Add a small CellFlow napari widget section that launches the Maestro `cellpose_full` pipeline from the plugin. The widget should expose the parameters currently used by `/home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh`, derive sensible defaults from the open CellFlow project and position, and start the pipeline in an external terminal when the user presses a button.

The plugin must not copy, serialize, request, or pass any SSH authentication file. Authentication remains the responsibility of the user's existing SSH setup, such as `ssh-agent`, `~/.ssh/config`, and normal `ssh`/`rsync` behavior.

## Pipeline Target

The target script is:

```text
/home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh
```

This script already accepts these command-line options:

- `--input-dir`
- `--output-dir`
- `--nuclei-input`
- `--cells-input`
- `--config`
- `--max-concurrent-jobs`
- `--remote-host`
- `--remote-user`

The widget should call this script instead of duplicating upload, Slurm submission, polling, download, or cleanup logic in the plugin.

## User Experience

Add a new `HPC Cellpose` collapsible section near the existing Cellpose workflow. The recommended placement is immediately after the current `2. Cellpose` section in the main CellFlow widget, before nucleus segmentation and tracking.

When a project is open and position `P` is selected, the widget should default to:

- input directory: `<project>/posPP/0_input`
- output directory: `<project>/posPP/1_cellpose`
- nuclei input: `nucleus_3dt.tif`
- cells input: `cell_3dt.tif`
- config file: `/home/aruppel/Projects/HPC/cellpose_full/cellpose_full.json`
- remote user: `aruppel`
- remote host: `maestro.pasteur.fr`
- max concurrent jobs: `4`

The section should show a compact input status line for the expected local nuclei and cell input files. The launch button should be disabled or report a clear status message when no project is open, the pipeline script is missing, the config file is missing, or the selected input files do not exist.

## Controls

Expose the pipeline controls that map directly to script arguments or config values:

- `Input dir`: directory field with browse button.
- `Output dir`: directory field with browse button.
- `Config`: file field with browse button.
- `Nuclei input`: text field.
- `Cells input`: text field.
- `Frames`: text field accepting `all`, a single zero-based frame index, or a comma-separated list.
- `Nuclei 3D`: checkbox mapped to `nuclei.do_3d`.
- `Nuclei anisotropy`: decimal spin box.
- `Nuclei diameter`: integer spin box.
- `Nuclei size`: integer spin box.
- `Nuclei gamma`: decimal spin box.
- `Cells size`: integer spin box.
- `Cells gamma`: decimal spin box.
- `Max concurrent jobs`: positive integer spin box.
- `Remote user`: text field.
- `Remote host`: text field.
- `Run in Terminal`: launches the pipeline command.

The first implementation should avoid adding Slurm resource controls because `cellpose_full.sbatch` owns partition, memory, CPU, GPU, and walltime settings. Those can be added later if the HPC script gains corresponding CLI support.

## Data Flow

On launch:

1. Read the current widget controls.
2. Validate that the pipeline script, local input directory, nuclei input, cells input, and config path exist.
3. Create a temporary runtime JSON config containing:
   - `input_dir`
   - `frames`
   - `nuclei.input`
   - `nuclei.do_3d`
   - `nuclei.anisotropy`
   - `nuclei.diameter`
   - `nuclei.size`
   - `nuclei.gamma`
   - `cells.input`
   - `cells.size`
   - `cells.gamma`
4. Build a shell command with `shlex.quote` for every path and user-controlled value:

```text
bash /home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh \
  --input-dir <input-dir> \
  --output-dir <output-dir> \
  --config <temporary-config> \
  --nuclei-input <nuclei-input> \
  --cells-input <cells-input> \
  --max-concurrent-jobs <n> \
  --remote-user <user> \
  --remote-host <host>
```

5. Launch that command through the existing `cellflow.napari.utils.launch_in_terminal` helper.
6. Update the widget status to say the command was launched, or copy the command to the clipboard if terminal launch fails.

The temporary config should be created with `tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="cellflow_hpc_cellpose_")`. It should contain only pipeline configuration data. It must not contain secrets, SSH settings, key paths, or authentication material.

## State

Include the HPC Cellpose widget state in `CellFlowMainWidget.get_state()` and `set_state()` under a new top-level key, for example:

```json
{
  "hpc_cellpose": {
    "config_path": "...",
    "nuclei_input": "nucleus_3dt.tif",
    "cells_input": "cell_3dt.tif",
    "frames": "all",
    "nuclei_do_3d": false,
    "nuclei_anisotropy": 1.5,
    "nuclei_diameter": 25,
    "nuclei_size": 0,
    "nuclei_gamma": 1.0,
    "cells_size": 0,
    "cells_gamma": 1.0,
    "max_concurrent_jobs": 4,
    "remote_user": "aruppel",
    "remote_host": "maestro.pasteur.fr"
  }
}
```

`input_dir` and `output_dir` can either be persisted explicitly or re-derived from the selected project and position. Prefer re-deriving them by default so switching `P` follows the normal CellFlow project layout. If the user manually changes either path, persist that override.

## SSH Authentication Boundary

The widget must observe these constraints:

- Do not add a control for an SSH key, identity file, password, token, certificate, or agent socket.
- Do not read from `~/.ssh`, environment variables such as `SSH_AUTH_SOCK`, or any private key path.
- Do not write any SSH authentication information to CellFlow config, temporary JSON, temporary scripts, logs, status labels, or clipboard fallback text.
- Do not pass `-i`, `IdentityFile`, `ProxyCommand`, `SSH_AUTH_SOCK`, or similar authentication options to `ssh` or `rsync`.
- Do not copy any authentication file to the remote scratch directory.

The command may pass only the non-secret connection target values already supported by the script: `--remote-user` and `--remote-host`.

## Error Handling

Use status text in the widget for local preflight errors:

- no project open
- pipeline script missing
- config file missing
- input directory missing
- nuclei input missing
- cells input missing
- invalid max concurrent jobs
- invalid numeric parameter value
- terminal launch failure

Once the terminal is launched, the plugin should not try to supervise the HPC job. Upload, Slurm submission, polling, output validation, and remote cleanup remain owned by `run_pipeline.sh`.

## Testing

Add focused napari widget tests for:

- the main widget exposes an `HPC Cellpose` section after `2. Cellpose`
- default paths derive from selected project and position
- `get_state()` and `set_state()` round-trip the HPC controls
- launch validation rejects missing local inputs before opening a terminal
- command construction includes the expected script path and CLI arguments
- runtime JSON contains the exposed Cellpose config values
- runtime JSON does not contain SSH auth-related keys or paths
- generated command does not contain `-i`, `IdentityFile`, `SSH_AUTH_SOCK`, private key paths, or copied auth material
- terminal launch uses `launch_in_terminal` and falls back to clipboard/status text on failure

## Out of Scope

- Editing `/home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh`.
- Copying the HPC pipeline into the CellFlow repository.
- Managing SSH credentials or testing remote authentication.
- Monitoring Slurm jobs inside napari after terminal launch.
- Exposing Slurm resource controls before the HPC script supports them as CLI parameters.
