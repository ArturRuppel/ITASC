# HPC Cellpose Widget New-Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone HPC Cellpose napari widget module and focused tests without modifying existing plugin files.

**Architecture:** Add `src/cellflow/napari/hpc_cellpose_widget.py` as a self-contained widget and command builder. Add `tests/napari/test_hpc_cellpose_widget.py` to lock down defaults, state, launch validation, runtime JSON generation, shell quoting, and the SSH authentication boundary. Main-widget integration is intentionally deferred because this pass may only write new files.

**Tech Stack:** Python, Qt via `qtpy`, pytest, existing `cellflow.napari.utils.launch_in_terminal`, JSON, `tempfile`, `shlex`.

---

### Task 1: New Widget Tests

**Files:**
- Create: `tests/napari/test_hpc_cellpose_widget.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `cellflow.napari.hpc_cellpose_widget.HpcCellposeWidget`. Include a lightweight fake viewer and `QApplication` setup, matching existing napari tests.

The tests must cover:

- default controls exist and use `/home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh`
- `refresh(pos_dir)` derives `<pos_dir>/0_input` and `<pos_dir>/1_cellpose`
- `get_state()` and `set_state()` round-trip controls
- missing inputs prevent terminal launch
- successful launch writes a temporary JSON config and calls `launch_in_terminal`
- command contains expected CLI arguments
- command and runtime JSON do not contain SSH-auth material such as `IdentityFile`, `SSH_AUTH_SOCK`, ` id_rsa`, ` id_ed25519`, `--identity`, or ` -i `

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest tests/napari/test_hpc_cellpose_widget.py -q
```

Expected: fail with `ModuleNotFoundError` or missing `HpcCellposeWidget`.

### Task 2: Standalone Widget Module

**Files:**
- Create: `src/cellflow/napari/hpc_cellpose_widget.py`
- Test: `tests/napari/test_hpc_cellpose_widget.py`

- [ ] **Step 1: Implement the minimal widget**

Create `HpcCellposeWidget(QWidget)` with these public methods:

- `refresh(pos_dir: Path | None) -> None`
- `get_state() -> dict`
- `set_state(state: dict) -> None`
- `build_runtime_config() -> dict`
- `build_command(config_path: Path) -> str`

The widget should include the controls from the approved spec, validate local paths in `_on_run_terminal`, write a temporary JSON config, build a fully shell-quoted command, and call `cellflow.napari.utils.launch_in_terminal`.

- [ ] **Step 2: Preserve the SSH auth boundary**

Do not add fields for identity files, keys, passwords, tokens, certificates, or agent sockets. Do not inspect `~/.ssh` or SSH environment variables. Do not add SSH options beyond `--remote-user` and `--remote-host`.

- [ ] **Step 3: Run tests and verify GREEN**

Run:

```bash
pytest tests/napari/test_hpc_cellpose_widget.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run syntax check**

Run:

```bash
python -m py_compile src/cellflow/napari/hpc_cellpose_widget.py tests/napari/test_hpc_cellpose_widget.py
```

Expected: command exits successfully.

### Task 3: New-File Scope Review

**Files:**
- Review only: `src/cellflow/napari/hpc_cellpose_widget.py`
- Review only: `tests/napari/test_hpc_cellpose_widget.py`

- [ ] **Step 1: Verify no existing files changed**

Run:

```bash
git diff --name-only
```

Expected: the implementation adds only:

```text
src/cellflow/napari/hpc_cellpose_widget.py
tests/napari/test_hpc_cellpose_widget.py
```

Existing dirty files may appear from pre-existing work, but this implementation must not modify them.

- [ ] **Step 2: Verify auth-sensitive strings are absent**

Run:

```bash
rg -n "IdentityFile|SSH_AUTH_SOCK|id_rsa|id_ed25519|password|token|certificate|--identity| -i " src/cellflow/napari/hpc_cellpose_widget.py tests/napari/test_hpc_cellpose_widget.py
```

Expected: matches only in test assertions that forbid those strings, not in widget behavior or command construction.
