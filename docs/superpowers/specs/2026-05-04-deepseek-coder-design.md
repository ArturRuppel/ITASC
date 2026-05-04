# deepseek-coder: DeepSeek Coding Subagent

**Date:** 2026-05-04  
**Status:** Approved

## Overview

A standalone CLI tool (`deepseek-coder`) that Claude can invoke as a subagent for small, well-scoped coding tasks. Uses DeepSeek V4 Pro via native function calling. Companion Claude Code skill controls when and how Claude invokes it.

## CLI Interface

```
deepseek-coder "task description" /path/to/repo [more/paths...]
    --model MODEL          default: deepseek-v4-pro
    --max-iterations N     default: 20
    --max-file-bytes N     default: 200000
```

- One or more directory paths are required; these become the sandbox for all file tools
- API key loaded from `~/.config/cheap-scout/env` (DEEPSEEK_API_KEY), same as cheap-scout
- Progress and tool calls printed to stderr; final summary printed to stdout
- Exit 0 when agent calls `finish`; exit 1 on error or iteration limit reached

## Architecture

Single Python script at `~/.local/bin/deepseek-coder`. No external dependencies (stdlib only). Mirrors the cheap-scout structure: argument parsing → API key loading → agentic loop.

The agentic loop:
1. Build initial messages with system prompt + task + allowed roots
2. Call DeepSeek API with native `tools` array
3. If response contains tool calls, execute each, append results, continue
4. If response contains `finish` tool call, print summary to stdout and exit 0
5. If max iterations reached, print error to stderr and exit 1

## Tools

### File tools (sandboxed to allowed roots)

| Tool | Description |
|---|---|
| `list_files(path, limit=100)` | List files under a directory |
| `read_file(path)` | Read full file content |
| `read_lines(path, start, end)` | Read a line range (max 200 lines per call) |
| `search_text(pattern, path, limit=50)` | Literal text search across files |
| `write_file(path, content)` | Create or overwrite a file |
| `replace_in_file(path, old_string, new_string)` | Exact-match string replacement; errors if match count ≠ 1 |

All file paths resolved and checked against allowed roots before execution. Paths outside the sandbox return an error result (no exception — agent sees the error and can recover).

### Whitelisted shell commands

Executed via `subprocess.run` with args as a list (no shell interpolation). The command name (`argv[0]`) must exactly match the whitelist:

```
grep, rg, find, ls, cat, wc, head, tail, python3, pytest
```

For `python3`, `-c` must be the first argument (no `-m`, no script file paths). Working directory defaults to the first allowed root. stdout+stderr captured and returned to the agent. Timeout: 30 seconds.

### Completion

`finish(summary)` — ends the loop. Summary printed to stdout. Exit 0.

## Skill File

Located at `~/.claude/skills/deepseek-coder/SKILL.md`. Instructs Claude:

**When to use:**
- Small, well-scoped tasks: bug fixes, adding a function, refactoring a single module, writing a test
- Tasks where the approach is already decided and just needs execution
- When the task is clearly bounded to a known set of files

**When NOT to use:**
- Tasks requiring architectural judgment or spanning many files
- Security-sensitive changes
- Anything Claude should review the approach for before implementation

**Invocation pattern:**
```bash
deepseek-coder "task description" /path/to/relevant/dir
```

**After invocation:** Claude reads the modified files directly to verify changes before reporting success. The summary is a starting point, not ground truth.

## Error Handling

- File outside sandbox → error result returned to agent (recoverable)
- `replace_in_file` with no match or multiple matches → error result (agent must retry with different string)
- Shell command not on whitelist → error result
- Shell command timeout (30s) → error result
- API error → print to stderr, exit 1
- Max iterations → print "iteration limit reached" to stderr, exit 1

## Files Produced

| Path | Purpose |
|---|---|
| `~/.local/bin/deepseek-coder` | CLI script |
| `~/.claude/skills/deepseek-coder/SKILL.md` | Claude skill |
