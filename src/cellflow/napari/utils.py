"""UI utility functions for the CellFlow napari plugin."""
from __future__ import annotations

import platform
import subprocess


def launch_in_terminal(command: str) -> None:
    """Open a new OS terminal and run *command* inside it."""
    system = platform.system()
    if system == "Linux":
        # Using a more generic terminal if kitty is not available might be better,
        # but for now let's stick to what was there or try a few common ones.
        try:
            subprocess.Popen(["kitty", "--", "bash", "-c", f"{command}; exec bash"])
        except FileNotFoundError:
            try:
                subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"])
            except FileNotFoundError:
                subprocess.Popen(["xterm", "-e", f"bash -c '{command}; exec bash'"])
    elif system == "Darwin":
        escaped = command.replace("'", "'\\''")
        apple_script = f'tell application "Terminal" to do script "{escaped}"'
        subprocess.Popen(["osascript", "-e", apple_script])
    elif system == "Windows":
        subprocess.Popen(f'start cmd /k "{command}"', shell=True)
    else:
        raise RuntimeError(f"Unsupported platform '{system}'")
