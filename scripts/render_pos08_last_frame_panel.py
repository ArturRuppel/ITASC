"""Save the last frame of pos08_channels.mp4 and pos08_analysis.mp4 side by side.

Output: pos08_last_frame_panel.png next to this script.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CHANNELS = HERE / "pos08_channels.mp4"
ANALYSIS = HERE / "pos08_analysis.mp4"
OUT = HERE / "pos08_last_frame_panel.png"

GAP = 8  # px between panels
BG = (255, 255, 255)


def last_frame(mp4: Path) -> np.ndarray:
    """Decode the final frame of `mp4` as an RGB uint8 array."""
    # Probe width/height first so we know the rawvideo shape.
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", str(mp4),
        ],
        capture_output=True, text=True, check=True,
    )
    W, H = (int(x) for x in probe.stdout.strip().split("x"))

    # -sseof -0.5 seeks ~half a second before EOF, then we keep the last frame.
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-sseof", "-1", "-i", str(mp4),
            "-update", "1", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(H, W, 3)


def main() -> None:
    a = last_frame(CHANNELS)
    b = last_frame(ANALYSIS)
    if a.shape[0] != b.shape[0]:
        raise RuntimeError(f"panel heights differ: {a.shape[0]} vs {b.shape[0]}")
    H = a.shape[0]
    panel = np.full((H, a.shape[1] + GAP + b.shape[1], 3), BG, dtype=np.uint8)
    panel[:, : a.shape[1]] = a
    panel[:, a.shape[1] + GAP :] = b
    Image.fromarray(panel).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
