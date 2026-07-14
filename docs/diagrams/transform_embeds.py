#!/usr/bin/env python3
"""Expand each diagram {figure} into a light + dark pair (Furo only-light/only-dark)."""
import re

ROOT = "/home/aruppel/Projects/ITASC/docs"
FILES = [
    "explanation/index.md", "explanation/input-maps.md", "explanation/nucleus-tracking.md",
    "explanation/cell-segmentation.md", "explanation/contact-analysis.md",
    "manual/install.md", "manual/full-app.md", "manual/cellpose.md",
    "manual/tracking.md", "manual/aggregate.md", "manual/core.md",
]

# A diagram figure block: ```{figure} <path>/diagram-<name>.svg\n<body>\n```
BLOCK = re.compile(
    r"```\{figure\} (\.\./_static/diagrams/diagram-[a-z0-9-]+)\.svg\n(.*?\n)```",
    re.DOTALL,
)


def expand(m):
    base, body = m.group(1), m.group(2)
    if ":figclass:" in body:  # already expanded — leave alone
        return m.group(0)
    light_body = body.replace(":width: 100%", ":figclass: only-light\n:width: 100%", 1)
    dark_body = body.replace(":width: 100%", ":figclass: only-dark\n:width: 100%", 1)
    light = f"```{{figure}} {base}.svg\n{light_body}```"
    dark = f"```{{figure}} {base}-dark.svg\n{dark_body}```"
    return light + "\n" + dark


for rel in FILES:
    path = f"{ROOT}/{rel}"
    with open(path) as f:
        text = f.read()
    new, n = BLOCK.subn(expand, text)
    if n:
        with open(path, "w") as f:
            f.write(new)
    print(f"{rel}: {n} diagram figure(s) expanded")
