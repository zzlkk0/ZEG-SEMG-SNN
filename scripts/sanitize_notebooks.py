#!/usr/bin/env python3
"""Remove local notebook state and rewrite workstation-specific paths."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "docs" / "tutorial" / "notebooks"

REPLACEMENTS = {
    "semg_snn_90_loop": "training/semg_snn_90_loop",
    "semg_snn_fpga_reproduction": "training/semg_snn_fpga_reproduction",
    "semg_snn_nexys4ddr_vivado": "optional local FPGA implementation project",
}


def sanitize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        cell["source"] = [
            replace_local_paths(line) for line in cell.get("source", [])
        ]
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    metadata = notebook.setdefault("metadata", {})
    metadata.pop("widgets", None)
    kernelspec = metadata.get("kernelspec")
    if isinstance(kernelspec, dict):
        kernelspec["display_name"] = "Python 3"
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def replace_local_paths(line: str) -> str:
    local_project = re.compile(
        r"/" r"home/[^/]+/projects/([^/`\"\\]+)"
    )
    match = local_project.search(line)
    if match:
        project = match.group(1)
        replacement = REPLACEMENTS.get(project, project)
        line = local_project.sub(replacement, line)
    return line


def main() -> None:
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        sanitize(path)
        print(f"sanitized {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
