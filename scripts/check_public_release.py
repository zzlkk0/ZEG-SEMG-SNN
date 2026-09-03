#!/usr/bin/env python3
"""Fail when common private or generated artifacts enter the public tree."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".h", ".hpp", ".html", ".ipynb", ".js",
    ".json", ".md", ".py", ".sh", ".sv", ".tcl", ".txt", ".xdc",
    ".yaml", ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".bit", ".ckpt", ".h5", ".hdf5", ".mat", ".npy", ".npz", ".onnx",
    ".p12", ".pem", ".pt", ".pth", ".xpr",
}
PATTERNS = {
    "GitHub token": re.compile(
        r"(?:gh" r"p_|github_" r"pat_)[A-Za-z0-9_]+"
    ),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "local home path": re.compile(r"/" r"home/[^/\s\"'`]+/"),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return [ROOT / name for name in output.splitlines() if name]


def main() -> None:
    errors: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"generated/private artifact: {relative}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"unexpected binary file: {relative}")
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {relative}")

    if errors:
        print("Public-release check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Public-release check passed ({len(tracked_files())} files checked).")


if __name__ == "__main__":
    main()
