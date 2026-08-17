"""Artifact provenance helpers."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
