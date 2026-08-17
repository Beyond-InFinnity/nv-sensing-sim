import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_runner_produces_artifact_with_provenance(tmp_path):
    cfg = REPO / "experiments/phase0/rabi.json"
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_phase0.py"), str(cfg),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    artifact = json.loads((tmp_path / "rabi.json").read_text())
    assert artifact["config"] == json.loads(cfg.read_text())
    assert isinstance(artifact["seed"], int)
    assert len(artifact["git_sha"]) >= 7
    assert "times_s" in artifact["results"]
    assert len(artifact["results"]["p0"]) == len(artifact["results"]["rabi_hz"])
