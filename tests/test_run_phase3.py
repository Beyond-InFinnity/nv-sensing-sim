import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_runner_writes_per_schedule_artifacts(tmp_path):
    cfg = json.loads((REPO / "experiments/phase3/adaptive_vs_fixed.json").read_text())
    cfg["n_replicates"] = 2
    cfg["time_budget_s"] = 0.01
    cfg_path = tmp_path / "mini.json"
    cfg_path.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_phase3.py"),
                       str(cfg_path), "--out-dir", str(tmp_path)],
                      capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for kind in cfg["schedules"]:
        art = json.loads((tmp_path / f"{kind}.json").read_text())
        assert len(art["runs"]) == 2
        assert art["runs"][0]["true_delta_hz"] == art["true_deltas_hz"][0]
        assert len(art["git_sha"]) >= 7
