import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_eval_produces_paired_estimates(tmp_path):
    # tiny dataset via the phase-1 runner machinery
    cfg = {"name": "mini", "kind": "ladder", "vary": "n_shots",
           "values": [2000],
           "base": json.loads(
               (REPO / "experiments/phase2/ramsey_eval_ladder.json").read_text()
           )["base"] | {"n_sweeps": 8}}
    cfg_path = tmp_path / "mini.json"
    cfg_path.write_text(json.dumps(cfg))
    r1 = subprocess.run([sys.executable, str(REPO / "scripts/run_phase1.py"),
                         str(cfg_path), "--out-dir", str(tmp_path)],
                        capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(
        [sys.executable, str(REPO / "scripts/eval_phase2.py"),
         str(tmp_path / "mini"), "--out-dir", str(tmp_path),
         "--estimators", "lsq,bayes"],
        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    art = json.loads((tmp_path / "estimates_n_shots_2000.json").read_text())
    assert len(art["records"]) == 8
    for rec in art["records"]:
        assert set(rec) >= {"lsq", "bayes"}   # paired: same record, all estimators
        assert "delta_hz" in rec["lsq"] and "delta_hz" in rec["bayes"]
    assert art["crb_sigma_delta_hz"] > 0
    assert len(art["git_sha"]) >= 7
