import subprocess
import sys
from pathlib import Path

from nvsim.experiment import load_dataset

REPO = Path(__file__).resolve().parents[1]


def test_ladder_runner_produces_one_dataset_per_rung(tmp_path):
    cfg = REPO / "experiments/phase1/ramsey_snr_ladder.json"
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_phase1.py"), str(cfg),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    rungs = sorted((tmp_path / "ramsey_snr_ladder").glob("*.npz"))
    assert len(rungs) == 4
    ds = load_dataset(rungs[0])
    assert ds["counts"].shape == (10, 150)
    assert len(ds["git_sha"]) >= 7
