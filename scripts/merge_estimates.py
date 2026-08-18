#!/usr/bin/env python3
"""Merge per-estimator artifact files (same datasets, same record order) into
the main artifacts dir: records gain the extra estimators' keys in place.

Usage: merge_estimates.py MAIN_DIR EXTRA_DIR [EXTRA_DIR ...]"""
import json
import sys
from pathlib import Path


def main():
    main_dir = Path(sys.argv[1])
    for extra_dir in map(Path, sys.argv[2:]):
        for extra_path in sorted(extra_dir.glob("estimates_*.json")):
            main_path = main_dir / extra_path.name
            extra = json.loads(extra_path.read_text())
            if not main_path.exists():
                main_path.write_text(json.dumps(extra))
                print(f"copied {extra_path.name}")
                continue
            art = json.loads(main_path.read_text())
            assert art["dataset"] == extra["dataset"]
            assert len(art["records"]) == len(extra["records"]), extra_path
            for rec, extra_rec in zip(art["records"], extra["records"]):
                rec.update(extra_rec)
            main_path.write_text(json.dumps(art))
            print(f"merged {extra_path.name}")


if __name__ == "__main__":
    main()
