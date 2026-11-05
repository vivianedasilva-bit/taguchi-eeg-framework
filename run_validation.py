"""
run_validation.py
=================

Independent validation of the framework study (manuscript section 2.6).

For each scenario and repetition chosen, a NEW dataset is simulated with the same
generative parameters and the same eight L8 sites but a different random seed
(new participants and new noise realisations). The robust configurations selected
on the ORIGINAL repetition (its l16/ folder) and the two default configurations are
then applied unchanged to the new data with framework_confirm.py. Nothing is
re-selected on the validation data.

    seed of the validation dataset = 1000 * scenario index + 500 + repetition
    (original datasets use 1000 * scenario index + repetition)

Run (resumable; finished steps are skipped)
    python run_validation.py --root framework_L8 --reps 1 2 --jobs 11
Summary table after the run
    python run_validation.py --root framework_L8 --summarize --out figures_validation

Outputs
    <root>/<scenario>/repNN/validation_sim/      simulated validation data
    <root>/<scenario>/repNN/validation/          confirmation.csv on the validation data
    <out>/table_validation.csv                   original vs validation, robust vs defaults
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
SCEN_INDEX = {"strong": 1, "weak": 2, "null": 3, "nonlinear_only": 4}


def run(cmd):
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=HERE)


def summarize(root: Path, out: Path):
    import pandas as pd
    rows = []
    for v in sorted(root.glob("*/rep*/validation/confirmation.csv")):
        scen, rep = v.parents[2].name, v.parents[1].name
        o = v.parents[1] / "confirm" / "confirmation.csv"
        if not o.exists():
            continue
        for kind, f in (("original", o), ("validation", v)):
            d = pd.read_csv(f)
            d["scenario"], d["rep"], d["data"] = scen, rep, kind
            rows.append(d)
    if not rows:
        raise SystemExit("No validation results found.")
    d = pd.concat(rows, ignore_index=True)
    t = (d.pivot_table(index=["scenario", "metric", "config"], columns="data",
                       values=["mean_abs_g", "SN_observed"], aggfunc="mean").round(3))
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "table_validation.csv")
    print(t.to_string())
    print(f"\nWritten {out / 'table_validation.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("framework_L8"))
    ap.add_argument("--scenarios", nargs="*", default=["strong", "weak", "null", "nonlinear_only"])
    ap.add_argument("--reps", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--per-group", type=int, default=30)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("figures_validation"))
    a = ap.parse_args()
    root = a.root if a.root.is_absolute() else HERE / a.root

    if a.summarize:
        summarize(root, a.out if a.out.is_absolute() else HERE / a.out)
        return

    for scen in a.scenarios:
        for rep in a.reps:
            d = root / scen / f"rep{rep:02d}"
            if not (d / "l16" / "main_effects.csv").exists():
                print(f"skip {scen} rep {rep}: original repetition not found")
                continue
            sim, val = d / "validation_sim", d / "validation"
            seed = 1000 * SCEN_INDEX[scen] + 500 + rep
            t0 = time.time()
            if not (sim / "ground_truth.csv").exists():
                run([PY, "simulate_eeg_framework.py", "--out", sim, "--scenario", scen,
                     "--per-group", a.per_group, "--duration", a.duration, "--seed", seed, "--outer", "L8"])
            if not (val / "confirmation.csv").exists():
                run([PY, "framework_confirm.py", "--sim", sim, "--l16", d / "l16", "--out", val,
                     "--jobs", a.jobs])
            print(f"== validation {scen} rep {rep} done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
