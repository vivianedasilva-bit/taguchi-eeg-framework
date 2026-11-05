"""
run_framework.py
================

Runs the whole framework study in sequence, for every scenario and repetition:
    1. simulate_eeg_framework.py   synthetic multi-site data with known ground truth
    2. lead_taguchi_batch.py       L16 array (compute + summarize)
    3. framework_confirm.py        robust configuration, confirmation run, defaults
Every step is skipped if its output already exists, so the run can be stopped
and resumed at any time. Timing of every step goes to <root>/timing.csv.

All scripts must be in the same folder as this one.

Pilot (measure time before the full run)
    python run_framework.py --pilot
Full run
    python run_framework.py --root framework_runs --reps 20 --sites 5 --per-group 30
Smaller full run, if the pilot shows it is too slow
    python run_framework.py --root framework_runs --reps 10 --sites 5 --per-group 20
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
SCEN_INDEX = {"strong": 1, "weak": 2, "null": 3, "nonlinear_only": 4}


def run(cmd, log):
    t0 = time.time()
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=HERE)
    dt = time.time() - t0
    log.append(dt)
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("framework_runs"))
    ap.add_argument("--scenarios", nargs="*", default=["strong", "weak", "null", "nonlinear_only"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--per-group", type=int, default=30)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--outer", choices=["none", "L8"], default="none",
                    help="L8 = crossed design, 8 sites from an L8 outer array of noise factors")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pilot", action="store_true",
                    help="2 sites, 5 per group, 1 repetition, strong scenario; prints a time estimate")
    a = ap.parse_args()

    if a.outer == "L8":
        a.sites = 8
    full = dict(sites=a.sites, per_group=a.per_group, reps=a.reps, scen=len(a.scenarios))
    if a.pilot:
        a.root, a.scenarios, a.reps = Path("framework_pilot" if a.outer == "none" else "framework_pilot_L8"), ["strong"], 1
        a.sites, a.per_group = (8, 3) if a.outer == "L8" else (2, 5)

    a.root = (HERE / a.root) if not a.root.is_absolute() else a.root
    a.root.mkdir(parents=True, exist_ok=True)
    tfile = a.root / "timing.csv"
    new = not tfile.exists()
    with open(tfile, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["scenario", "rep", "step", "seconds"])
        for scen in a.scenarios:
            for rep in range(1, a.reps + 1):
                d = a.root / scen / f"rep{rep:02d}"
                sim, l16, conf = d / "sim", d / "l16", d / "confirm"
                seed = 1000 * SCEN_INDEX[scen] + rep
                log = []
                if not (sim / "ground_truth.csv").exists():
                    dt = run([PY, "simulate_eeg_framework.py", "--out", sim, "--scenario", scen,
                              "--sites", a.sites, "--per-group", a.per_group,
                              "--duration", a.duration, "--seed", seed, "--outer", a.outer], log)
                    w.writerow([scen, rep, "simulate", round(dt, 1)]); fh.flush()
                if not (l16 / "taguchi_table.csv").exists():
                    dt = run([PY, "lead_taguchi_batch.py", "compute", "--data", sim, "--out", l16,
                              "--array", "L16", "--labels", "HC", "AD", "--jobs", a.jobs], log)
                    run([PY, "lead_taguchi_batch.py", "summarize", "--out", l16, "--array", "L16"], log)
                    w.writerow([scen, rep, "l16", round(dt, 1)]); fh.flush()
                if not (conf / "confirmation.csv").exists():
                    dt = run([PY, "framework_confirm.py", "--sim", sim, "--l16", l16, "--out", conf,
                              "--jobs", a.jobs], log)
                    w.writerow([scen, rep, "confirm", round(dt, 1)]); fh.flush()
                print(f"== {scen} rep {rep} done ({sum(log)/60:.1f} min)")

    if a.pilot:
        import pandas as pd
        t = pd.read_csv(tfile)
        per_rep = t.seconds.sum()
        n_pilot = a.sites * a.per_group * 2
        n_full = full["sites"] * full["per_group"] * 2
        est = per_rep * (n_full / n_pilot) * full["reps"] * full["scen"]
        print("\n================ PILOT ================")
        print(f"Pilot ({a.sites} sites x {a.per_group} per group, 1 repetition): {per_rep/60:.1f} min")
        print(f"Estimate for the full run ({full['sites']} sites x {full['per_group']} per group, "
              f"{full['reps']} repetitions, {full['scen']} scenarios): {est/3600:.1f} hours "
              f"({est/86400:.1f} days) with {a.jobs} parallel jobs")
        print("If this is too long, reduce --reps or --per-group and run the estimate again.")


if __name__ == "__main__":
    main()
