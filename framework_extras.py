"""
framework_extras.py
===================

Complementary analyses of the framework paper, computed from the outputs that
run_framework.py already produced (no new EEG computation).

    python framework_extras.py --root framework_L8 --out figures_extras

A. Floor sensitivity
   The larger-the-better S/N floors |g| at 0.05. Here the S/N of every L16 run is
   recomputed with floors 0.01, 0.05 and 0.10 from the stored site-level g, the
   robust levels are re-selected, and the agreement with the 0.05 selection is
   reported.                                   -> table_floor_sensitivity.csv

B. Type I error in the null scenario
   For every metric, L16 run and site, Welch's t-test between groups on the
   subject-level values (features.csv). Under the null scenario about 5% of the
   tests should have p < 0.05. The same is done for the robust configurations
   (confirm/parts).                            -> table_type1_error.csv

C. Sensitivity map (crossed design, --outer L8 only)
   Control factors: effect range of the mean S/N (dB) per factor, as in fig2.
   Noise factors: change in mean |g| between level 2 and level 1 of each L8
   noise column, averaged over the 16 runs and the repetitions (negative = that
   noise source reduces group separation).
                    -> table_noise_sensitivity_<scenario>.csv, fig5_sensitivity_map_<scenario>.png

All scripts must be in the same folder (imports lead_taguchi_batch,
framework_confirm and simulate_eeg_framework).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import lead_taguchi_batch as T
from framework_confirm import RELEVANT
from simulate_eeg_framework import L8, NOISE_FACTORS

FLOORS = (0.01, 0.05, 0.10)
SKIP = ("n_segments", "mse_valid_scales")


def rep_dirs(root: Path):
    for scen in sorted(p for p in root.iterdir() if p.is_dir()):
        for rep in sorted(scen.glob("rep*")):
            if (rep / "l16" / "taguchi_table.csv").exists():
                yield scen.name, rep.name, rep


def factor_cols(cols):
    return [c for c in cols if c[:1] == "c" and c[1:3].isdigit() and "_INT_" not in c]


def select_levels(tab_metric: pd.DataFrame, metric: str, sn_col: str) -> dict:
    out = {}
    for fc in factor_cols(tab_metric.columns):
        if fc.split("_", 1)[1] not in RELEVANT.get(metric, set()):
            continue
        means = tab_metric.groupby(fc)[sn_col].mean()
        if means.notna().any():
            out[fc] = int(means.idxmax())
    return out


# ---------------------------------------------------------------- A. floor
def floor_sensitivity(root: Path) -> pd.DataFrame:
    rows = []
    for scen, rep, d in rep_dirs(root):
        tab = pd.read_csv(d / "l16" / "taguchi_table.csv")
        tab = tab[tab.region == "global"]
        gcols = [c for c in tab.columns if c.startswith("g_")]
        for f in FLOORS:
            tab[f"SN_{f}"] = [T.sn_larger_better(r, floor=f) for r in tab[gcols].to_numpy()]
        for metric, tm in tab.groupby("metric"):
            ref = select_levels(tm, metric, "SN_0.05")
            for f in FLOORS:
                sel = select_levels(tm, metric, f"SN_{f}")
                same = sum(sel.get(k) == v for k, v in ref.items())
                rows.append({"scenario": scen, "rep": rep, "metric": metric, "floor": f,
                             "n_factors": len(ref), "n_same_as_0.05": same,
                             "levels": json.dumps(sel)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B. type I
def site_pvalues(df: pd.DataFrame, groups=("HC", "AD")) -> pd.DataFrame:
    df = df[(df.region == "global") & ~df.metric.str.startswith(SKIP)]
    out = []
    for (run, metric, site), d in df.groupby(["run", "metric", "dataset"]):
        a = d.loc[d.label == groups[0], "value"].dropna()
        b = d.loc[d.label == groups[1], "value"].dropna()
        if len(a) > 2 and len(b) > 2:
            out.append({"run": run, "metric": metric, "site": site,
                        "p": ttest_ind(a, b, equal_var=False).pvalue})
    return pd.DataFrame(out)


def type1_error(root: Path) -> pd.DataFrame:
    l16, rob = [], []
    for scen, rep, d in rep_dirs(root):
        if scen != "null" or not (d / "l16" / "features.csv").exists():
            continue
        p = site_pvalues(pd.read_csv(d / "l16" / "features.csv"))
        l16.append(p.assign(rep=rep))
        cj, parts = d / "confirm" / "robust_configs.json", d / "confirm" / "parts"
        if cj.exists() and parts.exists():
            purpose = json.load(open(cj))["purpose"]
            run_of = {tag.split(":")[1]: int(rid) for rid, tags in purpose.items()
                      for tag in tags if tag.startswith("robust:")}
            conf = pd.concat((pd.read_csv(f) for f in parts.glob("*.csv")), ignore_index=True)
            pc = site_pvalues(conf)
            pc = pc[[run_of.get(m) == r for m, r in zip(pc.metric, pc.run)]]
            rob.append(pc.assign(rep=rep))
    if not l16:
        return pd.DataFrame()
    a = pd.concat(l16).groupby("metric").p.agg(
        prop_p05_L16=lambda s: (s < 0.05).mean(), n_tests_L16="size")
    if rob:
        b = pd.concat(rob).groupby("metric").p.agg(
            prop_p05_robust=lambda s: (s < 0.05).mean(), n_tests_robust="size")
        a = a.join(b)
    return a.reset_index()


# ---------------------------------------------------------------- C. map
def sensitivity_map(root: Path, out: Path):
    noise_names = [n for n, _, _ in NOISE_FACTORS]
    by_scen = {}
    for scen, rep, d in rep_dirs(root):
        tab = pd.read_csv(d / "l16" / "taguchi_table.csv")
        tab = tab[tab.region == "global"]
        sites = [f"L8R{i}" for i in range(1, 9)]
        if not all(f"g_{s}" in tab.columns for s in sites):
            continue                                   # not a crossed (L8) run
        me = pd.read_csv(d / "l16" / "main_effects.csv")
        me = me[(me.region == "global") & me.factor.isin(factor_cols(me.factor.unique()))]
        ctrl = me.groupby(["metric", "factor"]).mean_SN.agg(lambda s: s.max() - s.min()).unstack()
        absg = tab.groupby("metric")[[f"g_{s}" for s in sites]].apply(lambda x: x.abs().mean())
        noise = pd.DataFrame({
            name: [np.mean([r[k] for k in range(8) if L8[k][j] == 2]) -
                   np.mean([r[k] for k in range(8) if L8[k][j] == 1]) for r in absg.to_numpy()]
            for j, name in enumerate(noise_names)}, index=absg.index)
        by_scen.setdefault(scen, []).append((ctrl, noise))
    for scen, items in by_scen.items():
        ctrl = sum(c for c, _ in items) / len(items)
        noise = sum(n for _, n in items) / len(items)
        ctrl.columns = [c.split("_", 1)[1] for c in ctrl.columns]
        noise.round(3).to_csv(out / f"table_noise_sensitivity_{scen}.csv")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 0.42 * len(ctrl) + 2.2),
                                     gridspec_kw={"width_ratios": [ctrl.shape[1], noise.shape[1] + 1]})
        im1 = a1.imshow(ctrl.values, aspect="auto", cmap="Blues")
        a1.set_xticks(range(ctrl.shape[1]), ctrl.columns, rotation=60, ha="right")
        a1.set_yticks(range(len(ctrl)), ctrl.index)
        a1.set_title("Control factors: effect range of S/N (dB)")
        fig.colorbar(im1, ax=a1, fraction=0.04)
        lim = np.nanmax(np.abs(noise.values)) or 1
        im2 = a2.imshow(noise.values, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim)
        a2.set_xticks(range(noise.shape[1]), noise.columns, rotation=60, ha="right")
        a2.set_yticks(range(len(noise)), [""] * len(noise))
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                a2.text(j, i, f"{noise.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        a2.set_title("Noise factors (L8): change in mean |g|, level 2 - level 1")
        fig.colorbar(im2, ax=a2, fraction=0.04, label="red = noise reduces separation")
        fig.suptitle(f"Sensitivity map, {scen} scenario ({len(items)} repetitions)")
        fig.tight_layout()
        fig.savefig(out / f"fig5_sensitivity_map_{scen}.png", dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("framework_L8"))
    ap.add_argument("--out", type=Path, default=Path("figures_extras"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    fs = floor_sensitivity(a.root)
    fs.to_csv(a.out / "table_floor_sensitivity.csv", index=False)
    summ = (fs.assign(agree=fs["n_same_as_0.05"] / fs.n_factors.replace(0, np.nan))
              .groupby(["scenario", "floor"]).agree.mean().unstack().round(3))
    print("A. Agreement of selected levels with the 0.05 floor\n", summ, "\n")

    t1 = type1_error(a.root)
    if not t1.empty:
        t1.to_csv(a.out / "table_type1_error.csv", index=False)
        print("B. Type I error in the null scenario (expected ~0.05)\n", t1.round(3).to_string(index=False), "\n")

    sensitivity_map(a.root, a.out)
    print(f"C. Sensitivity maps written. All outputs in {a.out}")


if __name__ == "__main__":
    main()
