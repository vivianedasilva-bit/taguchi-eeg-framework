"""
framework_confirm.py
====================

Step 3 of the framework study, run after the L16 array on one simulated data set.

For every metric (global region) it
  1. reads the L16 main effects and selects the level of each factor with the
     highest mean S/N (the "robust configuration" of that metric);
  2. predicts the S/N of that configuration with the additive model, using the
     top-k factors with the largest effects (Taguchi confirmation logic);
  3. computes the metric under the robust configuration (confirmation run) and
     under two default settings common in the literature;
  4. writes confirmation.csv with observed S/N, mean |g|, log variance of |g|
     and the predicted S/N.

Default settings compared
  default_30s  m=2, r=0.20 SD, tau=1, 0.5-45 Hz, 200 Hz, 30 s segments for all measures
  default_8s   same, with 8 s segments for all measures
Levels are chosen only among the factors that can influence each metric (RELEVANT);
interaction columns are not used to choose levels (main effects only). This is
stated in the paper and checked by the agreement between predicted and observed S/N.

Usage
  python framework_confirm.py --sim run/sim --l16 run/l16 --out run/confirm --jobs 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import lead_taguchi_batch as T


# Factors that can influence each metric by construction. Levels are selected only
# among these; the other factors stay at their default values. An apparent effect
# of a factor that cannot influence a metric signals an aliased interaction
# (reported by make_figures.py as an aliasing diagnostic).
COMMON = {"band", "fs", "window", "detrend"}
RELEVANT = {
    "lzc": COMMON, "perm_entropy": COMMON | {"m", "tau"}, "higuchi_fd": COMMON | {"kmax"},
    "sampen": COMMON | {"m", "r", "tau"},
    "mse_ci": COMMON | {"m", "r", "tau", "mse_scale"}, "mse_fine": COMMON | {"m", "r", "tau", "mse_scale"},
    "mse_coarse": COMMON | {"m", "r", "tau", "mse_scale"},
    "dfa_alpha": COMMON | {"dfa_fit_min_s", "dfa_band"},
    "iapf": COMMON, "theta_alpha": COMMON, "aperiodic_exp": COMMON,
    "wpli_alpha": COMMON | {"wpli_epoch_s"}, "wpli_theta": COMMON | {"wpli_epoch_s"},
}


def factor_columns(cols):
    return [c for c in cols if c[:1] == "c" and c[1:3].isdigit() and "_INT_" not in c]


def robust_configs(l16_dir: Path, top_k: int):
    me = pd.read_csv(l16_dir / "main_effects.csv")
    tab = pd.read_csv(l16_dir / "taguchi_table.csv")
    me, tab = me[me.region == "global"], tab[tab.region == "global"]
    out = {}
    for metric, d in me.groupby("metric"):
        t_bar = tab[tab.metric == metric].SN_larger_better.mean()
        if not np.isfinite(t_bar):
            continue
        cfg, levels, effects = dict(T.DEFAULTS, window="long"), {}, []
        for fc in factor_columns(d.factor.unique()):
            dd = d[d.factor == fc].dropna(subset=["mean_SN"])
            if dd.empty:
                continue
            best = dd.loc[dd.mean_SN.idxmax()]
            lvl = int(best.level)
            fname = fc.split("_", 1)[1]
            if fname not in RELEVANT.get(metric, set(T.FACTORS2)):
                continue
            cfg[T.L16_TO_INTERNAL.get(fname, fname)] = T.FACTORS2[fname][lvl - 1]
            levels[fc] = lvl
            effects.append((dd.mean_SN.max() - dd.mean_SN.min(), best.mean_SN - t_bar, fc))
        effects.sort(reverse=True)
        predicted = t_bar + sum(gain for _, gain, _ in effects[:top_k])
        out[metric] = {"cfg": cfg, "levels": levels, "predicted_SN": float(predicted),
                       "top_factors": [fc for _, _, fc in effects[:top_k]]}
    return out


def config_key(cfg):
    keys = ["window", "A_segment_s", "B_m", "C_r", "E_tau", "G_fs", "D_band", "F_mse_scale",
            "kmax", "dfa_fit_min_s", "dfa_band", "detrend", "wpli_epoch_s"]
    return json.dumps([str(cfg.get(k)) for k in keys])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--l16", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", nargs=2, default=["HC", "AD"])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--budget-s", type=float, default=240.0)
    a = ap.parse_args()

    robust = robust_configs(a.l16, a.top_k)
    d30 = dict(T.DEFAULTS, window=None, A_segment_s=30)
    d8 = dict(T.DEFAULTS, window=None, A_segment_s=8)

    # unique configurations -> run ids
    configs, purpose = [], {}
    def add(cfg, tag):
        k = config_key(cfg)
        for c in configs:
            if c["key"] == k:
                purpose[c["run"]].append(tag)
                return c["run"]
        rid = len(configs) + 1
        configs.append(dict(cfg, run=rid, levels={}, key=k))
        purpose[rid] = [tag]
        return rid
    run_of = {"default_30s": add(d30, "default_30s"), "default_8s": add(d8, "default_8s")}
    for metric, r in robust.items():
        run_of[f"robust:{metric}"] = add(r["cfg"], f"robust:{metric}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "parts").mkdir(exist_ok=True)
    json.dump({"robust": {m: {k: v for k, v in r.items() if k != "cfg"} for m, r in robust.items()},
               "purpose": purpose}, open(a.out / "robust_configs.json", "w"), indent=1, default=str)

    files = sorted(str(f) for f in a.sim.glob("*/*.npz"))
    todo = [f for f in files if not (a.out / "parts" /
            f"{Path(f).parent.name}__{Path(f).stem}.csv").exists()]
    print(f"{len(configs)} configurations, {len(todo)} subjects to process")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(T.process_subject, f, configs, str(a.out), a.budget_s): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] elapsed {el/60:.1f} min, est. remaining {el/i*(len(todo)-i)/60:.1f} min")

    df = pd.concat((pd.read_csv(p) for p in (a.out / "parts").glob("*.csv")), ignore_index=True)
    df = df[df.region == "global"]
    sites = sorted(df.dataset.unique())
    g0, g1 = a.groups
    rows = []
    for tag, rid in run_of.items():
        kind, metric = (tag.split(":") + [None])[:2]
        metrics = [metric] if metric else sorted(robust)
        for mt in metrics:
            d = df[(df.run == rid) & (df.metric == mt)]
            gs = [T.hedges_g(d[(d.dataset == s) & (d.label == g0)].value,
                             d[(d.dataset == s) & (d.label == g1)].value) for s in sites]
            ag = np.abs(np.asarray(gs, float))
            ag = ag[np.isfinite(ag)]
            rows.append({"metric": mt, "config": kind, "run": rid,
                         "SN_observed": T.sn_larger_better(gs),
                         "SN_predicted": robust[mt]["predicted_SN"] if kind == "robust" else np.nan,
                         "mean_abs_g": float(ag.mean()) if len(ag) else np.nan,
                         "log_var_abs_g": float(np.log(ag.var(ddof=1))) if len(ag) >= 2 and ag.var(ddof=1) > 0 else np.nan,
                         **{f"g_{s}": g for s, g in zip(sites, gs)}})
    pd.DataFrame(rows).to_csv(a.out / "confirmation.csv", index=False)
    print(f"Written {a.out / 'confirmation.csv'}")


if __name__ == "__main__":
    main()
