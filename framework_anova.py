"""
framework_anova.py
==================

Taguchi analysis of variance of the S/N ratio, prediction interval and
confirmation check, for every scenario x repetition x metric of the framework
study. Reads only the outputs of run_framework.py; changes nothing in them.

    python framework_anova.py --root framework_L8 --out figures_anova

For each L16 (global region) the 16 S/N values are decomposed over the 15
columns of the array (12 factors + 3 interaction columns, 1 df each; the
design is saturated, so the raw ANOVA has no error term).

Error term. The design is saturated within one repetition (15 columns, 15 df),
so the error is estimated from the replication of the whole experiment: the five
independent repetitions (new simulated subjects, same sites) are treated as
blocks. Model for one scenario and metric (80 S/N values):
    S/N = mean + repetition (4 df) + 15 columns (1 df each) + error (60 df)
No pooling is needed. F = SS / Ve is compared with F(0.05; 1, 60) = 4.00.
Percent contribution rho = (SS - Ve) / SS_T for significant columns.

Prediction (Taguchi additive model), one per repetition, for the robust
configuration chosen by framework_confirm.py in that repetition:
    eta_hat = T + sum over significant relevant factors (mean S/N at chosen level - T)
              + significant interaction columns whose two parent factors are in the
                configuration (level implied by the parents' levels)
with T and level means from that repetition. Confidence interval of the
confirmation (Taguchi 1987; Ross 1996):
    CI = sqrt( F(0.05; 1, ve) * Ve * (1/n_eff + 1/r) ),  n_eff = 16 / (1 + df used), r = 1
The confirmation succeeds when the observed S/N lies within eta_hat +/- CI.

Outputs
    anova_long.csv          one row per scenario x metric x column (SS, F, p, contribution %)
    anova_summary.csv       per scenario x metric: SS_T, Ve, error df, repetition % and error %
    confirmation_ci.csv     eta_hat, CI, observed S/N and success flag per run
    confirmation_ci_rate.csv  success rate by scenario and metric
    crossed_table_example.csv the complete crossed data table (16 x 8 g values) of one run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

ALPHA = 0.05
INTER = {"c03_INT_m_x_r": ("c01_m", "c02_r"),
         "c05_INT_m_x_tau": ("c01_m", "c04_tau"),
         "c15_INT_window_x_fs": ("c08_window", "c07_fs")}


def anova_replicated(d: pd.DataFrame, cols: list[str]):
    """d: 16 runs x R repetitions, columns cols, 'rep', 'SN_larger_better'."""
    y = d.SN_larger_better.to_numpy(float)
    N, G = len(y), y.mean()
    sst = float(((y - G) ** 2).sum())
    reps = d.rep.nunique()
    ss_rep = float(sum(16 * (dd.SN_larger_better.mean() - G) ** 2 for _, dd in d.groupby("rep")))
    ss = {c: N / 4.0 * (y[d[c] == 1].mean() - y[d[c] == 2].mean()) ** 2 for c in cols}
    df_e = N - 1 - (reps - 1) - len(cols)
    sse = sst - ss_rep - sum(ss.values())
    ve = sse / df_e if df_e > 0 else np.nan
    fcrit = fdist.ppf(1 - ALPHA, 1, df_e)
    rows = []
    for c in cols:
        F = ss[c] / ve if ve > 0 else np.nan
        sig = bool(np.isfinite(F) and F > fcrit)
        rows.append(dict(column=c, SS=ss[c], F=F, p=float(fdist.sf(F, 1, df_e)) if np.isfinite(F) else np.nan,
                         significant=sig, contribution_pct=100 * (ss[c] - ve) / sst if sig else 0.0))
    rep_pct = 100 * ss_rep / sst
    err_pct = 100 - sum(r["contribution_pct"] for r in rows) - rep_pct
    return dict(SS_T=sst, Ve=ve, df_e=df_e, F_crit=fcrit, rep_pct=rep_pct, error_pct=err_pct), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("framework_L8"))
    ap.add_argument("--out", type=Path, default=Path("figures_anova"))
    ap.add_argument("--example", default="strong/rep01:perm_entropy")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tabs = []
    for tf in sorted(a.root.glob("*/rep*/l16/taguchi_table.csv")):
        t = pd.read_csv(tf)
        t = t[t.region == "global"].copy()
        t["scenario"], t["rep"] = tf.parents[2].name, tf.parents[1].name
        tabs.append(t)
    tab = pd.concat(tabs, ignore_index=True)
    cols = [c for c in tab.columns if c[:1] == "c" and c[1:3].isdigit()]

    long, summ, conf_rows = [], [], []
    for (scen, metric), d in tab.groupby(["scenario", "metric"]):
        if d.SN_larger_better.isna().any() or len(d) % 16:
            continue
        info, rows = anova_replicated(d, cols)
        summ.append(dict(scenario=scen, metric=metric, **info))
        for r in rows:
            long.append(dict(scenario=scen, metric=metric, **r))
        sig = {r["column"] for r in rows if r["significant"]}
        for rep, t in d.groupby("rep"):
            cdir = a.root / scen / rep / "confirm"
            if not (cdir / "confirmation.csv").exists():
                continue
            rc = json.load(open(cdir / "robust_configs.json"))["robust"]
            conf = pd.read_csv(cdir / "confirmation.csv")
            if metric not in rc:
                continue
            levels = rc[metric]["levels"]
            y = t.SN_larger_better.to_numpy(float)
            T, eta, used = y.mean(), y.mean(), 0
            for c in cols:
                if c not in sig:
                    continue
                if c in INTER:
                    p1, p2 = INTER[c]
                    if not (p1 in levels and p2 in levels):
                        continue
                    lvl = 1 if levels[p1] == levels[p2] else 2
                elif c in levels:
                    lvl = levels[c]
                else:
                    continue                      # significant but not relevant: aliasing, not used
                eta += y[t[c].to_numpy() == lvl].mean() - T
                used += 1
            neff = 16 / (1 + used)
            ci = float(np.sqrt(fdist.ppf(1 - ALPHA, 1, info["df_e"]) * info["Ve"] * (1 / neff + 1)))
            obs = conf[(conf.config == "robust") & (conf.metric == metric)].SN_observed
            obs = float(obs.iloc[0]) if len(obs) else np.nan
            conf_rows.append(dict(scenario=scen, rep=rep, metric=metric, T_bar=T, eta_hat=eta, CI=ci,
                                  lower=eta - ci, upper=eta + ci, SN_observed=obs, n_terms=used,
                                  within_CI=bool(abs(obs - eta) <= ci) if np.isfinite(obs) else np.nan))

    pd.DataFrame(long).to_csv(a.out / "anova_long.csv", index=False)
    pd.DataFrame(summ).to_csv(a.out / "anova_summary.csv", index=False)
    cr = pd.DataFrame(conf_rows)
    cr.to_csv(a.out / "confirmation_ci.csv", index=False)
    rate = cr.groupby(["scenario", "metric"]).agg(within_CI=("within_CI", "mean"),
                                                 eta_hat=("eta_hat", "mean"), CI=("CI", "mean"),
                                                 SN_observed=("SN_observed", "mean")).round(2)
    rate.to_csv(a.out / "confirmation_ci_rate.csv")

    scen_rep, metric = a.example.split(":")
    tf = a.root / scen_rep / "l16" / "taguchi_table.csv"
    if tf.exists():
        ex = pd.read_csv(tf)
        ex = ex[(ex.region == "global") & (ex.metric == metric)]
        ex.to_csv(a.out / "crossed_table_example.csv", index=False)
    print(rate.to_string())
    print(f"\nWritten to {a.out}")


if __name__ == "__main__":
    main()
