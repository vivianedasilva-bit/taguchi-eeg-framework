"""
make_figures.py
===============

Builds the figures and tables of the framework paper from the outputs of
run_framework.py (all scenarios and repetitions).

    python make_figures.py --root framework_runs --out figures

Outputs (PNG at 300 dpi + CSV tables)
  fig2_factor_importance_<scenario>.png   effect range of each factor on the S/N, per metric
  fig3_recovery.png                       mean |g| under robust vs default settings (strong, weak)
  fig4_null.png                           distribution of |g| in the null scenario
  table_confirmation.csv                  predicted vs observed S/N of the robust configurations
  table_false_positives.csv               proportion of configurations with |g| above threshold (null)
  table_stability.csv                     how often each factor level is selected across repetitions
  table_recovery.csv                      numbers behind fig3
  table_aliasing_diagnostic.csv           apparent effects of factors that cannot influence a metric
Metrics marked with * in the figures are directly linked to a generative
parameter that differs between groups (alpha frequency, theta gain, aperiodic
exponent, Hurst exponent of the alpha envelope, posterior coupling). In the
nonlinear_only scenario the groups differ only in phase coupling, so linear
measures are expected to show no effect.
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

TARGETED = {"iapf", "theta_alpha", "aperiodic_exp", "dfa_alpha", "wpli_alpha"}


def load(root: Path, name: str, sub: str):
    frames = []
    for f in root.glob(f"*/rep*/{sub}/{name}"):
        d = pd.read_csv(f)
        d["scenario"], d["rep"] = f.parents[2].name, f.parents[1].name
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def label(m):
    return f"{m}*" if m in TARGETED else m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("framework_runs"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--threshold", type=float, default=0.3, help="|g| above this counts as an effect")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    me = load(a.root, "main_effects.csv", "l16")
    tab = load(a.root, "taguchi_table.csv", "l16")
    conf = load(a.root, "confirmation.csv", "confirm")
    if me.empty:
        raise SystemExit("No results found. Run run_framework.py first.")
    me, tab = me[me.region == "global"], tab[tab.region == "global"]
    me = me[me.factor.str.match(r"c\d\d_") & ~me.factor.str.contains("_INT_")]

    # ---- Figure 2: factor importance (effect range of mean S/N), averaged over repetitions
    for scen, d in me.groupby("scenario"):
        rng = (d.groupby(["rep", "metric", "factor"]).mean_SN.agg(lambda s: s.max() - s.min())
                .groupby(["metric", "factor"]).mean().unstack("factor"))
        rng.to_csv(a.out / f"table_factor_importance_{scen}.csv")
        fig, ax = plt.subplots(figsize=(1 + 0.6 * rng.shape[1], 1 + 0.4 * rng.shape[0]))
        im = ax.imshow(rng.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(rng.shape[1]), [c.split("_", 1)[1] for c in rng.columns], rotation=60, ha="right")
        ax.set_yticks(range(rng.shape[0]), [label(m) for m in rng.index])
        fig.colorbar(im, ax=ax, label="effect range of S/N (dB)")
        ax.set_title(f"Factor importance, {scen} scenario")
        fig.tight_layout(); fig.savefig(a.out / f"fig2_factor_importance_{scen}.png", dpi=300); plt.close(fig)

    # ---- Aliasing diagnostic: apparent effects of factors that cannot influence a metric
    try:
        from framework_confirm import RELEVANT
        rows = []
        for scen, d in me.groupby("scenario"):
            r = d.groupby(["rep", "metric", "factor"]).mean_SN.agg(lambda s: s.max() - s.min()).reset_index()
            r["relevant"] = [f.split("_", 1)[1] in RELEVANT.get(m, set()) for m, f in zip(r.metric, r.factor)]
            agg = r.groupby(["metric", "relevant"]).mean_SN.mean().unstack("relevant")
            agg = agg.rename(columns={False: "mean_range_irrelevant", True: "mean_range_relevant"})
            agg["scenario"] = scen
            rows.append(agg.reset_index())
        pd.concat(rows).to_csv(a.out / "table_aliasing_diagnostic.csv", index=False)
    except ImportError:
        pass

    # ---- Figure 3: recovery under robust vs default settings
    if not conf.empty:
        rec = (conf[conf.scenario.isin(["strong", "weak", "nonlinear_only"])]
               .groupby(["scenario", "metric", "config"]).mean_abs_g.agg(["mean", "std"]).reset_index())
        rec.to_csv(a.out / "table_recovery.csv", index=False)
        scen_list = [s for s in ["strong", "weak", "nonlinear_only"] if s in rec.scenario.unique()]
        fig, axes = plt.subplots(1, len(scen_list), figsize=(7 * len(scen_list), 5), squeeze=False)
        for ax, scen in zip(axes[0], scen_list):
            d = rec[rec.scenario == scen]
            metrics = sorted(d.metric.unique())
            x = np.arange(len(metrics))
            for i, cfg in enumerate(["robust", "default_30s", "default_8s"]):
                dd = d[d.config == cfg].set_index("metric").reindex(metrics)
                ax.bar(x + (i - 1) * 0.27, dd["mean"], 0.27, yerr=dd["std"], label=cfg, capsize=2)
            ax.set_xticks(x, [label(m) for m in metrics], rotation=60, ha="right")
            ax.set_ylabel("mean |g| across sites"); ax.set_title(f"{scen} scenario"); ax.legend()
        fig.tight_layout(); fig.savefig(a.out / "fig3_recovery.png", dpi=300); plt.close(fig)

        # confirmation table
        rob = conf[conf.config == "robust"]
        ct = (rob.groupby(["scenario", "metric"])[["SN_predicted", "SN_observed"]]
              .agg(["mean", "std"]).round(2))
        ct.to_csv(a.out / "table_confirmation.csv")

    # ---- Figure 4 and false positives in the null scenario
    null = tab[tab.scenario == "null"]
    if not null.empty:
        fp = (null.assign(fp=null.mean_abs_g > a.threshold)
              .groupby("metric").fp.mean().rename("prop_L16_configs_above_threshold"))
        if not conf.empty and (conf.scenario == "null").any():
            rn = conf[(conf.scenario == "null") & (conf.config == "robust")]
            fp = pd.concat([fp, (rn.mean_abs_g > a.threshold).groupby(rn.metric).mean()
                            .rename("prop_robust_above_threshold")], axis=1)
        fp.to_csv(a.out / "table_false_positives.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null.mean_abs_g.dropna(), bins=40)
        ax.axvline(a.threshold, ls="--", color="k")
        ax.set_xlabel("mean |g| across sites"); ax.set_ylabel("count (configurations x metrics x reps)")
        ax.set_title("Null scenario (no true group difference)")
        fig.tight_layout(); fig.savefig(a.out / "fig4_null.png", dpi=300); plt.close(fig)

    # ---- Stability of the selected levels across repetitions
    rows = []
    for f in a.root.glob("*/rep*/confirm/robust_configs.json"):
        js = json.load(open(f))["robust"]
        for metric, r in js.items():
            for fc, lvl in r["levels"].items():
                rows.append({"scenario": f.parents[2].name, "rep": f.parents[1].name,
                             "metric": metric, "factor": fc, "level": lvl})
    if rows:
        st = pd.DataFrame(rows)
        stab = (st.groupby(["scenario", "metric", "factor"])["level"]
                  .agg(lambda s: s.value_counts(normalize=True).iloc[0]).rename("prop_modal_level")
                  .reset_index())
        stab.to_csv(a.out / "table_stability.csv", index=False)

    print(f"Figures and tables written to {a.out}")


if __name__ == "__main__":
    main()
