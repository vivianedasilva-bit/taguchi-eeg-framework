"""
framework_window.py
===================

Minimum reliable window (manuscript section 2.7). Reads one simulated dataset and
its L16 results; changes nothing in them.

For every measure, the robust configuration selected on that dataset is kept fixed
and only the segment duration changes, on a grid of 2 to 128 s within the limits of
each family. For each duration, the first K consecutive non-overlapping segments
of every participant are analysed (K = min(4, floor(300 s / duration)), at least 2).

  Reliability   ICC(2,1), two-way random effects, absolute agreement, single
                measure (Shrout and Fleiss 1979), segments as raters, computed
                separately in each group (so that the group difference does not
                inflate it) and averaged.
  Robustness    participant value = mean over the K segments; Hedges' g per site;
                larger-the-better S/N (floor 0.05) and mean |g| across the 8 sites.
  Minimum reliable window
                shortest duration with ICC >= 0.75; the S/N at that duration and
                the duration of the best S/N are reported alongside.

Run (resumable; one file per participant in <out>/parts)
    python framework_window.py compute --run framework_L8/strong/rep01 --out framework_L8/strong/rep01/window --jobs 11
Summary, table and figure
    python framework_window.py summarize --out framework_L8/strong/rep01/window --fig figures_window
Quick test
    python framework_window.py compute --run framework_L8/strong/rep01 --out test_window --limit 1 --jobs 2
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

import lead_taguchi_batch as T
from framework_confirm import config_key, robust_configs

GRID = {"fast": [2, 4, 8, 16, 32, 64],
        "medium": [4, 8, 16, 32, 64, 128],
        "slow": [8, 16, 32, 64, 128]}
FAMILY_OF = {"lzc": "fast", "perm_entropy": "fast", "higuchi_fd": "fast", "sampen": "medium"}
MAX_SEG = 4
ICC_REF = 0.75


def family(metric):
    return FAMILY_OF.get(metric, "slow")


def plan(run_dir: Path):
    """Unique robust configurations -> list of measures that use each one."""
    rob = robust_configs(run_dir / "l16", top_k=4)
    groups = {}
    for metric, r in rob.items():
        k = config_key(r["cfg"])
        groups.setdefault(k, {"cfg": r["cfg"], "metrics": []})["metrics"].append(metric)
    return list(groups.values())


def metric_value(metric, seg, fs, band, cfg):
    """One measure on one segment (channels x samples), same definitions as
    lead_taguchi_batch._family_values, but only the measure requested."""
    m, r, tau = cfg["B_m"], cfg["C_r"], cfg["E_tau"]
    if metric.startswith("wpli"):
        b = (8, 13) if metric == "wpli_alpha" else (4, 8)
        return T.wpli_global(seg, fs, b, cfg["wpli_epoch_s"])
    vals = []
    for s in seg:
        if metric == "lzc":
            v = T.lempel_ziv(s)
        elif metric == "perm_entropy":
            v = T.permutation_entropy(s, m + 1, tau)
        elif metric == "higuchi_fd":
            v = T.higuchi_fd(s, cfg["kmax"])
        elif metric == "sampen":
            v = T.sample_entropy(s, m, r * np.std(s), tau)
        elif metric.startswith("mse"):
            v = T.multiscale_entropy(s, m, r, tau, cfg["F_mse_scale"])[metric]
        elif metric == "dfa_alpha":
            v = T.dfa_alpha_envelope(s, fs, cfg["dfa_fit_min_s"], cfg["dfa_band"])
        else:
            v = T.spectral_measures(s, fs, band, cfg["welch_s"], cfg["aper_range"])[metric]
        vals.append(v)
    return float(np.nanmean(vals))


def process_subject(npz_path: str, jobs: list[dict], out_dir: str) -> str:
    z = np.load(npz_path, allow_pickle=False)
    data, fs0 = z["data"].astype(np.float64), float(z["fs"])
    site, subject, label = str(z["dataset"]), str(z["subject"]), str(z["label"])
    rows = []
    for job in jobs:
        cfg, metrics = job["cfg"], job["metrics"]
        fs = cfg["G_fs"]
        x = resample_poly(data, int(fs), int(fs0), axis=1) if fs != fs0 else data
        band = (cfg["D_band"][0], min(cfg["D_band"][1], fs / 2 - 1))
        x = sosfiltfilt(butter(4, band, btype="band", fs=fs, output="sos"), x, axis=1)
        n = x.shape[1]
        for fam in sorted({family(m) for m in metrics}):
            mine = [m for m in metrics if family(m) == fam]
            for d in GRID[fam]:
                L = int(d * fs)
                K = min(MAX_SEG, n // L)
                if K < 2:
                    continue
                for k in range(K):
                    seg = x[:, k * L:(k + 1) * L]
                    if cfg["detrend"] == "linear":
                        seg = detrend(seg, axis=1)
                    for m in mine:
                        rows.append(dict(site=site, subject=subject, label=label, metric=m,
                                         duration=d, seg=k, value=metric_value(m, seg, fs, band, cfg)))
    out = Path(out_dir) / "parts" / f"{site}__{subject}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def icc_2_1(Y: np.ndarray) -> float:
    """ICC(2,1): Y is participants x segments, complete rows only."""
    Y = Y[np.isfinite(Y).all(axis=1)]
    n, k = Y.shape
    if n < 3 or k < 2:
        return np.nan
    g = Y.mean()
    msr = k * ((Y.mean(axis=1) - g) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(axis=0) - g) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + g) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def summarize(out_dir: Path, fig_dir: Path, groups=("HC", "AD")):
    parts = sorted((out_dir / "parts").glob("*.csv"))
    if not parts:
        raise SystemExit("No results in parts/.")
    df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    rows = []
    for (metric, d), dd in df.groupby(["metric", "duration"]):
        iccs = {}
        for gname in groups:
            Y = dd[dd.label == gname].pivot_table(index="subject", columns="seg", values="value").to_numpy()
            iccs[gname] = icc_2_1(Y)
        subj = dd.groupby(["site", "subject", "label"]).value.mean().reset_index()
        gs = [T.hedges_g(s[s.label == groups[0]].value, s[s.label == groups[1]].value)
              for _, s in subj.groupby("site")]
        ag = np.abs(np.asarray(gs, float))
        rows.append(dict(metric=metric, family=family(metric), duration=d, K=int(dd.seg.nunique()),
                         ICC_HC=iccs[groups[0]], ICC_AD=iccs[groups[1]],
                         ICC=np.nanmean([iccs[g] for g in groups]),
                         SN=T.sn_larger_better(gs), mean_abs_g=float(np.nanmean(ag)) if len(ag) else np.nan))
    tab = pd.DataFrame(rows).sort_values(["metric", "duration"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab.round(3).to_csv(fig_dir / "table_window.csv", index=False)

    mins = []
    for metric, t in tab.groupby("metric"):
        ok = t[t.ICC >= ICC_REF].sort_values("duration")
        r = ok.iloc[0] if len(ok) else None
        mins.append(dict(metric=metric, family=family(metric),
                         min_window_s=int(r.duration) if r is not None else np.nan,
                         ICC_at_min=float(r.ICC) if r is not None else np.nan,
                         SN_at_min=float(r.SN) if r is not None else np.nan,
                         best_SN=float(t.SN.max()),
                         duration_best_SN=int(t.loc[t.SN.idxmax(), "duration"]) if t.SN.notna().any() else np.nan))
    mins = pd.DataFrame(mins)
    mins.round(3).to_csv(fig_dir / "table_min_window.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = sorted(tab.metric.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m in metrics:
        t = tab[tab.metric == m]
        axes[0].plot(t.duration, t.ICC, marker="o", label=m)
        axes[1].plot(t.duration, t.SN, marker="o", label=m)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("segment duration (s)")
    axes[0].axhline(ICC_REF, ls="--", color="k")
    axes[0].set_ylabel("ICC(2,1), mean of the two groups")
    axes[1].set_ylabel("S/N across sites (dB)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_min_window.png", dpi=300)
    plt.close(fig)
    print(mins.to_string(index=False))
    print(f"\nWritten to {fig_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--run", type=Path, required=True, help="folder of one repetition (contains sim/ and l16/)")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    c.add_argument("--limit", type=int, default=None, help="participants per site and group (testing)")
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--fig", type=Path, default=Path("figures_window"))
    a = ap.parse_args()

    if a.cmd == "summarize":
        summarize(a.out, a.fig)
        return

    jobs = plan(a.run)
    (a.out / "parts").mkdir(parents=True, exist_ok=True)
    files = sorted((a.run / "sim").glob("*/*.npz"))
    if a.limit:
        keep, seen = [], {}
        for f in files:
            key = (f.parent.name, f.stem.rstrip("0123456789"))
            if seen.get(key, 0) < a.limit:
                keep.append(f)
                seen[key] = seen.get(key, 0) + 1
        files = keep
    todo = [str(f) for f in files if not (a.out / "parts" / f"{f.parent.name}__{f.stem}.csv").exists()]
    print(f"{len(jobs)} robust configurations, {len(todo)} participants to process")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(process_subject, f, jobs, str(a.out)): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] elapsed {el / 60:.1f} min, est. remaining {el / i * (len(todo) - i) / 60:.1f} min",
                  flush=True)
    print(f"Done. Next: python framework_window.py summarize --out {a.out}")


if __name__ == "__main__":
    main()
