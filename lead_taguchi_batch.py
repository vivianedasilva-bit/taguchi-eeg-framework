"""
lead_taguchi_batch.py
=====================

Stage 1 of the study (Dasilva, Pino, Borges Silva). Runs every condition of a
Taguchi orthogonal array on every subject and produces the table used for the
signal-to-noise analysis.

Input   the continuous recordings produced by lead_continuous.py
        (continuous/<dataset>/<subject>.npz)
Output  results/parts/<dataset>__<subject>.csv   one file per subject (resumable)
        results/features.csv                     all subjects, long format
        results/taguchi_table.csv                one row per run x metric x region,
                                                 with Hedges' g per site, S/N and the
                                                 dual-response columns (mean |g|,
                                                 log variance of |g| across sites)
        results/main_effects.csv                 mean S/N per factor level

Usage
-----
# 1) compute (run overnight; safe to stop and restart, finished subjects are skipped)
python lead_taguchi_batch.py compute --data continuous --out results --array L18 --jobs 8 \
       --datasets ADFTD APAVA ADSZ ADFSU --labels HC AD

# 2) summarise for the Taguchi analysis
python lead_taguchi_batch.py summarize --out results --groups HC AD

# quick test on 2 subjects per dataset
python lead_taguchi_batch.py compute --data continuous --out test --array L9 --limit 2

Design notes
------------
* Only analysis parameters vary across runs. Cleaning (filtering, ICA, artefact
  rejection) is fixed in lead_continuous.py, so the costly preprocessing runs once.
* Arrays. L9 uses factors A-D. L18 uses H (2 levels) and A-G (3 levels).
  L16 is a two-level design with 12 factors and 3 interaction columns
  (m x r, m x tau, window x fs), column assignment provisional (L16_ASSIGN).
  The window factor uses sliding levels per metric family (WINDOWS_S).
  The spectrum window is fixed at 4 s for better alpha-peak resolution.
  The levels can be edited in FACTORS below, after discussion with the co-authors.
* Metrics are written in plain numpy/scipy (independent research implementation,
  no proprietary code). RQA is not included in this version because of its
  computational cost; it will be added as windowed RQA in a later version.
* The aperiodic exponent is a simple log-log fit outside the alpha band. It can be
  replaced by specparam (FOOOF) for the final analysis.
"""

from __future__ import annotations

import argparse
import warnings
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=RuntimeWarning)

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # numpy 1.x and 2.x

# --------------------------------------------------------------------------
# Taguchi design
# --------------------------------------------------------------------------

FACTORS = {
    "A_segment_s": [10, 30, 60],
    "B_m":         [2, 3, 4],
    "C_r":         [0.15, 0.20, 0.25],
    "D_band":      [(0.5, 45.0), (1.0, 40.0), (1.0, 30.0)],
    "E_tau":       [1, 2, 3],
    "F_mse_scale": [10, 15, 20],
    "G_fs":        [100, 128, 200],
    "H_seg_agg":   ["mean", "median"],          # 2-level factor (L18 only)
}

L9 = np.array([
    [1, 1, 1, 1], [1, 2, 2, 2], [1, 3, 3, 3],
    [2, 1, 2, 3], [2, 2, 3, 1], [2, 3, 1, 2],
    [3, 1, 3, 2], [3, 2, 1, 3], [3, 3, 2, 1]])
L9_COLS = ["A_segment_s", "B_m", "C_r", "D_band"]

L18 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 2, 2, 2, 2], [1, 1, 3, 3, 3, 3, 3, 3],
    [1, 2, 1, 1, 2, 2, 3, 3], [1, 2, 2, 2, 3, 3, 1, 1], [1, 2, 3, 3, 1, 1, 2, 2],
    [1, 3, 1, 2, 1, 3, 2, 3], [1, 3, 2, 3, 2, 1, 3, 1], [1, 3, 3, 1, 3, 2, 1, 2],
    [2, 1, 1, 3, 3, 2, 2, 1], [2, 1, 2, 1, 1, 3, 3, 2], [2, 1, 3, 2, 2, 1, 1, 3],
    [2, 2, 1, 2, 3, 1, 3, 2], [2, 2, 2, 3, 1, 2, 1, 3], [2, 2, 3, 1, 2, 3, 2, 1],
    [2, 3, 1, 3, 2, 3, 1, 2], [2, 3, 2, 1, 3, 1, 2, 3], [2, 3, 3, 2, 1, 2, 3, 1]])
L18_COLS = ["H_seg_agg", "A_segment_s", "B_m", "C_r", "D_band",
            "E_tau", "F_mse_scale", "G_fs"]

# values used for factors that are not in the chosen array
# ---- L16 screening design (agreed Oct 2026, column assignment provisional) ----
# 12 two-level factors + 3 interaction columns. Factors were chosen for the
# HC / MCI / AD question: slowing, loss of complexity and disconnection.
FACTORS2 = {
    "window":        ["short", "long"],          # sliding levels, see WINDOWS_S
    "m":             [2, 3],
    "r":             [0.15, 0.25],
    "tau":           [1, 2],
    "fs":            [128, 200],
    "band":          [(0.5, 45.0), (1.0, 30.0)],
    "mse_scale":     [10, 20],
    "kmax":          [8, 16],
    "dfa_fit_min_s": [1.0, 2.0],
    "dfa_band":      [(8.0, 13.0), (7.0, 13.0)],
    "detrend":       ["none", "linear"],
    "wpli_epoch_s":  [2.0, 4.0],
}

# Sliding levels for the window factor (Taguchi sliding-level technique):
# "short" and "long" mean different durations for each metric family.
WINDOWS_S = {
    "fast":   {"short": 2,  "long": 10},    # Lempel-Ziv, permutation entropy, Higuchi
    "medium": {"short": 5,  "long": 30},    # sample entropy
    "slow":   {"short": 30, "long": 120},   # multiscale entropy, DFA, spectrum, wPLI
}

# Provisional column assignment on the classic L16 linear graph.
# Interactions: m x r -> col 3, m x tau -> col 5, window x fs -> col 15.
# To be confirmed or changed by Prof. Messias Borges Silva.
L16_ASSIGN = {1: "m", 2: "r", 3: "INT_m_x_r", 4: "tau", 5: "INT_m_x_tau",
              6: "band", 7: "fs", 8: "window", 9: "mse_scale", 10: "kmax",
              11: "dfa_fit_min_s", 12: "dfa_band", 13: "detrend", 14: "wpli_epoch_s",
              15: "INT_window_x_fs"}


def make_L16() -> np.ndarray:
    """Standard Taguchi L16(2^15), classic column order
    a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd
    (column 3 is the interaction of columns 1 and 2, and so on)."""
    sets = ["a", "b", "ab", "c", "ac", "bc", "abc", "d", "ad", "bd", "abd",
            "cd", "acd", "bcd", "abcd"]
    rows = []
    for i in range(16):
        bit = {"a": (i >> 3) & 1, "b": (i >> 2) & 1, "c": (i >> 1) & 1, "d": i & 1}
        rows.append([1 + (sum(bit[ch] for ch in st) % 2) for st in sets])
    return np.array(rows)


L16 = make_L16()

DEFAULTS = {"A_segment_s": 30, "B_m": 2, "C_r": 0.20, "D_band": (0.5, 45.0),
            "E_tau": 1, "F_mse_scale": 15, "G_fs": 200, "H_seg_agg": "mean",
            "kmax": 10, "dfa_fit_min_s": 1.0, "dfa_band": (8.0, 13.0),
            "wpli_epoch_s": 2.0, "welch_s": 4.0, "aper_range": (2.0, 40.0),
            "detrend": "none", "window": None}
# L16 factor name -> internal parameter name
L16_TO_INTERNAL = {"m": "B_m", "r": "C_r", "tau": "E_tau", "fs": "G_fs",
                   "band": "D_band", "mse_scale": "F_mse_scale"}


def build_runs(name: str) -> list[dict]:
    runs = []
    if name == "L16":
        for i, row in enumerate(L16, start=1):
            cfg = dict(DEFAULTS)
            levels = {}
            for col, fname in L16_ASSIGN.items():
                lvl = int(row[col - 1])
                levels[f"c{col:02d}_{fname}"] = lvl
                if fname.startswith("INT_"):
                    continue
                cfg[L16_TO_INTERNAL.get(fname, fname)] = FACTORS2[fname][lvl - 1]
            cfg["run"], cfg["levels"] = i, levels
            runs.append(cfg)
        return runs
    arr, cols = {"L9": (L9, L9_COLS), "L18": (L18, L18_COLS)}[name]
    for i, row in enumerate(arr, start=1):
        cfg = dict(DEFAULTS)
        for col, lvl in zip(cols, row):
            cfg[col] = FACTORS[col][lvl - 1]
        cfg["run"] = i
        cfg["levels"] = {c: int(l) for c, l in zip(cols, row)}
        runs.append(cfg)
    return runs


def check_orthogonal(arr: np.ndarray) -> bool:
    """Every pair of columns must contain each level combination equally often."""
    for i in range(arr.shape[1]):
        for j in range(i + 1, arr.shape[1]):
            _, counts = np.unique(arr[:, [i, j]], axis=0, return_counts=True)
            if len(set(counts)) != 1:
                return False
    return True


# --------------------------------------------------------------------------
# Channels and regions
# --------------------------------------------------------------------------

REGIONS = {
    "frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central":   ["C3", "Cz", "C4"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "parietal":  ["P3", "Pz", "P4"],
    "occipital": ["O1", "O2"],
}

# --------------------------------------------------------------------------
# Metrics (single channel, 1-D numpy array)
# --------------------------------------------------------------------------


def _embed(x, dim, tau, n):
    return np.stack([x[i * tau: i * tau + n] for i in range(dim)], axis=1)


def _sample_entropy_kdtree(x, m, r_abs, tau=1):
    n = len(x) - m * tau
    if n < 50:
        return np.nan
    em, em1 = _embed(x, m, tau, n), _embed(x, m + 1, tau, n)
    B = (cKDTree(em).count_neighbors(cKDTree(em), r_abs, p=np.inf) - n) / 2
    A = (cKDTree(em1).count_neighbors(cKDTree(em1), r_abs, p=np.inf) - n) / 2
    return -log(A / B) if A > 0 and B > 0 else np.nan


# Optional acceleration (same definition, identical results): templates are
# sorted by their first coordinate and only candidates within r on that
# coordinate are compared (sort-and-sweep). Used when numba is installed;
# otherwise the cKDTree version above is used.
try:
    from numba import njit

    @njit(cache=True)
    def _sampen_counts(x, m, r, tau, n):
        order = np.argsort(x[:n])
        A = 0
        B = 0
        for a in range(n):
            i = order[a]
            xi = x[i]
            for b in range(a + 1, n):
                j = order[b]
                if x[j] - xi > r:
                    break
                ok = True
                for k in range(1, m):
                    if abs(x[i + k * tau] - x[j + k * tau]) > r:
                        ok = False
                        break
                if ok:
                    B += 1
                    if abs(x[i + m * tau] - x[j + m * tau]) <= r:
                        A += 1
        return A, B

    def sample_entropy(x, m, r_abs, tau=1):
        x = np.ascontiguousarray(x, dtype=np.float64)
        n = len(x) - m * tau
        if n < 50:
            return np.nan
        A, B = _sampen_counts(x, m, float(r_abs), tau, n)
        return -log(A / B) if A > 0 and B > 0 else np.nan
except ImportError:
    sample_entropy = _sample_entropy_kdtree


def multiscale_entropy(x, m, r_rel, tau, max_scale):
    r_abs = r_rel * np.std(x)          # tolerance fixed at scale 1 (Costa et al.)
    vals = []
    for s in range(1, max_scale + 1):
        L = len(x) // s
        cg = x[:L * s].reshape(L, s).mean(axis=1)
        vals.append(sample_entropy(cg, m, r_abs, tau))
    v = np.array(vals)
    half = max_scale // 2
    return {"mse_ci": np.nanmean(v) if np.isfinite(v).any() else np.nan,
            "mse_fine": np.nanmean(v[:5]) if np.isfinite(v[:5]).any() else np.nan,
            "mse_coarse": np.nanmean(v[half:]) if np.isfinite(v[half:]).any() else np.nan,
            "mse_valid_scales": int(np.isfinite(v).sum())}


def permutation_entropy(x, order, tau=1):
    n = len(x) - (order - 1) * tau
    if n < 10:
        return np.nan
    patt = np.argsort(_embed(x, order, tau, n), axis=1)
    _, counts = np.unique(patt, axis=0, return_counts=True)
    p = counts / counts.sum()
    from math import factorial
    return float(-(p * np.log2(p)).sum() / np.log2(factorial(order)))


def lempel_ziv(x):
    """LZ76 complexity of the median-binarised signal, normalised by n / log2(n).
    Exhaustive-history parsing (Lempel & Ziv, 1976), using C-level substring search."""
    s = (x > np.median(x)).astype(np.uint8).tobytes()
    n = len(s)
    i, c = 0, 0
    while i < n:
        l = 1
        while i + l <= n and s[i:i + l] in s[0:i + l - 1]:
            l += 1
        c += 1
        i += l
    return c * np.log2(n) / n


def higuchi_fd(x, kmax=10):
    N = len(x)
    lk, kk = [], []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lm.append(np.abs(np.diff(x[idx])).sum() * (N - 1) / ((len(idx) - 1) * k) / k)
        if Lm:
            lk.append(np.log(np.mean(Lm)))
            kk.append(np.log(1.0 / k))
    return float(np.polyfit(kk, lk, 1)[0])


def dfa_alpha_envelope(x, fs, fit_min_s=1.0, band=(8.0, 13.0)):
    """DFA exponent of the alpha amplitude envelope, windows from fit_min_s to N/10."""
    if band[1] >= fs / 2:
        return np.nan
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    y = np.cumsum(env - env.mean())
    n_min, n_max = int(fit_min_s * fs), len(y) // 10
    if n_max <= n_min * 2:
        return np.nan
    scales = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), 10).astype(int))
    F = []
    for n in scales:
        k = len(y) // n
        seg = y[:k * n].reshape(k, n)
        t = np.arange(n)
        coef = np.polyfit(t, seg.T, 1)
        trend = np.outer(coef[0], t) + coef[1][:, None]
        F.append(np.sqrt(np.mean((seg - trend) ** 2)))
    return float(np.polyfit(np.log(scales), np.log(F), 1)[0])


def spectral_measures(x, fs, band, welch_s=2.0, aper_range=(2.0, 40.0)):
    f, P = welch(x, fs=fs, nperseg=min(len(x), int(welch_s * fs)))
    def bp(lo, hi):
        sel = (f >= lo) & (f < hi)
        return _trapz(P[sel], f[sel]) if sel.any() else np.nan
    alpha_sel = (f >= 7) & (f <= 13)
    iapf = f[alpha_sel][np.argmax(P[alpha_sel])] if alpha_sel.any() else np.nan
    hi = min(aper_range[1], band[1] - 2)
    fit = (f >= aper_range[0]) & (f <= hi) & ~((f >= 7) & (f <= 14))
    expo = -np.polyfit(np.log10(f[fit]), np.log10(P[fit]), 1)[0] if fit.sum() > 5 else np.nan
    return {"iapf": iapf, "theta_alpha": bp(4, 8) / bp(8, 13), "aperiodic_exp": expo}


def wpli_global(seg, fs, band, epoch_s=2.0):
    """Mean weighted phase lag index over all channel pairs (Vinck et al., 2011)."""
    n = int(epoch_s * fs)
    k = seg.shape[1] // n
    if k < 2:
        return np.nan
    ep = seg[:, :k * n].reshape(seg.shape[0], k, n) * np.hanning(n)
    X = np.fft.rfft(ep, axis=2)
    f = np.fft.rfftfreq(n, 1 / fs)
    sel = (f >= band[0]) & (f < band[1])
    X = X[:, :, sel]
    iu = np.triu_indices(seg.shape[0], 1)
    im = np.imag(X[iu[0]] * np.conj(X[iu[1]]))          # pairs x epochs x freqs
    num = np.abs(im.mean(axis=1))
    den = np.abs(im).mean(axis=1)
    w = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(w))


# --------------------------------------------------------------------------
# One subject, all runs
# --------------------------------------------------------------------------


def segments_from_runs(data, runs, L, max_segments):
    out = []
    for a, b in runs:
        for s in range(a, b - L + 1, L):
            out.append(data[:, s:s + L])
            if len(out) >= max_segments:
                return out
    return out


FAMILY_METRICS = {"fast": "Lempel-Ziv, permutation entropy, Higuchi",
                  "medium": "sample entropy",
                  "slow": "multiscale entropy, DFA, spectrum, wPLI"}


def _window_s(cfg, family):
    if cfg.get("window"):                       # L16, sliding levels
        return WINDOWS_S[family][cfg["window"]]
    return cfg["A_segment_s"]                    # L9 / L18, one window for all


def _family_values(family, seg, fs, band, cfg):
    """Metrics of one family for one segment -> dict metric -> array(channels)."""
    m, r, tau = cfg["B_m"], cfg["C_r"], cfg["E_tau"]
    nch = seg.shape[0]
    res = {}

    def put(name, ci, v):
        res.setdefault(name, np.full(nch, np.nan))[ci] = v

    for ci in range(nch):
        s = seg[ci]
        if family == "fast":
            put("lzc", ci, lempel_ziv(s))
            put("perm_entropy", ci, permutation_entropy(s, m + 1, tau))
            put("higuchi_fd", ci, higuchi_fd(s, cfg["kmax"]))
        elif family == "medium":
            put("sampen", ci, sample_entropy(s, m, r * np.std(s), tau))
        else:
            for k, v in multiscale_entropy(s, m, r, tau, cfg["F_mse_scale"]).items():
                put(k, ci, v)
            put("dfa_alpha", ci, dfa_alpha_envelope(s, fs, cfg["dfa_fit_min_s"], cfg["dfa_band"]))
            for k, v in spectral_measures(s, fs, band, cfg["welch_s"], cfg["aper_range"]).items():
                put(k, ci, v)
    if family == "slow":
        res["wpli_alpha"] = np.full(nch, wpli_global(seg, fs, (8, 13), cfg["wpli_epoch_s"]))
        res["wpli_theta"] = np.full(nch, wpli_global(seg, fs, (4, 8), cfg["wpli_epoch_s"]))
    return res


def process_subject(npz_path: str, runs_cfg: list[dict], out_dir: str,
                    budget_s: float = 240.0, max_segments: int = 30) -> str:
    """For each run and each metric family, take segments of that family's window
    (up to budget_s seconds of signal, at most max_segments segments)."""
    z = np.load(npz_path, allow_pickle=False)
    data, fs0, ch = z["data"].astype(np.float64), float(z["fs"]), list(z["ch_names"])
    clean_runs = z["runs"]
    dataset, subject, label = str(z["dataset"]), str(z["subject"]), str(z["label"])
    rows = []
    regions = {"global": ch, **REGIONS}
    for cfg in runs_cfg:
        fs = cfg["G_fs"]
        x = resample_poly(data, int(fs), int(fs0), axis=1) if fs != fs0 else data
        runs_fs = (clean_runs * fs / fs0).astype(int)
        band = (cfg["D_band"][0], min(cfg["D_band"][1], fs / 2 - 1))
        x = sosfiltfilt(butter(4, band, btype="band", fs=fs, output="sos"), x, axis=1)
        agg = np.nanmean if cfg["H_seg_agg"] == "mean" else np.nanmedian
        for family in ("fast", "medium", "slow"):
            win = _window_s(cfg, family)
            L = int(win * fs)
            n_max = max(1, min(max_segments, int(budget_s // win)))
            segs = segments_from_runs(x, runs_fs, L, n_max)
            rows.append(dict(dataset=dataset, subject=subject, label=label, run=cfg["run"],
                             metric=f"n_segments_{family}", region="global", value=len(segs)))
            if not segs:
                continue
            if cfg["detrend"] == "linear":
                from scipy.signal import detrend as _detrend
                segs = [_detrend(sg, axis=1) for sg in segs]
            per_seg = [_family_values(family, sg, fs, band, cfg) for sg in segs]
            for metric in per_seg[0]:
                chan_vals = agg(np.stack([p[metric] for p in per_seg]), axis=0)
                for reg, names in regions.items():
                    if metric.startswith("wpli") and reg != "global":
                        continue
                    idx = [ch.index(c) for c in names if c in ch]
                    rows.append(dict(dataset=dataset, subject=subject, label=label,
                                     run=cfg["run"], metric=metric, region=reg,
                                     value=float(np.nanmean(chan_vals[idx]))))
    out = Path(out_dir) / "parts" / f"{dataset}__{subject}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


# --------------------------------------------------------------------------
# Summary for the Taguchi analysis
# --------------------------------------------------------------------------


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    return J * (b.mean() - a.mean()) / sp


# Floor applied to |g| before the larger-the-better S/N. An effect size of
# exactly zero at one site would otherwise send 1/g^2 towards infinity and drive
# the S/N of the whole run to about -120 dB, an artefact of the arithmetic rather
# than of the data. With a floor of 0.05 (a negligible effect) the worst possible
# per-site contribution is bounded at about -26 dB. Report the floor in the paper
# and check in a sensitivity analysis that conclusions do not depend on it.
G_FLOOR = 0.05


def sn_larger_better(g, floor=None):
    floor = G_FLOOR if floor is None else floor
    g = np.abs(np.asarray(g, float))
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return np.nan
    g = np.maximum(g, floor)
    return -10 * np.log10(np.mean(1 / g ** 2))


def summarize(out_dir: Path, groups: tuple[str, str], array: str):
    parts = sorted((out_dir / "parts").glob("*.csv"))
    if not parts:
        sys.exit("No subject files found in parts/.")
    df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    df.to_csv(out_dir / "features.csv", index=False)
    feat = df[~(df.metric.str.startswith("n_segments") | (df.metric == "mse_valid_scales"))]  # diagnostics stay in features.csv only
    g0, g1 = groups
    sites = sorted(feat.dataset.unique())
    runs = {r["run"]: r for r in build_runs(array)}
    rows = []
    for (run, metric, region), d in feat.groupby(["run", "metric", "region"]):
        row = {"run": run, **runs[run]["levels"], "metric": metric, "region": region}
        gs = []
        for site in sites:
            ds = d[d.dataset == site]
            g = hedges_g(ds[ds.label == g0].value, ds[ds.label == g1].value)
            row[f"g_{site}"] = g
            gs.append(g)
        row["n_sites"] = int(np.isfinite(gs).sum())
        row["SN_larger_better"] = sn_larger_better(gs)
        # Dual-response modelling (Vining and Myers, 1990): location and dispersion
        # of the effect size across sites, reported separately from the S/N ratio.
        ag = np.abs(np.asarray(gs, float))
        ag = ag[np.isfinite(ag)]
        row["mean_abs_g"] = float(ag.mean()) if len(ag) else np.nan
        row["log_var_abs_g"] = (float(np.log(ag.var(ddof=1))) if len(ag) >= 2 and ag.var(ddof=1) > 0
                                else np.nan)
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "taguchi_table.csv", index=False)
    # quick main-effects view (the formal ANOVA is done by the Taguchi analysis)
    factor_cols = [c for c in tab.columns
                   if c[:2] in ("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_")
                   or (c[:1] == "c" and c[1:3].isdigit())]
    me = []
    for metric_region, d in tab.groupby(["metric", "region"]):
        for fc in factor_cols:
            for lvl, dd in d.groupby(fc):
                me.append({"metric": metric_region[0], "region": metric_region[1],
                           "factor": fc, "level": lvl,
                           "mean_SN": dd.SN_larger_better.mean(),
                           "mean_abs_g": dd.mean_abs_g.mean(),
                           "mean_log_var_abs_g": dd.log_var_abs_g.mean()})
    pd.DataFrame(me).to_csv(out_dir / "main_effects.csv", index=False)
    print(f"Subjects {df.subject.nunique()}, sites {sites}")
    print(f"Written {out_dir/'features.csv'}, {out_dir/'taguchi_table.csv'}, "
          f"{out_dir/'main_effects.csv'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--data", type=Path, default=Path("continuous"))
    c.add_argument("--out", type=Path, default=Path("results"))
    c.add_argument("--array", choices=["L9", "L18", "L16"], default="L9")
    c.add_argument("--datasets", nargs="*", default=None)
    c.add_argument("--labels", nargs="*", default=["HC", "AD"])
    c.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    c.add_argument("--budget-s", type=float, default=240.0,
                   help="seconds of signal per subject, run and metric family")
    c.add_argument("--limit", type=int, default=None, help="subjects per dataset (testing)")
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, default=Path("results"))
    s.add_argument("--groups", nargs=2, default=["HC", "AD"])
    s.add_argument("--array", choices=["L9", "L18", "L16"], default="L9")
    a = ap.parse_args()

    for name, arr in (("L9", L9), ("L18", L18), ("L16", L16)):
        assert check_orthogonal(arr), f"{name} is not orthogonal"

    if a.cmd == "summarize":
        summarize(a.out, tuple(a.groups), a.array)
        return

    runs_cfg = build_runs(a.array)
    (a.out / "parts").mkdir(parents=True, exist_ok=True)
    todo = []
    for ds_dir in sorted(p for p in a.data.iterdir() if p.is_dir()):
        if a.datasets and ds_dir.name not in a.datasets:
            continue
        files = sorted(ds_dir.glob("*.npz"))
        n = 0
        for f in files:
            z = np.load(f, allow_pickle=False)
            if str(z["label"]) not in a.labels:
                continue
            if (a.out / "parts" / f"{ds_dir.name}__{f.stem}.csv").exists():
                continue
            todo.append(str(f))
            n += 1
            if a.limit and n >= a.limit:
                break
    print(f"{len(todo)} subjects to process, array {a.array} ({len(runs_cfg)} runs), "
          f"{a.jobs} parallel jobs")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(process_subject, f, runs_cfg, str(a.out), a.budget_s): f
                for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                status = "ok"
            except Exception as e:
                status = f"FAILED ({e})"
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {Path(futs[fut]).stem} {status}  "
                  f"elapsed {el/60:.1f} min, est. remaining {el/i*(len(todo)-i)/60:.1f} min")
    print("Done. Next step: python lead_taguchi_batch.py summarize "
          f"--out {a.out} --array {a.array}")


if __name__ == "__main__":
    main()
