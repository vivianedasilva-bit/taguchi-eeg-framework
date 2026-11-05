"""
simulate_eeg_framework.py
=========================

Synthetic multi-site resting-state EEG with KNOWN group differences, for the
framework paper (robust parameter design of nonlinear EEG measures, no human data).

The output has exactly the same format as lead_continuous.py, so
lead_taguchi_batch.py runs on it unchanged:
    sim/<SITE>/<subject>.npz   (data in microvolts, fs, ch_names, runs, label, dataset, subject)
    sim/ground_truth.csv       the generative parameters of every simulated subject

Signal model (per channel)
--------------------------
x(t) = aperiodic background (1/f^beta noise)
     + alpha rhythm (peak frequency f_alpha, amplitude envelope with long-range
       temporal correlations controlled by a Hurst exponent H_env)
     + theta rhythm (relative amplitude theta_gain)
     + a shared, lagged posterior alpha source (coupling strength k_couple),
       which produces genuine phase-lagged connectivity
The alpha rhythm also carries a second harmonic whose phase is coupled to the
fundamental to a degree set by `phase_lock`. This coupling is a genuinely nonlinear
property: it changes the waveform and higher-order statistics without changing the
power spectrum. The scenario `nonlinear_only` differs between groups ONLY in this
coupling, which tests whether a measure detects nonlinear structure rather than
spectral or linear long-range differences.
Group differences are introduced ONLY through these generative parameters.
They are simulation settings chosen to mimic patterns reported in the
literature (alpha slowing, more theta, weaker long-range correlations, weaker
coupling); they are not claims about real patients.

Site effects (the noise factor of the Taguchi design)
-----------------------------------------------------
Each site applies its own acquisition distortions: amplifier gain, line noise
(50 or 60 Hz), white sensor noise, slow drift, low-pass characteristic,
muscle bursts and a small site-specific shift in alpha frequency
(population differences). These are what a robust configuration must resist.

Usage
-----
python simulate_eeg_framework.py --out sim --sites 5 --per-group 30 --duration 300
python simulate_eeg_framework.py --out sim --scenario weak      # MCI-like, small effects
python simulate_eeg_framework.py --out sim --scenario null      # no group difference (false-positive check)
python simulate_eeg_framework.py --out sim --scenario nonlinear_only   # same spectra, differ only in phase coupling
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
      "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
POSTERIOR = {"T5", "P3", "Pz", "P4", "T6", "O1", "O2"}
FRONTAL = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}

# Group parameters: mean and between-subject SD of each generative parameter.
SCENARIOS = {
    "strong": {  # HC vs AD-like, clear effects
        "HC": dict(f_alpha=(10.0, 0.7), beta=(1.2, 0.15), H_env=(0.75, 0.05),
                   theta_gain=(0.35, 0.10), k_couple=(0.45, 0.10), phase_lock=(0.20, 0.10)),
        "AD": dict(f_alpha=(8.6, 0.8), beta=(1.4, 0.15), H_env=(0.60, 0.05),
                   theta_gain=(0.70, 0.15), k_couple=(0.25, 0.10), phase_lock=(0.80, 0.10)),
    },
    "weak": {    # HC vs MCI-like, small effects
        "HC": dict(f_alpha=(10.0, 0.7), beta=(1.2, 0.15), H_env=(0.75, 0.05),
                   theta_gain=(0.35, 0.10), k_couple=(0.45, 0.10), phase_lock=(0.20, 0.10)),
        "AD": dict(f_alpha=(9.6, 0.8), beta=(1.27, 0.15), H_env=(0.70, 0.05),
                   theta_gain=(0.45, 0.12), k_couple=(0.38, 0.10), phase_lock=(0.50, 0.10)),
    },
    "null": {    # identical groups: any "effect" found is a false positive
        "HC": dict(f_alpha=(10.0, 0.7), beta=(1.2, 0.15), H_env=(0.75, 0.05),
                   theta_gain=(0.35, 0.10), k_couple=(0.45, 0.10), phase_lock=(0.20, 0.10)),
        "AD": dict(f_alpha=(10.0, 0.7), beta=(1.2, 0.15), H_env=(0.75, 0.05),
                   theta_gain=(0.35, 0.10), k_couple=(0.45, 0.10), phase_lock=(0.20, 0.10)),
    },
    "nonlinear_only": {  # identical spectra; groups differ ONLY in phase coupling
        "HC": dict(f_alpha=(10.0, 0.7), beta=(1.2, 0.15), H_env=(0.75, 0.05),
                   theta_gain=(0.35, 0.10), k_couple=(0.45, 0.10), phase_lock=(0.20, 0.10)),
        "AD": dict(f_alpha=(10.0, 0.7), beta=(1.2, 0.15), H_env=(0.75, 0.05),
                   theta_gain=(0.35, 0.10), k_couple=(0.45, 0.10), phase_lock=(0.80, 0.10)),
    },
}
HARMONIC_GAIN = 0.4   # amplitude of the second harmonic relative to the alpha fundamental


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def colored_noise(n, exponent, rng):
    """Gaussian noise with power spectrum ~ 1/f^exponent (spectral synthesis)."""
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / f ** (exponent / 2)
    x = np.fft.irfft(spec, n)
    return (x - x.mean()) / x.std()


def fgn_envelope(n, H, fs, rng, smooth_s=0.25):
    """Positive amplitude envelope with long-range correlations.
    Fractional Gaussian noise of Hurst exponent H has spectrum ~ 1/f^(2H-1)."""
    z = colored_noise(n, 2 * H - 1, rng)
    sos = butter(2, 1.0 / smooth_s, fs=fs, output="sos")       # keep slow fluctuations
    z = sosfiltfilt(sos, z)
    z = (z - z.mean()) / (z.std() + 1e-12)
    return np.exp(0.5 * z)


def wandering_phase(n, f0, fs, rng, jitter=0.4):
    df = colored_noise(n, 2.0, rng) * jitter
    return 2 * np.pi * np.cumsum(f0 + df) / fs + rng.uniform(0, 2 * np.pi)


def oscillation(n, f0, fs, rng, jitter=0.4):
    """Narrow-band rhythm: sinusoid with slowly wandering frequency."""
    return np.sin(wandering_phase(n, f0, fs, rng, jitter))


def nonlinear_alpha(n, f0, fs, rng, lock):
    """Alpha rhythm with a second harmonic whose phase is coupled to the fundamental.

    harmonic = lock * sin(2*phi) + sqrt(1 - lock^2) * sin(2*phi_indep)
    phi_indep is an independent phase with the same frequency and jitter, so the
    power spectrum does not depend on `lock`. Only the phase coupling between the
    fundamental and the harmonic (quadratic phase coupling, a genuinely nonlinear
    property that surrogate tests destroy) changes with `lock`."""
    lock = float(np.clip(lock, 0.0, 1.0))
    phi = wandering_phase(n, f0, fs, rng)
    phi_ind = wandering_phase(n, f0, fs, rng)
    harm = lock * np.sin(2 * phi) + np.sqrt(1 - lock ** 2) * np.sin(2 * phi_ind)
    return np.sin(phi) + HARMONIC_GAIN * harm


def draw(params, rng):
    return {k: rng.normal(m, s) for k, (m, s) in params.items()}


# --------------------------------------------------------------------------
# One subject
# --------------------------------------------------------------------------

def simulate_subject(p, site, fs, duration, rng):
    n = int(fs * duration)
    shared = oscillation(n, p["f_alpha"], fs, rng) * fgn_envelope(n, p["H_env"], fs, rng)
    X = np.zeros((len(CH), n))
    for i, ch in enumerate(CH):
        post = ch in POSTERIOR
        bg = colored_noise(n, p["beta"], rng) * 10.0
        alpha_gain = 12.0 if post else 5.0
        alpha = nonlinear_alpha(n, p["f_alpha"] + site["alpha_shift"], fs, rng, p["phase_lock"]) \
            * fgn_envelope(n, p["H_env"], fs, rng) * alpha_gain
        theta = oscillation(n, 6.0, fs, rng, jitter=0.6) * 8.0 * p["theta_gain"]
        x = bg + alpha + theta
        if post:   # lagged copy of a shared posterior source -> phase-lagged coupling
            lag = int(fs * rng.uniform(0.005, 0.030))
            x += np.roll(shared, lag) * 10.0 * max(p["k_couple"], 0.0)
        X[i] = x
    return apply_site(X, site, fs, rng, frontal_idx=[i for i, c in enumerate(CH) if c in FRONTAL])


# --------------------------------------------------------------------------
# Site distortions (noise factor)
# --------------------------------------------------------------------------

def make_site(k, rng):
    return dict(
        name=f"SITE{k + 1}",
        gain=rng.uniform(0.8, 1.25),
        line_hz=50.0 if k % 2 == 0 else 60.0,
        line_amp=rng.uniform(0.0, 3.0),
        white=rng.uniform(1.0, 4.0),
        drift=rng.uniform(0.0, 8.0),
        lowpass=rng.choice([35.0, 45.0, 70.0]),
        emg_rate=rng.uniform(0.0, 0.05),        # muscle bursts per second
        alpha_shift=rng.normal(0.0, 0.3),        # population difference between sites
    )


# Outer array (Taguchi crossed design): each row of the standard L8 is one
# simulated site whose acquisition distortions are set by design instead of
# drawn at random. Seven two-level noise factors, one per column.
L8 = [
    [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 2, 2, 2, 2], [1, 2, 2, 1, 1, 2, 2], [1, 2, 2, 2, 2, 1, 1],
    [2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 2, 1, 2, 1], [2, 2, 1, 1, 2, 2, 1], [2, 2, 1, 2, 1, 1, 2]]
NOISE_FACTORS = [            # (name, level 1, level 2)
    ("gain",        0.80, 1.25),   # amplifier gain
    ("line_amp",    0.5,  3.0),    # line noise amplitude (uV)
    ("white",       1.0,  4.0),    # sensor noise SD (uV)
    ("drift",       1.0,  8.0),    # slow drift amplitude (uV)
    ("lowpass",     35.0, 70.0),   # acquisition low-pass (Hz)
    ("emg_rate",    0.0,  0.05),   # muscle bursts per second
    ("alpha_shift", -0.3, 0.3),    # population shift of alpha frequency (Hz)
]


def make_site_L8(k):
    row = L8[k]
    site = dict(name=f"L8R{k + 1}", line_hz=50.0)
    for (name, lo, hi), lvl in zip(NOISE_FACTORS, row):
        site[name] = lo if lvl == 1 else hi
    site["outer_levels"] = "".join(str(v) for v in row)
    return site


def apply_site(X, s, fs, rng, frontal_idx):
    n = X.shape[1]
    t = np.arange(n) / fs
    X = X * s["gain"]
    X += s["line_amp"] * np.sin(2 * np.pi * s["line_hz"] * t)
    X += rng.standard_normal(X.shape) * s["white"]
    X += s["drift"] * colored_noise(n, 2.0, rng)
    # muscle bursts on frontal and temporal channels
    n_bursts = rng.poisson(s["emg_rate"] * n / fs)
    for _ in range(n_bursts):
        a = rng.integers(0, n - int(fs))
        L = int(fs * rng.uniform(0.2, 1.0))
        sos = butter(4, [25, min(90, fs / 2 - 1)], btype="band", fs=fs, output="sos")
        burst = sosfiltfilt(sos, rng.standard_normal(L)) * rng.uniform(10, 40)
        X[frontal_idx, a:a + L] += burst
    lp = min(s["lowpass"], fs / 2 - 1)
    X = sosfiltfilt(butter(4, lp, btype="low", fs=fs, output="sos"), X, axis=1)
    return X


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("sim"))
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="strong")
    ap.add_argument("--sites", type=int, default=5)
    ap.add_argument("--per-group", type=int, default=30)
    ap.add_argument("--duration", type=float, default=300.0, help="seconds per subject")
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outer", choices=["none", "L8"], default="none",
                    help="L8 = sites defined by an L8 outer array of 7 noise factors (overrides --sites)")
    a = ap.parse_args()
    if a.outer == "L8":
        a.sites = len(L8)

    rng = np.random.default_rng(a.seed)
    groups = SCENARIOS[a.scenario]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(a.sites):
        site = make_site_L8(k) if a.outer == "L8" else make_site(k, rng)
        d = a.out / site["name"]
        d.mkdir(parents=True, exist_ok=True)
        for label, params in groups.items():
            for j in range(a.per_group):
                p = draw(params, rng)
                X = simulate_subject(p, site, a.fs, a.duration, rng).astype(np.float32)
                subj = f"{label}{j + 1:03d}"
                np.savez_compressed(d / f"{subj}.npz", data=X, fs=a.fs, ch_names=np.array(CH),
                                    runs=np.array([[0, X.shape[1]]]), label=label,
                                    dataset=site["name"], subject=subj)
                rows.append({"site": site["name"], "subject": subj, "label": label,
                             "scenario": a.scenario, **{k2: round(v, 4) for k2, v in p.items()},
                             **{f"site_{k2}": v for k2, v in site.items() if k2 != "name"}})
        print(f"{site['name']}: {2 * a.per_group} subjects")
    with open(a.out / "ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Done. Ground truth in {a.out / 'ground_truth.csv'}")


if __name__ == "__main__":
    main()
