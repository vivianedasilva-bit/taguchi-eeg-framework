# Robust parameter design framework for nonlinear EEG measures — simulation study

Code for: Dasilva V, Pino O, Borges Silva M. *A robust parameter design framework for reproducible
nonlinear EEG biomarkers, a simulation study with known ground truth.* (manuscript in preparation)

No human data are used. Every dataset is regenerated exactly from the seeds below.

## Scripts
| Script | Step |
|---|---|
| `simulate_eeg_framework.py` | Synthetic 19-channel EEG with known group differences; L8 outer array of 7 noise factors = 8 sites |
| `lead_taguchi_batch.py` | L16 inner array (12 factors + 3 interactions), all measures, Hedges' g per site, S/N |
| `framework_confirm.py` | Robust configuration per measure, confirmation run, default configurations |
| `run_framework.py` | Runs the three steps above for every scenario and repetition (resumable) |
| `make_figures.py` | Recovery, null distribution, stability, aliasing diagnostic |
| `framework_extras.py` | Floor sensitivity, Welch tests in the null scenario, noise-factor effects, sensitivity maps |
| `framework_anova.py` | Taguchi ANOVA with repetitions as blocks, prediction and confidence interval of the confirmation |
| `run_validation.py` | Independent validation on new seeds |
| `framework_window.py` | Minimum reliable window: ICC(2,1) and S/N as a function of segment duration |

## Reproduce
```
python -m pip install -r requirements.txt
python run_framework.py --root framework_L8 --reps 5 --outer L8 --per-group 30 --duration 300
python make_figures.py --root framework_L8 --out figures_final
python framework_extras.py --root framework_L8 --out figures_extras_final
python framework_anova.py --root framework_L8 --out figures_anova
python run_validation.py --root framework_L8 --reps 1 2
python run_validation.py --root framework_L8 --summarize --out figures_validation
python framework_window.py compute --run framework_L8/strong/rep01 --out framework_L8/strong/rep01/window
python framework_window.py summarize --out framework_L8/strong/rep01/window --fig figures_window
```
One repetition of one scenario takes about 8 h on a 12-thread desktop (11 processes).

## Seeds
Dataset seed = 1000 × s + k (s: 1 strong, 2 weak, 3 null, 4 nonlinear_only; k: repetition 1–5).
Validation seed = 1000 × s + 500 + k.

## Environment
Python 3.13 with numpy 2.2.6, scipy 1.14.1, pandas 3.0.2, numba 0.68.0, matplotlib 3.9.2 (`requirements.txt`).
Full environment used for the published results: `requirements_lock.txt`.
Numba is optional (identical results, ~3× faster sample entropy).

## Licence
To be defined by the authors.
