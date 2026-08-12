# Figures — "Drift Inherits Its Driver"

Regenerate all five with:  `python3 ../code/make_figures.py`

**Figures 1–3 read stored results; they do not re-estimate.** Run the producing
script first if the panel or the load data changed, otherwise the figure will
silently show the previous run:

| Figure | reads | produced by |
|---|---|---|
| 1 | `../derived/primary_temporal_summary.csv` | `../code/primary_temporal.py` |
| 2 | `../derived/regime_distance_summary.csv` | `../code/regime_distance.py` |
| 3 | `../derived/power_multicutoff_origspec.csv` | `../code/power_multicutoff_origspec.py` |

Figures 4 and 5 have no table to match and are computed inside `make_figures.py`
from the same origspec electricity model behind Figure 3 (~40 s).

Until 2026-08-13 this file recomputed Figures 1 and 2 under its own specification
(50/50 entity split, `min_samples_leaf=20`) while the manuscript tables came from
the canonical scripts (70/30 split, `min_samples_leaf=5`). The figures and the
tables they illustrate were therefore drawn from different runs. No classifier
specification lives in `make_figures.py` any more.

| File | Manuscript | Shows |
|---|---|---|
| `fig1_regime_indexed` | Figure 1 | Hospital Δ AUC by calendar year, two training cutoffs overlaid. 2021 falls −0.088 (trained through 2016) and −0.080 (through 2018): the same dip at 5 and 3 years of model age, a gap well inside the seed spread (sd ≈ 0.019). |
| `fig2_unlabeled_fails` | Figure 2 | Unlabeled regime distance vs. realized degradation, 2019–2023. Out-of-sample r = +0.143 (p = 0.82) against signed Δ AUC and r = −0.056 (p = 0.93) against loss magnitude — uncorrelated, with the conventions disagreeing in direction. 2020 has the largest input shift and no performance loss. |
| `fig3_power_monotone` | Figure 3 | Electricity Δ MAPE vs. baseline, all three training cutoffs. **(a)** against model age: monotone at every cutoff (r = +0.949, +0.979, +0.981), with the single 2016-cutoff reversal at three years visible rather than averaged away. **(b)** the same estimates against the calendar — the Section 5.4 identification test, where the vertical gap at 2021 is 1.96× from newest to oldest model. |
| `fig4_hourly` | Figure 4 | Error increase by hour: +8.39 pp at 13:00 against +0.31 pp at 06:00. |
| `fig5_duck_curve` | Figure 5 | Residuals at 13:00 and 19:00 by year — midday suppression, evening excess. |

Both PNG (300 dpi) and PDF (vector) are written; use the PDF for submission.

## Palette
`#2a78d6` (blue) and `#eb6834` (orange). Validated with the dataviz skill's
`validate_palette.js`: all six checks PASS, worst adjacent CVD ΔE 24.7 (protan),
normal-vision ΔE 33.6, contrast ≥ 3:1 against the surface. Do not substitute
colors without re-running that validator. Identity is never carried by color
alone — every multi-series figure has both a legend and direct labels.

Figure 3 needs a third series and takes the next categorical slot, `#1baf7a`
(aqua). Re-validated as a set under `--pairs all`: checks PASS, worst CVD ΔE 9.2
(deutan), worst normal-vision ΔE 24.0. Aqua sits at 2.74:1 on the white surface,
under the 3:1 gate, so the relief rule applies and every Figure 3 curve carries a
direct label at its end in addition to the legend. Colors are keyed to the
training cutoff and match Figure 1 (2016 blue, 2018 orange), so the same cutoff is
the same color in both domains.
