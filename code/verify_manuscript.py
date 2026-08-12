"""DriftDriver - table -> script -> CSV -> manuscript, checked mechanically.

Six times in this project a number reached the manuscript from a run that left
no script and no stored output, and three of those were load-bearing. This
script closes that loop: it recomputes each reported quantity from the stored
CSV and asserts that the string the manuscript prints is the string the data
support.

Usage: python3 verify_manuscript.py [path/to/manuscript.md]
Exit status is non-zero if any check fails.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

# Paths are resolved relative to this file so the checks run from a fresh clone.
BASE = Path(__file__).resolve().parent.parent / "derived"
if len(sys.argv) > 1:
    MS = Path(sys.argv[1])
else:
    MS = Path(__file__).resolve().parent.parent / "paper" / "main.txt"
    if not MS.exists():
        sys.exit("usage: verify_manuscript.py <manuscript.md|.tex|.txt>")

text = MS.read_text(encoding="utf-8")
if MS.suffix == ".txt":
    # pdftotext output: the end-to-end check. pdflatex drops characters it cannot
    # set, and the .tex still holds them, so only the rendered text proves the
    # number reached the page. Line wrapping is the only thing normalised away.
    text = re.sub(r"\s+", " ", text)
if MS.suffix == ".tex":
    # Run the same checks on the ported LaTeX so a number cannot be lost in
    # translation. Undo only the escaping pandoc adds; nothing else is touched.
    for esc, plain in ((r"$-$", "−"), (r"$\times$", "×"), (r"$\pm$", "±"),
                       (r"$\approx$", "≈"), (r"$\Delta$", "Δ"), (r"$\le$", "≤"),
                       (r"\%", "%"), (r"\&", "&"), (r"\_", "_"), (r"\$", "$"),
                       (r"\#", "#"), ("~", " ")):
        text = text.replace(esc, plain)
    text = re.sub(r"\$\^\{(-?\d+)\}\$",
                  lambda m: "".join("⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".index(c)]
                                    for c in m.group(1)), text)
checks, failures = [], []


def want(label, needle):
    ok = needle in text
    checks.append((label, needle, ok))
    if not ok:
        failures.append((label, needle))


def sign(x, nd=3):
    """Manuscript convention: U+2212 for minus, plus sign kept."""
    s = f"{x:+.{nd}f}"
    return s.replace("-", "−")


# --- 5.1 primary temporal ------------------------------------------------
pt = pd.read_csv(BASE / "primary_temporal_summary.csv")
im18 = pt[(pt.panel == "imbalanced") & (pt.cutoff == 2018)].set_index("year")
for y in range(2019, 2024):
    want(f"5.1 delta {y}", sign(im18.loc[y, "delta"]))
    want(f"5.1 AUC {y}", f"{im18.loc[y, 'auc']:.3f}")
im16 = pt[(pt.panel == "imbalanced") & (pt.cutoff == 2016)].set_index("year")
for y in (2021, 2022, 2023):
    want(f"5.1 cutoff-2016 delta {y}", sign(im16.loc[y, "delta"]))

d = pd.read_csv(BASE / "primary_temporal.csv")
d18 = d[(d.panel == "imbalanced") & (d.cutoff == 2018)]
yr = d18.groupby("year").delta.mean()
r, p = stats.pearsonr(range(1, 6), yr.values)
want("5.1 elapsed r", f"r = {sign(r, 3)} (p = {p:.3f}")

# --- 5.2 contemporaneous -------------------------------------------------
hc = pd.read_csv(BASE / "hospital_contemporaneous.csv")
g = hc.groupby("year")[["auc_transfer", "auc_contemporaneous"]].mean()
for y in range(2019, 2024):
    want(f"5.2 transfer {y}", f"{g.loc[y, 'auc_transfer']:.3f}")
    want(f"5.2 contemp {y}", f"{g.loc[y, 'auc_contemporaneous']:.3f}")

# --- 5.3 monitor ---------------------------------------------------------
rd = pd.read_csv(BASE / "regime_distance_summary.csv").set_index("year")
for y in range(2019, 2024):
    want(f"5.3 distance {y}", f"{rd.loc[y, 'regime_distance']:.3f}")
oos = rd.loc[2019:2023]
r, p = stats.pearsonr(oos.regime_distance, oos.delta.abs())
want("5.3 monitor r", f"r = {sign(r, 3)} (p = {p:.2f}")

# --- 5.6 canary ----------------------------------------------------------
cp = pd.read_csv(BASE / "canary_power.csv")
tab = cp.pivot_table(index="n", columns="year", values="alarm_rate") * 100
for n in (100, 400):
    for y in (2021, 2022):
        want(f"5.6 power n={n} {y}", f"{tab.loc[n, y]:.1f}%")

# --- 5.7 window grid -----------------------------------------------------
mg = pd.read_csv(BASE / "grid_all_windows_marginals.csv")
mg = mg[mg.panel == "imbalanced"]
for _, row in mg[mg.outcome == "deficit"].iterrows():
    want(f"5.7 E {row.spec}", sign(row.E, 4))
    want(f"5.7 T {row.spec}", sign(row["T"], 4))


def marg(res):
    w = res.pivot_table(index=["model", "seed"], columns=["entity", "period"], values="auc")
    a, b = w[("E_seen", "P_seen")], w[("E_unseen", "P_seen")]
    c, dd = w[("E_seen", "P_unseen")], w[("E_unseen", "P_unseen")]
    return a - b, a - c, a - dd


ex = pd.read_csv(BASE / "grid_exit_windows.csv")
ex = ex[ex.spec == "exit 2011-2016 -> 2017-2020"]
w = ex.pivot_table(index=["model", "seed"], columns=["entity", "period"], values="auc")
E = ((w[("E_seen", "P_seen")] - w[("E_unseen", "P_seen")]) +
     (w[("E_seen", "P_unseen")] - w[("E_unseen", "P_unseen")])) / 2
T = ((w[("E_seen", "P_seen")] - w[("E_seen", "P_unseen")]) +
     (w[("E_unseen", "P_seen")] - w[("E_unseen", "P_unseen")])) / 2
want("5.7 exit E", sign(E.mean(), 4))
want("5.7 exit T", sign(T.mean(), 4))

for lab, f in [("imbalanced", "grid_2x2_cells_imbal.csv"), ("balanced", "grid_2x2_cells_bal.csv")]:
    e, t, j = marg(pd.read_csv(BASE / f))
    want(f"2x2 {lab} excess", sign((e + t - j).mean(), 4))

# --- 3.1 / 5.1 subsidy-excluded deficit rates ----------------------------
# Quoted in two sections from two runs that disagreed by 0.1-0.3 pp; recomputed
# from the panel so both sections are held to one number.
pan = pd.read_parquet(BASE / "unbalanced_panel_2011_2023.parquet")
for y in range(2019, 2024):
    g = pan[pan.year == y]
    obs = (g["Net Income"] < 0).mean() * 100
    exc = ((g["Net Income"] - g["Total Other Income"]) < 0).mean() * 100
    want(f"5.1 subsidy gap {y}", f"{exc - obs:.1f} pp")
    if y == 2020:
        want("5.1 excluded rate 2020", f"{exc:.1f}%")

# --- power domain --------------------------------------------------------
po = pd.read_csv(BASE / "power_multicutoff_origspec.csv")
for c, gg in po.groupby("cutoff"):
    gg = gg.sort_values("elapsed")
    r, _ = stats.pearsonr(gg.elapsed, gg.delta_pp)
    want(f"5.4 r cutoff {c}", f"+{r:.3f}")

# Identification table: every cell, plus the oldest/newest ratio. Figure 3b is
# these same numbers plotted, so a divergence between figure and table fails here.
wide = po.pivot(index="test_year", columns="cutoff", values="delta_pp")
for y in range(2019, 2026):  # the 2018-cutoff table, i.e. the orange curve of Fig 3a
    want(f"5.4 degradation {y}", f"+{wide.loc[y, 2018]:.2f} pp")
for y in range(2021, 2026):
    for c in (2016, 2018, 2020):
        want(f"5.4 identification {y}@{c}", f"+{wide.loc[y, c]:.2f} pp")
    want(f"5.4 ratio {y}", f"{wide.loc[y, 2016] / wide.loc[y, 2020]:.2f}×")
for y in (2018, 2019):  # the 2016-cutoff reversal, annotated on Fig 3a
    want(f"5.4 reversal {y}", f"({y}, +{wide.loc[y, 2016]:.2f} pp)")
assert wide.loc[2019, 2016] < wide.loc[2018, 2016], "the reported reversal is gone"

print(f"{len(checks) - len(failures)} PASS / {len(failures)} FAIL")
for label, needle in failures:
    print(f"  FAIL {label}: manuscript does not contain '{needle}'")
sys.exit(1 if failures else 0)
