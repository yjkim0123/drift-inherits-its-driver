"""DriftDriver - THE primary hospital result of Section 5.1, with a stored output.

The per-year degradation series the manuscript leads with (2019 through 2023,
against the in-window unseen-entity baseline) had no script of its own and no
stored CSV: the only stored run of that design was the `full` arm of
ablate_other_inc.py, whose numbers differ from the ones the manuscript quoted,
so the quoted figures came from a run nobody can reproduce.

This is that design, written down once, run on both panels, and saved:

  - split hospitals 70/30 into seen/unseen (per seed)
  - fit on seen hospitals, training years 2011-2018
  - baseline = AUC on UNSEEN hospitals in the TRAINING years (in-window,
    entity-novel), so the per-year delta isolates period novelty
  - delta(t) = AUC(unseen hospitals, year t) - baseline, per (seed, family)
  - 5 seeds x 3 families = 15 paired values per test year; the reported
    p-value is a two-sided one-sample t-test of those 15 against zero

Usage: python3 primary_temporal.py
Output: derived/primary_temporal.csv   (one row per seed x family x test year)
        derived/primary_temporal_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent / "derived"
FEATS = ["labor_ratio", "overhead_ratio", "debt_ratio", "other_inc_ratio", "log_beds"]
SEEDS = [0, 1, 2, 3, 4]

# (cutoff label, training years, test years). The 2016 cutoff is the
# identification test: it holds the calendar year fixed and varies model age.
CUTOFFS = [
    (2018, list(range(2011, 2019)), list(range(2019, 2024))),
    (2016, list(range(2011, 2017)), list(range(2017, 2024))),
]
FAMILIES = ["LogReg", "RF", "GBM"]

PANELS = [
    ("imbalanced", "unbalanced_panel_2011_2023.parquet"),
    ("balanced", "balanced_panel_2011_2023.parquet"),
]


def models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }


def run_panel(path, panel, cutoff, train_years, test_years):
    df = pd.read_parquet(path).dropna(subset=FEATS + ["deficit", "year", "Provider CCN"])
    hosp = df["Provider CCN"].unique()
    rows = []
    TRAIN_YEARS, TEST_YEARS = train_years, test_years
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(hosp)
        cut = int(len(perm) * 0.7)
        seen_h, unseen_h = set(perm[:cut]), set(perm[cut:])

        fit = df[df["Provider CCN"].isin(seen_h) & df["year"].isin(TRAIN_YEARS)]
        unseen = df[df["Provider CCN"].isin(unseen_h)]

        sc = StandardScaler().fit(fit[FEATS])
        Xf, yf = sc.transform(fit[FEATS]), fit["deficit"].values
        for mname, m in models(seed).items():
            m.fit(Xf, yf)
            base_sub = unseen[unseen["year"].isin(TRAIN_YEARS)]
            base = roc_auc_score(base_sub["deficit"].values,
                                 m.predict_proba(sc.transform(base_sub[FEATS]))[:, 1])
            for y in TEST_YEARS:
                sub = unseen[unseen["year"] == y]
                auc = roc_auc_score(sub["deficit"].values,
                                    m.predict_proba(sc.transform(sub[FEATS]))[:, 1])
                rows.append({"panel": panel, "cutoff": cutoff, "seed": seed,
                             "model": mname, "year": y, "elapsed": y - cutoff,
                             "auc": auc, "baseline": base, "delta": auc - base,
                             "n": len(sub), "n_baseline": len(base_sub)})
    return pd.DataFrame(rows)


def main():
    parts, summary = [], []
    for panel, fname in PANELS:
        for cutoff, tr, te in CUTOFFS:
            res = run_panel(BASE / fname, panel, cutoff, tr, te)
            parts.append(res)
            print(f"\n===== {panel}, cutoff {cutoff} =====")
            print(f"baseline (unseen entities, {tr[0]}-{tr[-1]}) = {res.baseline.mean():.3f}")
            for y in te:
                v = res[res.year == y]
                p = stats.ttest_1samp(v.delta.values, 0.0).pvalue
                summary.append({"panel": panel, "cutoff": cutoff, "year": y,
                                "elapsed": y - cutoff, "auc": v.auc.mean(),
                                "auc_sd": v.auc.std(), "baseline": v.baseline.mean(),
                                "delta": v.delta.mean(), "p": p})
                print(f"  {y} (elapsed {y - cutoff}): AUC={v.auc.mean():.3f} "
                      f"+- {v.auc.std():.3f}   delta={v.delta.mean():+.4f}  p={p:.3g}")
            yr = res.groupby("year").delta.mean()
            r, pr = stats.pearsonr(np.arange(1, len(yr) + 1), yr.values)
            print(f"  elapsed-time correlation r={r:+.3f} (p={pr:.3f})")
            print(f"  monotone in elapsed time: {bool(np.all(np.diff(yr.values) <= 0))}")

    pd.concat(parts, ignore_index=True).to_csv(BASE / "primary_temporal.csv", index=False)
    pd.DataFrame(summary).to_csv(BASE / "primary_temporal_summary.csv", index=False)
    print("\nwrote", BASE / "primary_temporal.csv")
    print("wrote", BASE / "primary_temporal_summary.csv")


if __name__ == "__main__":
    main()
