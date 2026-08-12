"""DriftDriver - Step (b) on the hospital domain: transfer vs contemporaneous (Table 3).

Section 5.2 reports this table but no script in code/ produced it and no output was
stored; the only contemporaneous script in the repo is power_contemporaneous.py, which
is the electricity domain. This implements the arms exactly as Section 4.2 defines them.

  arm A (transfer):        fit on entity set A, years 2011-2018;  score entity set B, year t
  arm B (contemporaneous): fit on entity set A, year t;           score entity set B, year t

Both arms are scored on the same rows (entity set B in year t), which are training data
for neither, so the only difference between them is the period the fit came from.
Entity split, features, seeds, and model families are identical to grid_all_windows.py.

Usage: python3 hospital_contemporaneous.py
Output: derived/hospital_contemporaneous.csv
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
TRAIN_YEARS = list(range(2011, 2019))
TEST_YEARS = [2019, 2020, 2021, 2022, 2023]


def models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }


def fit_score(fit_rows, score_rows, seed):
    """Fit every family on fit_rows, return {family: AUC on score_rows}."""
    sc = StandardScaler().fit(fit_rows[FEATS])
    X, y = sc.transform(fit_rows[FEATS]), fit_rows["deficit"].values
    Xs, ys = sc.transform(score_rows[FEATS]), score_rows["deficit"].values
    out = {}
    for name, m in models(seed).items():
        m.fit(X, y)
        out[name] = roc_auc_score(ys, m.predict_proba(Xs)[:, 1])
    return out


def main():
    df = pd.read_parquet(BASE / "unbalanced_panel_2011_2023.parquet")
    df = df.dropna(subset=FEATS + ["deficit", "year", "Provider CCN"])
    hosp = df["Provider CCN"].unique()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(hosp)
        cut = int(len(perm) * 0.7)
        seen_h, unseen_h = set(perm[:cut]), set(perm[cut:])

        transfer_fit = df[df["Provider CCN"].isin(seen_h) & df["year"].isin(TRAIN_YEARS)]
        for t in TEST_YEARS:
            score = df[df["Provider CCN"].isin(unseen_h) & (df["year"] == t)]
            contemp_fit = df[df["Provider CCN"].isin(seen_h) & (df["year"] == t)]
            a = fit_score(transfer_fit, score, seed)
            b = fit_score(contemp_fit, score, seed)
            for fam in a:
                rows.append({"year": t, "seed": seed, "model": fam,
                             "auc_transfer": a[fam], "auc_contemporaneous": b[fam],
                             "n_score": len(score), "n_contemp_fit": len(contemp_fit)})

    res = pd.DataFrame(rows)
    res.to_csv(BASE / "hospital_contemporaneous.csv", index=False)

    g = res.groupby("year")[["auc_transfer", "auc_contemporaneous"]].mean()
    g["recovered"] = g["auc_contemporaneous"] - g["auc_transfer"]
    print(f"{'year':>6} {'A (transfer)':>13} {'B (contemp)':>12} {'B - A':>8}")
    for y, r in g.iterrows():
        print(f"{y:>6} {r.auc_transfer:>13.3f} {r.auc_contemporaneous:>12.3f} "
              f"{r.recovered:>+8.3f}")
    print("\nper family, 2021:")
    for fam, r in res[res.year == 2021].groupby("model")[
            ["auc_transfer", "auc_contemporaneous"]].mean().iterrows():
        print(f"  {fam:7s} A {r.auc_transfer:.3f}  B {r.auc_contemporaneous:.3f}  "
              f"B-A {r.auc_contemporaneous - r.auc_transfer:+.3f}")
    d21 = res[res.year == 2021]
    p = stats.ttest_rel(d21.auc_contemporaneous, d21.auc_transfer).pvalue
    print(f"\n2021 paired t-test over 15 runs: p = {p:.3g}")
    print("wrote", BASE / "hospital_contemporaneous.csv")


if __name__ == "__main__":
    main()
