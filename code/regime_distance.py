"""DriftDriver - Section 5.3, the unlabeled monitor, with a stored output.

The regime distances the manuscript reports (0.501 in 2019 through 0.755 in
2020) existed only in STATUS.md: no script, no CSV. This is that measurement
written down, together with the per-year degradation it is scored against, so
the admission test of Section 4.3 can be re-run from one file.

Monitor (Section 4.3): for each year t, a discriminator separates the
2011-2018 training-window feature matrix from the year-t feature matrix, using
exactly the five features the deployed model consumes. The training-window side
is subsampled to the size of year t so the discriminator cannot exploit class
imbalance, and the reported regime distance is its held-out AUC (0.5 =
indistinguishable). 5 seeds x 3 model families, as everywhere else.

Degradation: the same quantity as primary_temporal.py, extended to all
thirteen years, so the thirteen-year correlation reported as the trap in
Section 4.3 can be recomputed on the same numbers as the five-year one.

Usage: python3 regime_distance.py
Output: derived/regime_distance.csv          (per seed x family x year)
        derived/regime_distance_summary.csv  (per year, plus the correlations)
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
TRAIN_YEARS = list(range(2011, 2019))
ALL_YEARS = list(range(2011, 2024))
TEST_YEARS = list(range(2019, 2024))
SEEDS = [0, 1, 2, 3, 4]


def models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }


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

        fit = df[df["Provider CCN"].isin(seen_h) & df["year"].isin(TRAIN_YEARS)]
        unseen = df[df["Provider CCN"].isin(unseen_h)]
        sc = StandardScaler().fit(fit[FEATS])
        Xf, yf = sc.transform(fit[FEATS]), fit["deficit"].values

        train_pool = df[df["year"].isin(TRAIN_YEARS)]
        for mname, m in models(seed).items():
            # --- the deployed model, and what it realizes each year
            m.fit(Xf, yf)
            base_sub = unseen[unseen["year"].isin(TRAIN_YEARS)]
            base = roc_auc_score(base_sub["deficit"].values,
                                 m.predict_proba(sc.transform(base_sub[FEATS]))[:, 1])

            for y in ALL_YEARS:
                sub = unseen[unseen["year"] == y]
                auc = roc_auc_score(sub["deficit"].values,
                                    m.predict_proba(sc.transform(sub[FEATS]))[:, 1])

                # --- the monitor: can a discriminator tell year y from 2011-2018?
                yr = df[df["year"] == y]
                pool = train_pool[train_pool["year"] != y]
                take = min(len(yr), len(pool))
                a = yr.sample(take, random_state=seed)
                b = pool.sample(take, random_state=seed)
                X = np.vstack([a[FEATS].values, b[FEATS].values])
                lab = np.r_[np.ones(take), np.zeros(take)]
                idx = np.random.default_rng(seed).permutation(len(X))
                X, lab = X[idx], lab[idx]
                half = int(len(X) * 0.7)
                dsc = StandardScaler().fit(X[:half])
                d = models(seed)[mname]
                d.fit(dsc.transform(X[:half]), lab[:half])
                dist = roc_auc_score(lab[half:],
                                     d.predict_proba(dsc.transform(X[half:]))[:, 1])

                rows.append({"seed": seed, "model": mname, "year": y,
                             "regime_distance": dist, "auc": auc,
                             "baseline": base, "delta": auc - base,
                             "in_sample": y in TRAIN_YEARS, "n_year": len(yr)})

    res = pd.DataFrame(rows)
    res.to_csv(BASE / "regime_distance.csv", index=False)
    g = res.groupby("year").agg(regime_distance=("regime_distance", "mean"),
                                delta=("delta", "mean"))
    g["in_sample"] = [y in TRAIN_YEARS for y in g.index]
    g.to_csv(BASE / "regime_distance_summary.csv")
    print(g.round(4).to_string())

    oos = g[~g.in_sample]
    r, p = stats.pearsonr(oos.regime_distance, oos.delta)
    rho, prho = stats.spearmanr(g.regime_distance, g.delta)
    print(f"\nout-of-sample (2019-2023):  Pearson r = {r:+.3f} (p = {p:.3f})")
    print(f"all thirteen years:         Spearman rho = {rho:+.3f} (p = {prho:.3f})")
    print("\nwrote", BASE / "regime_distance.csv")


if __name__ == "__main__":
    main()
