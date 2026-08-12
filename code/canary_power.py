"""DriftDriver - Section 5.6, the labeled-canary power curve, with a stored output.

Table 4 of the manuscript (power by audit-sample size, thresholds calibrated to
a 5% in-window false-alarm rate) existed only in STATUS.md: no script, no CSV.
This is that analysis written down.

Procedure, exactly as Section 4.4 states it:
  - the deployed model is the one of Section 5.1 (fit on seen hospitals,
    2011-2018), and the sampling frame is entity set B, the unseen hospitals
  - tau_n is the 5th percentile of the sampling distribution of the audit-sample
    AUC drawn WITHIN the stable training window, so the in-window false-alarm
    rate is 5% by construction
  - power at period t is the share of audit samples of size n drawn from year t
    whose AUC falls below tau_n
  - 2,000 bootstrap resamples per cell, per (seed, model family); the reported
    figure averages the 15 runs

Usage: python3 canary_power.py
Output: derived/canary_power.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent / "derived"
FEATS = ["labor_ratio", "overhead_ratio", "debt_ratio", "other_inc_ratio", "log_beds"]
TRAIN_YEARS = list(range(2011, 2019))
TEST_YEARS = list(range(2019, 2024))
SEEDS = [0, 1, 2, 3, 4]
NS = [25, 50, 100, 200, 400, 800]
B = 2000


def models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }


def boot_auc(score, label, n, rng, reps=B):
    """AUC of `reps` bootstrap audit samples of size n, vectorised.

    Samples with only one class present carry no AUC and are dropped, which is
    what an auditor would see: a sample of 25 hospitals with no deficit cannot
    raise a calibrated alarm.
    """
    idx = rng.integers(0, len(score), size=(reps, n))
    s, y = score[idx], label[idx]
    pos = y.sum(axis=1)
    ok = (pos > 0) & (pos < n)
    s, y, pos = s[ok], y[ok], pos[ok]
    order = np.argsort(s, axis=1)
    ranks = np.empty_like(order, dtype=float)
    np.put_along_axis(ranks, order, np.arange(1, n + 1, dtype=float)[None, :], axis=1)
    rsum = (ranks * y).sum(axis=1)
    neg = n - pos
    return (rsum - pos * (pos + 1) / 2) / (pos * neg)


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
        Xu = sc.transform(unseen[FEATS])

        for mname, m in models(seed).items():
            m.fit(Xf, yf)
            score = m.predict_proba(Xu)[:, 1]
            u = unseen.assign(score=score)
            win = u[u.year.isin(TRAIN_YEARS)]
            for n in NS:
                brng = np.random.default_rng(1000 + seed)
                cal = boot_auc(win.score.values, win.deficit.values.astype(int), n, brng)
                tau = np.percentile(cal, 5)
                for y in TEST_YEARS:
                    yr = u[u.year == y]
                    a = boot_auc(yr.score.values, yr.deficit.values.astype(int), n, brng)
                    rows.append({"seed": seed, "model": mname, "n": n, "tau": tau,
                                 "year": y, "alarm_rate": float((a < tau).mean()),
                                 "n_frame": len(yr)})

    res = pd.DataFrame(rows)
    res.to_csv(BASE / "canary_power.csv", index=False)
    tab = res.pivot_table(index="n", columns="year", values="alarm_rate")
    tau = res.groupby("n").tau.mean()
    out = pd.concat([tau.rename("tau"), (tab * 100).round(1)], axis=1)
    print(out.to_string())
    print("\nentity-B rows per year:", int(res.n_frame.mean()))
    print("wrote", BASE / "canary_power.csv")


if __name__ == "__main__":
    main()
