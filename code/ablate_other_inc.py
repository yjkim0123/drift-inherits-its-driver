"""Ablation: is the 2021 pit an artifact of including `other_inc_ratio`?

`other_inc_ratio` is simultaneously (a) a model feature and (b) the exact
variable that federal COVID-19 relief moved. If the 2021 degradation pit is
driven by the model leaning on a feature the policy rewrote, then dropping the
feature should shrink the pit. If the pit survives, the regime-shift claim is
not a feature-choice artifact.

Design mirrors the primary analysis: imbalanced panel, train 2011-2018 on a
70% hospital split, evaluate per test year on held-out (unseen) hospitals,
5 seeds x 3 model families. Baseline = mean AUC on the training years for the
same unseen hospitals, so the per-year delta is comparable across arms.

Usage: python3 ablate_other_inc.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent / "derived")
FULL = ["labor_ratio", "overhead_ratio", "debt_ratio", "other_inc_ratio", "log_beds"]
ABLATED = [f for f in FULL if f != "other_inc_ratio"]
TRAIN_YEARS = list(range(2011, 2019))
TEST_YEARS = list(range(2019, 2024))
SEEDS = [0, 1, 2, 3, 4]


def models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }


def run(df, feats, arm):
    hosp = df["Provider CCN"].unique()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(hosp)
        cut = int(len(perm) * 0.7)
        seen_h, unseen_h = set(perm[:cut]), set(perm[cut:])

        fit = df[df["Provider CCN"].isin(seen_h) & df["year"].isin(TRAIN_YEARS)]
        unseen = df[df["Provider CCN"].isin(unseen_h)]

        sc = StandardScaler().fit(fit[feats])
        Xf, yf = sc.transform(fit[feats]), fit["deficit"].values
        for mname, m in models(seed).items():
            m.fit(Xf, yf)

            base_sub = unseen[unseen["year"].isin(TRAIN_YEARS)]
            base = roc_auc_score(base_sub["deficit"].values,
                                 m.predict_proba(sc.transform(base_sub[feats]))[:, 1])
            for yr in TEST_YEARS:
                sub = unseen[unseen["year"] == yr]
                auc = roc_auc_score(sub["deficit"].values,
                                    m.predict_proba(sc.transform(sub[feats]))[:, 1])
                rows.append({"arm": arm, "seed": seed, "model": mname, "year": yr,
                             "auc": auc, "baseline": base, "delta": auc - base})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_parquet(f"{BASE}/unbalanced_panel_2011_2023.parquet").dropna(
        subset=FULL + ["deficit", "year", "Provider CCN"])

    res = pd.concat([run(df, FULL, "full"), run(df, ABLATED, "ablated")],
                    ignore_index=True)
    res.to_csv(f"{BASE}/ablate_other_inc.csv", index=False)

    print(f"baseline (train-year AUC on unseen hospitals): "
          f"full {res[res.arm=='full'].baseline.mean():.4f}  "
          f"ablated {res[res.arm=='ablated'].baseline.mean():.4f}")
    print("\n year |   full delta        |  ablated delta      | pit retained")
    print("-" * 68)
    for yr in TEST_YEARS:
        f = res[(res.arm == "full") & (res.year == yr)].delta
        a = res[(res.arm == "ablated") & (res.year == yr)].delta
        pf = stats.ttest_1samp(f, 0).pvalue
        pa = stats.ttest_1samp(a, 0).pvalue
        keep = (a.mean() / f.mean() * 100) if f.mean() != 0 else float("nan")
        print(f" {yr} | {f.mean():+.4f} (p={pf:.1e}) | {a.mean():+.4f} (p={pa:.1e}) "
              f"| {keep:6.1f}%")

    print("\nper-model 2021 delta (sign consistency check):")
    for mname in ["LogReg", "RF", "GBM"]:
        f = res[(res.arm == "full") & (res.year == 2021) & (res.model == mname)].delta.mean()
        a = res[(res.arm == "ablated") & (res.year == 2021) & (res.model == mname)].delta.mean()
        print(f"   {mname:7s} full {f:+.4f}   ablated {a:+.4f}")

    # non-monotonicity: is the pit still a pit after ablation?
    ab = res[res.arm == "ablated"].groupby("year").delta.mean()
    print(f"\nablated arm elapsed-time correlation: "
          f"r = {stats.pearsonr(ab.index.values, ab.values)[0]:+.3f} "
          f"(p = {stats.pearsonr(ab.index.values, ab.values)[1]:.3f})")
    print(f"ablated arm worst year = {ab.idxmin()} ({ab.min():+.4f}), "
          f"2023 = {ab.loc[2023]:+.4f}")
