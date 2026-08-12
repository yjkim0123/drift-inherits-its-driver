"""DriftDriver — 2x2 novelty grid under two genuinely DISJOINT held-out windows.

Answers reviewer finding #6: Section 4.1 requires "at least two disjoint held-out
windows", but the paper's own two windows (2019-2023 and 2017-2020) overlap at
2019-2020, so the procedure did not meet its own stated requirement.

This runs the grid on a disjoint pair:

    Window A   train 2011-2016  ->  test 2017-2020   (regime-stable)
    Window B   train 2011-2018  ->  test 2021-2023   (contains the relief withdrawal)

Test years {2017,2018,2019,2020} and {2021,2022,2023} share nothing.
Everything else (features, seeds, model families, cell construction, the 20%
holdout that keeps the entity-seen/period-seen cell out of sample) is identical
to code/grid_2x2_cells.py, so the only quantity varying is the window.

Output: derived/grid_disjoint_windows.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent / "derived"
FEATS = ["labor_ratio", "overhead_ratio", "debt_ratio", "other_inc_ratio", "log_beds"]
SEEDS = [0, 1, 2, 3, 4]

WINDOWS = [
    ("A: 2011-2016 -> 2017-2020", list(range(2011, 2017)), list(range(2017, 2021))),
    ("B: 2011-2018 -> 2021-2023", list(range(2011, 2019)), list(range(2021, 2024))),
]


def models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }


def run(df, train_years, test_years, wlabel, panel):
    hosp = df["Provider CCN"].unique()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(hosp)
        cut = int(len(perm) * 0.7)
        seen_h, unseen_h = set(perm[:cut]), set(perm[cut:])

        tr_seen = df[df["Provider CCN"].isin(seen_h) & df["year"].isin(train_years)]
        idx = rng.permutation(len(tr_seen))
        hold = int(len(tr_seen) * 0.2)
        fit = tr_seen.iloc[idx[hold:]]
        cells = {
            ("E_seen", "P_seen"): tr_seen.iloc[idx[:hold]],
            ("E_unseen", "P_seen"): df[df["Provider CCN"].isin(unseen_h) & df["year"].isin(train_years)],
            ("E_seen", "P_unseen"): df[df["Provider CCN"].isin(seen_h) & df["year"].isin(test_years)],
            ("E_unseen", "P_unseen"): df[df["Provider CCN"].isin(unseen_h) & df["year"].isin(test_years)],
        }
        sc = StandardScaler().fit(fit[FEATS])
        Xf, yf = sc.transform(fit[FEATS]), fit["deficit"].values
        for mname, m in models(seed).items():
            m.fit(Xf, yf)
            for (e, p), sub in cells.items():
                auc = roc_auc_score(sub["deficit"].values,
                                    m.predict_proba(sc.transform(sub[FEATS]))[:, 1])
                rows.append({"window": wlabel, "panel": panel, "seed": seed,
                             "model": mname, "entity": e, "period": p,
                             "auc": auc, "n": len(sub)})
    return pd.DataFrame(rows)


def marginals(res):
    w = res.pivot_table(index=["model", "seed"], columns=["entity", "period"], values="auc")
    ent = ((w[("E_seen", "P_seen")] - w[("E_unseen", "P_seen")]) +
           (w[("E_seen", "P_unseen")] - w[("E_unseen", "P_unseen")])) / 2
    per = ((w[("E_seen", "P_seen")] - w[("E_seen", "P_unseen")]) +
           (w[("E_unseen", "P_seen")] - w[("E_unseen", "P_unseen")])) / 2
    per_model = {m: (ent.xs(m).mean(), per.xs(m).mean()) for m in ["LogReg", "RF", "GBM"]}
    return ent.mean(), per.mean(), per_model


def main():
    df = pd.read_parquet(BASE / "unbalanced_panel_2011_2023.parquet")
    df = df.dropna(subset=FEATS + ["deficit", "year", "Provider CCN"])
    allres = []
    for wlabel, tr, te in WINDOWS:
        res = run(df, tr, te, wlabel, "imbalanced")
        allres.append(res)
        e, p, pm = marginals(res)
        print(f"\n=== {wlabel} ===  (test years {te[0]}-{te[-1]})")
        print(f"  entity cost E = {e:+.4f}   period cost T = {p:+.4f}   T > E: {p > e}")
        for m, (em, pmv) in pm.items():
            print(f"    {m:7s} E {em:+.4f}  T {pmv:+.4f}  T>E: {pmv > em}")
    out = pd.concat(allres, ignore_index=True)
    out.to_csv(BASE / "grid_disjoint_windows.csv", index=False)
    print("\nwrote", BASE / "grid_disjoint_windows.csv")


if __name__ == "__main__":
    main()
