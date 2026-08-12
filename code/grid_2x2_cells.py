"""2x2 novelty grid (entity known/unknown x period known/unknown), cell-level AUC.

Runs on both panels so the imbalanced-panel marginals recorded in STATUS.md
(+0.008 entity / +0.046 period) can be reproduced before the cell means are used.

Usage: python3 grid_2x2_cells.py
Output: prints cell AUC mean+-sd per model per panel, and the two marginal costs.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent / "derived")
FEATS = ["labor_ratio", "overhead_ratio", "debt_ratio", "other_inc_ratio", "log_beds"]
TRAIN_YEARS = list(range(2011, 2019))   # 2011-2018
TEST_YEARS = list(range(2019, 2024))    # 2019-2023
SEEDS = [0, 1, 2, 3, 4]


def models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=seed),
    }


def run_panel(path, label):
    df = pd.read_parquet(path).dropna(subset=FEATS + ["deficit", "year", "Provider CCN"])
    hosp = df["Provider CCN"].unique()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(hosp)
        cut = int(len(perm) * 0.7)
        seen_h, unseen_h = set(perm[:cut]), set(perm[cut:])

        tr_seen = df[df["Provider CCN"].isin(seen_h) & df["year"].isin(TRAIN_YEARS)]
        # hold out 20% of rows of the seen/train cell so "entity seen, period seen"
        # is still an out-of-sample estimate rather than a training-set fit
        idx = rng.permutation(len(tr_seen))
        hold = int(len(tr_seen) * 0.2)
        fit = tr_seen.iloc[idx[hold:]]
        cells = {
            ("E_seen", "P_seen"): tr_seen.iloc[idx[:hold]],
            ("E_unseen", "P_seen"): df[df["Provider CCN"].isin(unseen_h) & df["year"].isin(TRAIN_YEARS)],
            ("E_seen", "P_unseen"): df[df["Provider CCN"].isin(seen_h) & df["year"].isin(TEST_YEARS)],
            ("E_unseen", "P_unseen"): df[df["Provider CCN"].isin(unseen_h) & df["year"].isin(TEST_YEARS)],
        }

        sc = StandardScaler().fit(fit[FEATS])
        Xf, yf = sc.transform(fit[FEATS]), fit["deficit"].values
        for mname, m in models(seed).items():
            m.fit(Xf, yf)
            for (e, p), sub in cells.items():
                auc = roc_auc_score(sub["deficit"].values,
                                    m.predict_proba(sc.transform(sub[FEATS]))[:, 1])
                rows.append({"panel": label, "seed": seed, "model": mname,
                             "entity": e, "period": p, "auc": auc, "n": len(sub)})
    return pd.DataFrame(rows)


def report(res, label):
    print(f"\n===== {label} =====")
    g = res.groupby(["model", "entity", "period"])["auc"].agg(["mean", "std"])
    for mname in ["LogReg", "RF", "GBM"]:
        print(f"\n-- {mname}")
        for e in ["E_seen", "E_unseen"]:
            for p in ["P_seen", "P_unseen"]:
                m, s = g.loc[(mname, e, p)]
                print(f"   {e:9s} {p:9s}  {m:.3f} +- {s:.3f}")
    w = res.pivot_table(index=["model", "seed"], columns=["entity", "period"], values="auc")
    ent = ((w[("E_seen", "P_seen")] - w[("E_unseen", "P_seen")]) +
           (w[("E_seen", "P_unseen")] - w[("E_unseen", "P_unseen")])) / 2
    per = ((w[("E_seen", "P_seen")] - w[("E_seen", "P_unseen")]) +
           (w[("E_unseen", "P_seen")] - w[("E_unseen", "P_unseen")])) / 2
    print(f"\n  marginal entity cost = +{ent.mean():.3f}   "
          f"marginal period cost = +{per.mean():.3f}   ratio = {per.mean()/ent.mean():.1f}x")


if __name__ == "__main__":
    for path, label in [(f"{BASE}/unbalanced_panel_2011_2023.parquet", "IMBALANCED (entry/exit allowed)"),
                        (f"{BASE}/balanced_panel_2011_2023.parquet", "BALANCED (survivors only)")]:
        res = run_panel(path, label)
        report(res, label)
        res.to_csv(f"{BASE}/grid_2x2_cells_{'imbal' if 'IMB' in label else 'bal'}.csv", index=False)
