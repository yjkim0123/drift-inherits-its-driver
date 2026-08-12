"""DriftDriver - every 2x2 novelty-grid window, both outcomes, ONE run.

Replaces the two separate tables in Section 5.7, which reported the same
specification (deficit, 2011-2016 -> 2017-2020) twice with different numbers
because the rows came from two different runs. Everything the manuscript
quotes for the window-artifact argument is produced here, in one process,
from one panel load, with one definition of E and T.

E (entity cost) and T (period cost) are the marginal AUC costs of the 2x2:
    E = mean over the two period levels of  AUC(entity seen) - AUC(entity unseen)
    T = mean over the two entity levels of  AUC(period seen) - AUC(period unseen)
Each is computed per (model family, seed), giving 15 values per spec; the
reported figure is their mean and the p-value is a two-sided one-sample
t-test of those 15 values against zero.

Usage: python3 grid_all_windows.py
Output: derived/grid_all_windows.csv   (cell-level AUCs, every spec)
        derived/grid_all_windows_marginals.csv  (E, T, p-values, per-family)
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
FAMILIES = ["LogReg", "RF", "GBM"]

# (label, outcome column, train years, test years)
SPECS = [
    ("deficit 2011-2018 -> 2019-2023", "deficit", range(2011, 2019), range(2019, 2024)),
    ("deficit 2011-2016 -> 2017-2020", "deficit", range(2011, 2017), range(2017, 2021)),
    ("deficit 2011-2018 -> 2021-2023", "deficit", range(2011, 2019), range(2021, 2024)),
    ("exit    2011-2016 -> 2017-2020", "is_exiter", range(2011, 2017), range(2017, 2021)),
]

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


def run_spec(df, outcome, train_years, test_years, label, panel):
    """Identical cell construction to grid_2x2_cells.py; only the window varies."""
    sub_df = df.dropna(subset=FEATS + [outcome, "year", "Provider CCN"])
    train_years, test_years = list(train_years), list(test_years)
    hosp = sub_df["Provider CCN"].unique()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(hosp)
        cut = int(len(perm) * 0.7)
        seen_h, unseen_h = set(perm[:cut]), set(perm[cut:])

        tr_seen = sub_df[sub_df["Provider CCN"].isin(seen_h) & sub_df["year"].isin(train_years)]
        idx = rng.permutation(len(tr_seen))
        hold = int(len(tr_seen) * 0.2)
        fit = tr_seen.iloc[idx[hold:]]
        cells = {
            ("E_seen", "P_seen"): tr_seen.iloc[idx[:hold]],
            ("E_unseen", "P_seen"): sub_df[sub_df["Provider CCN"].isin(unseen_h) & sub_df["year"].isin(train_years)],
            ("E_seen", "P_unseen"): sub_df[sub_df["Provider CCN"].isin(seen_h) & sub_df["year"].isin(test_years)],
            ("E_unseen", "P_unseen"): sub_df[sub_df["Provider CCN"].isin(unseen_h) & sub_df["year"].isin(test_years)],
        }
        sc = StandardScaler().fit(fit[FEATS])
        Xf, yf = sc.transform(fit[FEATS]), fit[outcome].values
        for mname, m in models(seed).items():
            m.fit(Xf, yf)
            for (e, p), cell in cells.items():
                auc = roc_auc_score(cell[outcome].values,
                                    m.predict_proba(sc.transform(cell[FEATS]))[:, 1])
                rows.append({"spec": label, "panel": panel, "outcome": outcome,
                             "seed": seed, "model": mname, "entity": e, "period": p,
                             "auc": auc, "n": len(cell)})
    return pd.DataFrame(rows)


def marginals(res):
    """Return per-(model,seed) E and T series."""
    w = res.pivot_table(index=["model", "seed"], columns=["entity", "period"], values="auc")
    ent = ((w[("E_seen", "P_seen")] - w[("E_unseen", "P_seen")]) +
           (w[("E_seen", "P_unseen")] - w[("E_unseen", "P_unseen")])) / 2
    per = ((w[("E_seen", "P_seen")] - w[("E_seen", "P_unseen")]) +
           (w[("E_unseen", "P_seen")] - w[("E_unseen", "P_unseen")])) / 2
    return ent, per


def main():
    cells, summary = [], []
    for panel, fname in PANELS:
        df = pd.read_parquet(BASE / fname)
        for label, outcome, tr, te in SPECS:
            # the balanced panel is survivors only, so it carries no exit column
            if outcome not in df.columns:
                print(f"[{panel:10s}] {label}   skipped: no '{outcome}' column")
                continue
            res = run_spec(df, outcome, tr, te, label, panel)
            cells.append(res)
            ent, per = marginals(res)
            pe = stats.ttest_1samp(ent.values, 0.0).pvalue
            pt = stats.ttest_1samp(per.values, 0.0).pvalue
            flips = sum(per.xs(m).mean() > ent.xs(m).mean() for m in FAMILIES)
            summary.append({"panel": panel, "spec": label, "outcome": outcome,
                            "E": ent.mean(), "p_E": pe, "T": per.mean(), "p_T": pt,
                            "T_gt_E_models": f"{flips}/3"})
            print(f"[{panel:10s}] {label}   E={ent.mean():+.4f} (p={pe:.3g})   "
                  f"T={per.mean():+.4f} (p={pt:.3g})   T>E in {flips}/3")
            for m in FAMILIES:
                print(f"             {m:7s} E={ent.xs(m).mean():+.4f}  T={per.xs(m).mean():+.4f}")

    pd.concat(cells, ignore_index=True).to_csv(BASE / "grid_all_windows.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(BASE / "grid_all_windows_marginals.csv", index=False)
    print("\nwrote", BASE / "grid_all_windows.csv")
    print("wrote", BASE / "grid_all_windows_marginals.csv")
    print("\n===== manuscript table (imbalanced panel) =====")
    print(summ[summ.panel == "imbalanced"].to_string(index=False))


if __name__ == "__main__":
    main()
