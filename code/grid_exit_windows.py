"""DriftDriver - the exit-outcome row of Section 5.7, with a recorded definition.

The manuscript's exit row (E = +0.0008, T = +0.0655) has no script and no stored
output, and the panel's own `is_exiter` column cannot be what produced it: that
column is a hospital-level "eventually exits" flag, so a hospital that stops
reporting in 2022 carries a 1 on its 2011 row. Training on it leaks the future
into every year and its base rate falls monotonically from .132 (2011) to .000
(2023) for purely mechanical reasons.

This defines exit the way STATUS.md's own validation supports:

  - a hospital's exit year is its last observed reporting year L
  - only L <= 2019 counts as a real exit; STATUS.md records that pre-2020
    breaks have a 3.8-7.7% reappearance rate (real exits) while 2021-2022
    breaks are about 57% real, so those hospitals are dropped rather than
    labelled either way
  - row (hospital h, year t) is labelled 1 if h exits within HORIZON years
    of t, i.e. 0 <= L(h) - t <= HORIZON

Everything else - features, seeds, model families, cell construction, the E and
T definitions - is identical to grid_all_windows.py.

Usage: python3 grid_exit_windows.py
Output: derived/grid_exit_windows.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from grid_all_windows import FAMILIES, FEATS, SEEDS, marginals, run_spec

BASE = Path(__file__).resolve().parent.parent / "derived"
HORIZON = 2
LAST_RELIABLE_EXIT_YEAR = 2019

SPECS = [
    ("exit 2011-2016 -> 2017-2020", range(2011, 2017), range(2017, 2021)),
    ("exit 2011-2018 -> 2019-2023", range(2011, 2019), range(2019, 2024)),
]


def label_exit(df):
    last = df.groupby("Provider CCN")["year"].max()
    survivors = last[last == last.max()].index          # still reporting in the final year
    real_exits = last[last <= LAST_RELIABLE_EXIT_YEAR].index
    keep = df["Provider CCN"].isin(set(survivors) | set(real_exits))
    out = df[keep].copy()
    L = out["Provider CCN"].map(last)
    out["exit_soon"] = ((L - out["year"] >= 0) & (L - out["year"] <= HORIZON)
                        & out["Provider CCN"].isin(set(real_exits))).astype(int)
    print(f"  kept {out['Provider CCN'].nunique()} hospitals "
          f"({len(real_exits)} exiters, {len(survivors)} survivors); "
          f"dropped {df['Provider CCN'].nunique() - out['Provider CCN'].nunique()} "
          f"with ambiguous 2020-2022 breaks")
    print(f"  positive rate {out['exit_soon'].mean():.4f}; by year "
          f"{out.groupby('year')['exit_soon'].mean().round(4).to_dict()}")
    return out


def main():
    df = pd.read_parquet(BASE / "unbalanced_panel_2011_2023.parquet")
    df = label_exit(df)
    cells, summary = [], []
    for label, tr, te in SPECS:
        res = run_spec(df, "exit_soon", tr, te, label, "imbalanced")
        cells.append(res)
        ent, per = marginals(res)
        pe = stats.ttest_1samp(ent.values, 0.0).pvalue
        pt = stats.ttest_1samp(per.values, 0.0).pvalue
        flips = sum(per.xs(m).mean() > ent.xs(m).mean() for m in FAMILIES)
        summary.append({"spec": label, "E": ent.mean(), "p_E": pe,
                        "T": per.mean(), "p_T": pt, "T_gt_E_models": f"{flips}/3"})
        print(f"\n{label}   E={ent.mean():+.4f} (p={pe:.3g})   "
              f"T={per.mean():+.4f} (p={pt:.3g})   T>E in {flips}/3")
        for m in FAMILIES:
            print(f"    {m:7s} E={ent.xs(m).mean():+.4f}  T={per.xs(m).mean():+.4f}")

    pd.concat(cells, ignore_index=True).to_csv(BASE / "grid_exit_windows.csv", index=False)
    print("\nwrote", BASE / "grid_exit_windows.csv")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
