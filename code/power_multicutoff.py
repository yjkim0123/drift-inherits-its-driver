"""DriftDriver — power domain: multi-cutoff identification test.

Answers reviewer finding #4: with a single training cutoff, "degradation is
monotone in elapsed time" is not identified against "degradation is monotone in
calendar year".  Fixing the calendar year and varying only model age separates
the two.

Design
------
Data      KPX national hourly demand, 2013-01-01 .. 2025-12-31 (13 years).
          Files live in ~/Documents/project_power/data/.  NOTE the file named
          power_2020.csv actually holds 2013-2020 (filename trap).
Features  Calendar + autoregressive only.  No weather: the Open-Meteo pull
          covers 2021-2025 only, so weather features are unavailable for the
          2013-2020 training windows and for the 2019-2020 test years.
          FEATURES = hour, dow, month, is_weekend, doy, lag1, lag24, lag168, roll24
Target    Hourly national demand (MW).
Models    XGBoost (primary, matches project_power/model_multiyear.py) and
          Linear (secondary, the patent baseline).
Metric    MAPE (%).  Reported degradation is in percentage POINTS of MAPE:
              delta_pp(Y) = MAPE(test year Y) - MAPE(in-era validation)
Baseline  In-era validation = random 10% of the training-window rows, held out
          before fitting (seed 42).  Same protocol for every cutoff, so the
          three columns are comparable.

Cutoffs   train through 2016 / 2018 / 2020; every later year tested separately.

Outputs   derived/power_multicutoff.csv   (long: cutoff, test_year, elapsed, ...)
          derived/power_multicutoff.json  (correlations + identification table)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

POWER = Path.home() / "Documents" / "project_power"
OUT = Path(__file__).resolve().parent.parent / "derived"
SEED = 42
CUTOFFS = [2016, 2018, 2020]
FEATURES = ["hour", "dow", "month", "is_weekend", "doy",
            "lag1", "lag24", "lag168", "roll24"]

FILES = [
    POWER / "data/multiyear/power_2020.csv",   # 2013-2020
    POWER / "data/multiyear/power_2021.csv",
    POWER / "data/multiyear/power_2022.csv",
    POWER / "data/multiyear/power_2023.csv",
    POWER / "data/multiyear/power_2024.csv",
    POWER / "data/power_demand.csv",           # 2025
]


def load_year(path):
    df = pd.read_csv(path).dropna(subset=["날짜"])
    df = df[df["날짜"].astype(str).str.strip() != ""]
    long = df.melt(id_vars="날짜", var_name="hs", value_name="demand")
    long["hour"] = long["hs"].str.replace("시", "", regex=False).astype(int) - 1
    long["date"] = pd.to_datetime(long["날짜"], errors="coerce")
    long = long.dropna(subset=["date"])
    long["demand"] = pd.to_numeric(long["demand"], errors="coerce")
    long = long.dropna(subset=["demand"])
    long["dt"] = long["date"] + pd.to_timedelta(long["hour"], unit="h")
    return long[["dt", "date", "demand"]]


def build_panel():
    d = pd.concat([load_year(p) for p in FILES], ignore_index=True)
    d = d.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    # continuity check: hourly series with no holes
    gaps = d["dt"].diff().dropna()
    n_gap = int((gaps != pd.Timedelta(hours=1)).sum())
    d["year"] = d["date"].dt.year
    d["hour"] = d["dt"].dt.hour
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["doy"] = d["date"].dt.dayofyear
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["lag1"] = d["demand"].shift(1)
    d["lag24"] = d["demand"].shift(24)
    d["lag168"] = d["demand"].shift(168)
    d["roll24"] = d["demand"].shift(1).rolling(24).mean()
    d = d.dropna().reset_index(drop=True)
    return d, n_gap


def mape(yt, p):
    return float(np.mean(np.abs((yt - p) / yt)) * 100)


def fit_predict(Xtr, ytr, Xte):
    xg = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.9, random_state=SEED, verbosity=0)
    xg.fit(Xtr, ytr)
    s = StandardScaler().fit(Xtr)
    lin = LinearRegression().fit(s.transform(Xtr), ytr)
    return xg.predict(Xte), lin.predict(s.transform(Xte))


def main():
    d, n_gap = build_panel()
    print(f"panel: {len(d)} hours | {d['dt'].min()} .. {d['dt'].max()}")
    print(f"years: {sorted(d['year'].unique())}")
    print(f"non-hourly steps in series: {n_gap}")

    X = d[FEATURES].values.astype(float)
    y = d["demand"].values
    years = d["year"].values
    max_year = int(years.max())

    rows = []
    rng = np.random.default_rng(SEED)

    for C in CUTOFFS:
        tr_all = years <= C
        idx = np.where(tr_all)[0]
        holdout = rng.choice(idx, size=int(round(0.1 * len(idx))), replace=False)
        fit_mask = np.zeros(len(d), bool)
        fit_mask[idx] = True
        fit_mask[holdout] = False

        Xfit, yfit = X[fit_mask], y[fit_mask]
        px_val, pl_val = fit_predict(Xfit, yfit, X[holdout])
        base_xgb = mape(y[holdout], px_val)
        base_lin = mape(y[holdout], pl_val)
        print(f"\n[cutoff <= {C}] fit n={fit_mask.sum()} | in-era val n={len(holdout)} "
              f"| baseline MAPE XGB {base_xgb:.3f}% Lin {base_lin:.3f}%")

        for Y in range(C + 1, max_year + 1):
            te = years == Y
            if te.sum() < 1000:
                continue
            px, pl = fit_predict(Xfit, yfit, X[te])
            m_xgb, m_lin = mape(y[te], px), mape(y[te], pl)
            rows.append({
                "cutoff": C, "test_year": Y, "elapsed": Y - C,
                "n_test": int(te.sum()),
                "mape_xgb": round(m_xgb, 4),
                "delta_pp_xgb": round(m_xgb - base_xgb, 4),
                "r2_xgb": round(float(r2_score(y[te], px)), 4),
                "mape_lin": round(m_lin, 4),
                "delta_pp_lin": round(m_lin - base_lin, 4),
                "baseline_mape_xgb": round(base_xgb, 4),
                "baseline_mape_lin": round(base_lin, 4),
            })
            print(f"  test {Y} (elapsed {Y-C}): XGB MAPE {m_xgb:.3f}% "
                  f"delta {m_xgb-base_xgb:+.3f}pp | Lin {m_lin:.3f}% "
                  f"delta {m_lin-base_lin:+.3f}pp")

    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "power_multicutoff.csv", index=False)

    # (1) monotonicity in elapsed time, within each cutoff
    corr = {}
    for C in CUTOFFS:
        s = res[res.cutoff == C]
        for tag, col in [("xgb", "delta_pp_xgb"), ("lin", "delta_pp_lin")]:
            r, p = pearsonr(s["elapsed"], s[col])
            corr[f"cutoff_{C}_{tag}"] = {"r": round(float(r), 4),
                                         "p": float(p), "n": int(len(s))}

    # (2) identification: fix calendar year, vary model age
    shared = sorted(set.intersection(*[set(res[res.cutoff == C].test_year) for C in CUTOFFS]))
    ident = []
    for Y in shared:
        row = {"calendar_year": int(Y)}
        for C in CUTOFFS:
            v = res[(res.cutoff == C) & (res.test_year == Y)].iloc[0]
            row[f"train_to_{C}"] = float(v.delta_pp_xgb)
            row[f"elapsed_{C}"] = int(v.elapsed)
        row["oldest_over_newest"] = (round(row[f"train_to_{CUTOFFS[0]}"] /
                                           row[f"train_to_{CUTOFFS[-1]}"], 3)
                                     if row[f"train_to_{CUTOFFS[-1]}"] else None)
        ident.append(row)

    # (3) same elapsed age, different calendar year -> capacity confound
    same_age = {}
    for e in sorted(res.elapsed.unique()):
        s = res[res.elapsed == e]
        if len(s) > 1:
            same_age[int(e)] = {int(r.cutoff): float(r.delta_pp_xgb)
                                for r in s.itertuples()}

    out = {
        "spec": {
            "data": "KPX national hourly demand 2013-2025",
            "features": FEATURES,
            "weather": "none (Open-Meteo pull covers 2021-2025 only)",
            "target": "hourly national demand (MW)",
            "metric": "MAPE (%); degradation reported in percentage points",
            "baseline": "random 10% in-era holdout of the training window, seed 42",
            "models": {"primary": "XGBoost(n=500, depth=6, lr=0.05, subsample=0.9)",
                       "secondary": "LinearRegression on standardized features"},
            "seed": SEED,
        },
        "panel_hours": int(len(d)),
        "non_hourly_steps": n_gap,
        "elapsed_correlation": corr,
        "identification_table": ident,
        "same_elapsed_across_cutoffs": same_age,
    }
    (OUT / "power_multicutoff.json").write_text(json.dumps(out, indent=2))

    print("\n=== elapsed-time correlation (XGB) ===")
    for C in CUTOFFS:
        c = corr[f"cutoff_{C}_xgb"]
        print(f"  train<={C}: r = {c['r']:+.4f} (p = {c['p']:.2e}, n = {c['n']})")
    print("\n=== identification: same calendar year, different model age (XGB, pp) ===")
    print(pd.DataFrame(ident).to_string(index=False))
    print("\n=== same elapsed age across cutoffs (XGB, pp) ===")
    for e, v in same_age.items():
        print(f"  elapsed {e}: " + " | ".join(f"train<={c} {x:+.3f}" for c, x in v.items()))
    print("\nwrote", OUT / "power_multicutoff.csv")
    print("wrote", OUT / "power_multicutoff.json")


if __name__ == "__main__":
    main()
