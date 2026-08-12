"""DriftDriver — power domain: Step (b), transfer versus contemporaneous training.

Answers reviewer finding #2: Section 2.1 claims both domains are concept drift
"established in Section 5.2", but Section 5.2 is hospital-only.  The electricity
domain had never been given the definitional test.  This script gives it.

Design mirrors the hospital Step (b) as closely as the data allows:

  Arm A (transfer)         train on years <= 2018, predict year Y
  Arm B (contemporaneous)  train on year Y itself, predict year Y

The hospital version keeps the two arms honest by splitting on entities.
Electricity has no entity axis, so we split each year by ISO week parity:
arm B trains on odd weeks of Y, and BOTH arms are scored on the even weeks of Y.
Identical test rows for both arms; no hour is ever in arm B's train and test.
Week-block splitting (rather than random hours) keeps the daily and weekly
autocorrelation out of the contemporaneous arm's favour.

Specification is the main-table one, recovered from code/make_figures.py:151-171:
target = load / that year's mean load, features = hour/month/dow/doy,
GradientBoostingRegressor(random_state=0), metric = MAPE on normalized load.

Reading: if arm B recovers normal-period error while arm A degrades, the
calendar->shape mapping itself moved, which is concept drift in P(y|x).  If both
arms degrade together, the shape has become less predictable from the calendar
at all, which would be signal loss rather than a moved mapping.

Output: derived/power_contemporaneous.csv
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

POWER = os.path.expanduser('~/Documents/project_power/data')
OUT = Path(__file__).resolve().parent.parent / 'derived'
PFE = ['hour', 'month', 'dow', 'doy']
TRAIN_END = 2018


def load_power():
    files = sorted(glob.glob(os.path.join(POWER, 'multiyear', '*.csv'))) + \
            [os.path.join(POWER, 'power_demand.csv')]
    a = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    a['dt'] = pd.to_datetime(a['날짜'], errors='coerce')
    a = a.dropna(subset=['dt']).drop_duplicates(subset=['dt']).sort_values('dt')
    hrs = [c for c in a.columns if c.endswith('시')]
    L = a.melt(id_vars=['dt'], value_vars=hrs, var_name='hr', value_name='load')
    L['hour'] = L['hr'].str.replace('시', '', regex=False).astype(int)
    L['year'] = L.dt.dt.year
    L['month'] = L.dt.dt.month
    L['dow'] = L.dt.dt.dayofweek
    L['doy'] = L.dt.dt.dayofyear
    L['week'] = L.dt.dt.isocalendar().week.astype(int)
    L['norm'] = L['load'] / L.groupby('year')['load'].transform('mean')
    return L.dropna(subset=['load']).reset_index(drop=True)


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def main():
    L = load_power()
    even = L['week'] % 2 == 0

    # arm A: one model trained on <= 2018, scored on each year's even weeks
    trA = L[L.year <= TRAIN_END]
    mA = GradientBoostingRegressor(random_state=0).fit(trA[PFE], trA['norm'])

    rows = []
    for Y in range(TRAIN_END + 1, 2026):
        teY = L[(L.year == Y) & even]
        trB = L[(L.year == Y) & ~even]
        mB = GradientBoostingRegressor(random_state=0).fit(trB[PFE], trB['norm'])
        a = mape(teY['norm'].values, mA.predict(teY[PFE]))
        b = mape(teY['norm'].values, mB.predict(teY[PFE]))
        rows.append({'year': Y, 'elapsed': Y - TRAIN_END, 'n_test': len(teY),
                     'mape_transfer': round(a, 4),
                     'mape_contemporaneous': round(b, 4),
                     'recovered_pp': round(a - b, 4)})
        print(f'  {Y} (elapsed {Y-TRAIN_END}): transfer {a:.3f}%  '
              f'contemporaneous {b:.3f}%  recovered {a-b:+.3f} pp')

    # in-era reference: same protocol inside the training era
    ref = []
    for Y in range(2013, TRAIN_END + 1):
        teY = L[(L.year == Y) & even]
        trB = L[(L.year == Y) & ~even]
        mB = GradientBoostingRegressor(random_state=0).fit(trB[PFE], trB['norm'])
        ref.append(mape(teY['norm'].values, mB.predict(teY[PFE])))
    ref_mean = float(np.mean(ref))
    print(f'\nin-era contemporaneous reference (2013-{TRAIN_END}): {ref_mean:.3f}%')

    res = pd.DataFrame(rows)
    res['contemporaneous_vs_reference_pp'] = (res['mape_contemporaneous'] - ref_mean).round(4)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / 'power_contemporaneous.csv', index=False)

    r1, p1 = stats.pearsonr(res['elapsed'], res['recovered_pp'])
    r2, p2 = stats.pearsonr(res['elapsed'], res['contemporaneous_vs_reference_pp'])
    print(f'\nrecovered pp vs elapsed:                r={r1:+.3f} (p={p1:.2e})')
    print(f'contemporaneous drift vs elapsed:       r={r2:+.3f} (p={p2:.2e})')
    frac = (res['recovered_pp'] / (res['mape_transfer'] - ref_mean)).round(3)
    print('fraction of the transfer gap recovered by retraining:',
          list(zip(res['year'], frac)))
    print('\nwrote', OUT / 'power_contemporaneous.csv')


if __name__ == '__main__':
    main()
