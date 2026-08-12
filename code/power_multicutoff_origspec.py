"""Power multi-cutoff under the ORIGINAL spec recovered from code/make_figures.py:151-171.
target = load / year-mean load ; features = hour,month,dow,doy ; GBR(random_state=0)
baseline = mean per-year MAPE over the TRAINING years (in-sample).

THIS is the script behind the Section 5.4 degradation and identification tables.
code/power_multicutoff.py is a DIFFERENT specification (XGB, other features) and its
derived/power_multicutoff.csv carries materially different numbers (2021 at cutoff 2016
is +0.81 pp there against +1.87 pp here). Do not read that file as the source for
Section 5.4. Outputs here are written to derived/power_multicutoff_origspec.csv."""
import glob, os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
POWER = os.path.expanduser('~/Documents/project_power/data')
PFE = ['hour', 'month', 'dow', 'doy']

files = sorted(glob.glob(os.path.join(POWER, 'multiyear', '*.csv'))) + [os.path.join(POWER, 'power_demand.csv')]
a = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
a['dt'] = pd.to_datetime(a['날짜'], errors='coerce')
a = a.dropna(subset=['dt']).drop_duplicates(subset=['dt']).sort_values('dt')
hrs = [c for c in a.columns if c.endswith('시')]
L = a.melt(id_vars=['dt'], value_vars=hrs, var_name='hr', value_name='load')
L['hour'] = L['hr'].str.replace('시', '').astype(int)
L['year'] = L.dt.dt.year; L['month'] = L.dt.dt.month
L['dow'] = L.dt.dt.dayofweek; L['doy'] = L.dt.dt.dayofyear
L['norm'] = L['load'] / L.groupby('year')['load'].transform('mean')

res = {}
for C in [2016, 2018, 2020]:
    tr = L[L.year <= C]
    m = GradientBoostingRegressor(random_state=0).fit(tr[PFE], tr['norm'])
    ape = np.abs((L['norm'] - m.predict(L[PFE])) / L['norm']) * 100
    per = {y: ape[L.year == y].mean() for y in range(2013, 2026)}
    base = np.mean([per[y] for y in range(2013, C + 1)])
    ys = list(range(C + 1, 2026))
    d = {y: per[y] - base for y in ys}
    r, p = stats.pearsonr([y - C for y in ys], [d[y] for y in ys])
    res[C] = d
    print(f'[train<={C}] baseline MAPE {base:.3f}%  r={r:+.4f} (p={p:.2e}, n={len(ys)})')
    for y in ys:
        print(f'    {y} (elapsed {y-C}): {d[y]:+.3f}pp')

print('\n=== identification table (pp) ===')
print(f'{"year":>6} {"~2016":>9} {"~2018":>9} {"~2020":>9}   ratio')
for y in range(2021, 2026):
    v = [res[C][y] for C in [2016, 2018, 2020]]
    print(f'{y:>6} {v[0]:>+9.2f} {v[1]:>+9.2f} {v[2]:>+9.2f}   {v[0]/v[2]:.2f}x')
print('\n=== same elapsed age across cutoffs (pp) ===')
for e in range(1, 10):
    row = {C: res[C].get(C + e) for C in [2016, 2018, 2020] if (C + e) in res[C]}
    if len(row) > 1:
        print(f'  elapsed {e}: ' + ' | '.join(f'~{C} {v:+.2f}' for C, v in row.items()))

OUT = str(Path(__file__).resolve().parent.parent / "derived" / "power_multicutoff_origspec.csv")
pd.DataFrame([{'cutoff': C, 'test_year': y, 'elapsed': y - C, 'delta_pp': d}
              for C, ys in res.items() for y, d in ys.items()]).to_csv(OUT, index=False)
print('\nwrote', OUT)
