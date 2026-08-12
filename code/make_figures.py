#!/usr/bin/env python3
"""Regenerate all five manuscript figures for 'Drift Inherits Its Driver'.

  Fig 1  hospital: degradation is indexed to calendar year, not elapsed time
  Fig 2  unlabeled regime distance does not predict degradation
  Fig 3  electricity: degradation is monotone in elapsed time, at all three cutoffs
         (a) by model age, (b) by calendar year - the Section 5.4 identification test
  Fig 4  electricity: the error increase is localized to midday
  Fig 5  electricity: duck-curve residual signature

Figures 1-3 READ the canonical CSVs rather than recomputing. An earlier version of
this file re-estimated Figures 1 and 2 from the panel under its own specification
(50/50 entity split, min_samples_leaf=20) while the manuscript tables came from
primary_temporal.py and regime_distance.py (70/30 split, min_samples_leaf=5), so the
figures and the tables they illustrate were drawn from different runs and did not
agree. One number, one script, one stored output, one figure.

  Fig 1  <- derived/primary_temporal_summary.csv   (code/primary_temporal.py)
  Fig 2  <- derived/regime_distance_summary.csv    (code/regime_distance.py)
  Fig 3  <- derived/power_multicutoff_origspec.csv (code/power_multicutoff_origspec.py)

Figures 4 and 5 are hour-of-day and residual decompositions with no table to match,
and are computed here from the same origspec electricity model used for Figure 3.

Run the three producing scripts before this one if the panel or the load data changed.

Usage:  python3 make_figures.py [--outdir ../figures]
"""
import argparse, glob, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# ── validated palette (dataviz skill) ────────────────────────────────────────
# Two-slot BLUE/ORANGE: all six checks PASS, CVD dE 24.7.
# Three-slot BLUE/ORANGE/AQUA (Figure 3, --pairs all): checks PASS, worst CVD
# dE 9.2, worst normal-vision dE 24.0; AQUA sits at 2.74:1 on the white surface,
# below the 3:1 gate, so every Figure 3 series carries a visible direct label.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8a86', '#e2e2df'
plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 300, 'font.size': 9,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlesize': 10,
    'axes.titleweight': 'semibold', 'axes.titlecolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
def finish(ax, ylab, xlab=None, title=None):
    ax.grid(axis='y', alpha=.9); ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab)
    if xlab: ax.set_xlabel(xlab)
    if title: ax.set_title(title, loc='left', pad=10)

POWER = os.path.expanduser('~/Documents/project_power/data')
# No classifier specification lives in this file. The hospital panel is not re-fit
# here; Figures 1 and 2 read what primary_temporal.py and regime_distance.py stored.
DERIVED = os.path.join(os.path.dirname(__file__), '..', 'derived')


def canonical(name):
    """Load a stored result CSV, failing loudly rather than silently recomputing."""
    path = os.path.join(DERIVED, name)
    if not os.path.exists(path):
        raise SystemExit(f'missing {path}\nrun the script that produces it first')
    return pd.read_csv(path)

def load_power():
    files = sorted(glob.glob(os.path.join(POWER, 'multiyear', '*.csv'))) + \
            [os.path.join(POWER, 'power_demand.csv')]
    a = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    a['dt'] = pd.to_datetime(a['날짜'], errors='coerce')
    a = a.dropna(subset=['dt']).drop_duplicates(subset=['dt']).sort_values('dt')
    hrs = [c for c in a.columns if c.endswith('시')]
    L = a.melt(id_vars=['dt'], value_vars=hrs, var_name='hr', value_name='load')
    L['hour'] = L['hr'].str.replace('시', '').astype(int)
    L['year'] = L.dt.dt.year; L['month'] = L.dt.dt.month
    L['dow'] = L.dt.dt.dayofweek; L['doy'] = L.dt.dt.dayofyear
    L['norm'] = L['load'] / L.groupby('year')['load'].transform('mean')
    return L

def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    PFE = ['hour', 'month', 'dow', 'doy']
    mape = lambda y, p: np.mean(np.abs((y - p) / y)) * 100

    # ── Figure 1 ────────────────────────────────────────────────────────────
    pt = canonical('primary_temporal_summary.csv')
    pt = pt[pt.panel == 'imbalanced']
    d16 = dict(zip(pt[pt.cutoff == 2016].year, pt[pt.cutoff == 2016].delta))
    d18 = dict(zip(pt[pt.cutoff == 2018].year, pt[pt.cutoff == 2018].delta))
    b16 = pt[pt.cutoff == 2016].baseline.iloc[0]
    b18 = pt[pt.cutoff == 2018].baseline.iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for (d, end, col, lab) in [(d16, 2016, BLUE, 'trained through 2016'),
                               (d18, 2018, ORANGE, 'trained through 2018')]:
        ys = [y for y in range(end + 1, 2024)]
        ax.plot(ys, [d[y] for y in ys], color=col, lw=2, marker='o', ms=5,
                mec='white', mew=1.2, label=lab, zorder=3)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate('2021 falls by the same amount whether\nthe model is 3 or 5 years old',
                xy=(2021, d16[2021]), xytext=(2017.05, -0.048), color=INK2, fontsize=8,
                va='center', ha='left', arrowprops=dict(arrowstyle='-', color=MUTED,
                lw=.9, shrinkA=6, shrinkB=6, connectionstyle='arc3,rad=-0.15'))
    ax.set_xticks(range(2017, 2024)); ax.set_xlim(2016.6, 2023.4)
    finish(ax, 'Δ AUC vs. training-period baseline', 'test year',
           'Degradation is indexed to the calendar year, not to model age')
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{outdir}/fig1_regime_indexed.png', bbox_inches='tight')
    fig.savefig(f'{outdir}/fig1_regime_indexed.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'  fig1  baseline(2016)={b16:.3f} baseline(2018)={b18:.3f} '
          f'2021: {d16[2021]:+.3f}/{d18[2021]:+.3f}')

    # ── Figure 2 ────────────────────────────────────────────────────────────
    rd = canonical('regime_distance_summary.csv')
    rd = rd[~rd.in_sample]
    shift = dict(zip(rd.year, rd.regime_distance))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xs = [shift[y] for y in range(2019, 2024)]
    ys = [d18[y] for y in range(2019, 2024)]
    ax.scatter(xs, ys, s=70, color=BLUE, ec='white', lw=1.4, zorder=3)
    for y, x, v in zip(range(2019, 2024), xs, ys):
        ax.annotate(str(y), (x, v), textcoords='offset points', xytext=(9, -3),
                    fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    xlo, xhi = min(xs) - .035, max(xs) + .055
    ax.set_xlim(xlo, xhi); ax.set_ylim(min(ys) - .012, .022)
    ax.annotate('largest input shift,\nno performance loss',
                xy=(shift[2020], d18[2020]), xytext=(shift[2020] - 0.028, 0.016),
                fontsize=8, color=INK2, ha='right',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=.9,
                shrinkA=2, shrinkB=6))
    finish(ax, 'Δ AUC vs. baseline', 'unlabeled regime distance (discriminator AUC)',
           'The loudest alarm lands in the year the model was fine')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig2_unlabeled_fails.png', bbox_inches='tight')
    fig.savefig(f'{outdir}/fig2_unlabeled_fails.pdf', bbox_inches='tight'); plt.close(fig)
    from scipy import stats
    r, p = stats.pearsonr(xs, ys)
    print(f'  fig2  out-of-sample r={r:+.3f} (p={p:.3f})')

    # ── Figures 3-5 ─────────────────────────────────────────────────────────
    L = load_power()
    tr = L[L.year <= 2018]
    m = GradientBoostingRegressor(random_state=0).fit(tr[PFE], tr['norm'])
    L['pred'] = m.predict(L[PFE])
    L['ape'] = np.abs((L['norm'] - L['pred']) / L['norm']) * 100
    L['resid'] = L['norm'] - L['pred']
    per = {y: L[L.year == y]['ape'].mean() for y in range(2013, 2026)}
    pbase = np.mean([per[y] for y in range(2013, 2019)])

    # Both panels draw all three training cutoffs of the same stored run, so the
    # figure and the two tables of Section 5.4 report the identical estimates.
    pm = canonical('power_multicutoff_origspec.csv')
    cuts = [(2016, BLUE), (2018, ORANGE), (2020, AQUA)]
    series = {c: pm[pm.cutoff == c].sort_values('test_year') for c, _ in cuts}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)

    # (a) elapsed time — the monotonicity claim, one curve per cutoff
    for c, col in cuts:
        g = series[c]
        axA.plot(g.elapsed, g.delta_pp, color=col, lw=2, marker='o', ms=5,
                 mec='white', mew=1.2, label=f'trained through {c}', zorder=3)
        axA.annotate(str(c), (g.elapsed.iloc[-1], g.delta_pp.iloc[-1]),
                     textcoords='offset points', xytext=(6, -1), fontsize=8, color=col)
    g16 = series[2016]
    axA.annotate('one reversal, 2016 cutoff\n(2019 below 2018)',
                 xy=(3, g16[g16.elapsed == 3].delta_pp.iloc[0]), xytext=(4.1, 0.45),
                 fontsize=7.5, color=INK2, va='center',
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=.9, shrinkA=4,
                                 shrinkB=6, connectionstyle='arc3,rad=-0.2'))
    axA.set_xticks(range(1, 10)); axA.set_xlim(0.5, 9.9)
    finish(axA, 'Δ MAPE vs. baseline (pp)', 'model age at test time (years)',
           '(a) Degradation grows with model age')
    axA.legend(loc='upper left', fontsize=8)

    # (b) calendar year — the identification test, read down each column
    for c, col in cuts:
        g = series[c]
        axB.plot(g.test_year, g.delta_pp, color=col, lw=2, marker='o', ms=5,
                 mec='white', mew=1.2, zorder=3)
        axB.annotate(str(c), (g.test_year.iloc[-1], g.delta_pp.iloc[-1]),
                     textcoords='offset points', xytext=(6, -1), fontsize=8, color=col)
    lo = series[2020].set_index('test_year').delta_pp[2021]
    hi = series[2016].set_index('test_year').delta_pp[2021]
    axB.annotate('', xy=(2021, hi), xytext=(2021, lo),
                 arrowprops=dict(arrowstyle='<->', color=INK2, lw=1))
    axB.annotate(f'2021, three model ages:\n{hi/lo:.2f}× oldest over newest',
                 xy=(2021, lo), xytext=(2021.45, 0.32), fontsize=7.5,
                 color=INK2, va='center',
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=.9, shrinkA=4,
                                 shrinkB=8, connectionstyle='arc3,rad=0'))
    axB.set_xticks(range(2017, 2026, 2)); axB.set_xlim(2016.4, 2026.4)
    finish(axB, None, 'test year', '(b) At a fixed year, the fresher model loses less')
    for ax in (axA, axB):
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f'{outdir}/fig3_power_monotone.png', bbox_inches='tight')
    fig.savefig(f'{outdir}/fig3_power_monotone.pdf', bbox_inches='tight'); plt.close(fig)
    rr = []
    for c, _ in cuts:
        g = series[c]
        r2, p2 = stats.pearsonr(g.elapsed, g.delta_pp)
        rr.append(f'{c}: r={r2:+.3f} (p={p2:.1e})')
    print('  fig3  ' + ' | '.join(rr) + f' | 2021 oldest/newest {hi/lo:.2f}x')

    early = L[(L.year >= 2013) & (L.year <= 2018)]; late = L[L.year >= 2023]
    inc = [late[late.hour == h]['ape'].mean() - early[early.hour == h]['ape'].mean()
           for h in range(1, 25)]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    peak = int(np.argmax(inc))
    ax.bar(range(1, 25), inc, color=[ORANGE if i == peak else BLUE for i in range(24)],
           width=.72, zorder=3)
    ax.annotate(f'{inc[peak]:+.2f} pp at {peak+1}:00', xy=(peak + 1, inc[peak]),
                xytext=(0, 6), textcoords='offset points', ha='center',
                fontsize=8.5, color=INK)
    ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
    finish(ax, 'increase in MAPE (pp)', 'hour of day',
           'The entire increase is localized to midday')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig4_hourly.png', bbox_inches='tight')
    fig.savefig(f'{outdir}/fig4_hourly.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'  fig4  peak {peak+1}:00 {inc[peak]:+.2f}pp / 06:00 {inc[5]:+.2f}pp')

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    yy = list(range(2013, 2026))
    for h, col, lab in [(13, BLUE, '13:00 (solar peak)'), (19, ORANGE, '19:00 (evening ramp)')]:
        s = [L[(L.year == y) & (L.hour == h)]['resid'].mean() for y in yy]
        ax.plot(yy, s, color=col, lw=2, marker='o', ms=4.5, mec='white', mew=1.1,
                label=lab, zorder=3)
        ax.annotate(lab.split(' ')[0], (yy[-1], s[-1]), textcoords='offset points',
                    xytext=(6, -2), fontsize=8, color=col)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.axvspan(2013, 2018, color=GRID, alpha=.45, zorder=0)
    ax.annotate('training period', xy=(2015.4, ax.get_ylim()[1] * .88),
                fontsize=8, color=MUTED, ha='center')
    ax.set_xticks(range(2013, 2026, 2)); ax.set_xlim(2012.6, 2026.2)
    finish(ax, 'mean residual, normalized load', 'year',
           'Midday suppression and evening excess: the duck-curve signature')
    ax.legend(loc='lower left', fontsize=8, bbox_to_anchor=(0.02, 0.02))
    fig.tight_layout(); fig.savefig(f'{outdir}/fig5_duck_curve.png', bbox_inches='tight')
    fig.savefig(f'{outdir}/fig5_duck_curve.pdf', bbox_inches='tight'); plt.close(fig)
    print('  fig5  written')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), '..', 'figures'))
    a = ap.parse_args()
    print('regenerating figures ...')
    main(a.outdir)
    print(f'done -> {os.path.abspath(a.outdir)}')
