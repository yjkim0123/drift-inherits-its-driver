# Drift Inherits Its Driver

Code, derived data, and stored results for *Drift Inherits Its Driver: Why
Deployed Models Recover from Policy Shocks but Not from Infrastructure Buildout*
(Yongjun Kim, Ajou University).

The claim of the paper is that the **shape of a deployed model's degradation is
inherited from the exogenous process driving it**: a bounded policy shock
produces drift indexed to the calendar that recovers, and a monotone
infrastructure buildout produces drift indexed to model age that does not. The
two domains here are US hospital financial distress (CMS HCRIS, 2011–2023) and
Korean national electricity demand (KPX hourly load, 2013–2025).

## Reproducing every number in the paper

```
python3 code/verify_manuscript.py <manuscript.md|.tex|.txt>
```

This recomputes each reported quantity from the stored outputs in `derived/` and
asserts that the string the manuscript prints is the string the data support. It
runs against the Markdown source, the LaTeX source, and the text extracted from
the typeset PDF, and all three must agree:

```
md              82 PASS / 0 FAIL
paper/main.tex  82 PASS / 0 FAIL
paper/main.txt  82 PASS / 0 FAIL
```

The three targets are not redundant. `pdflatex` silently drops characters it
cannot set: an earlier build of this manuscript rendered `r = −0.430` as
`r = 0.430` in the PDF while the `.tex` still held the right character. Only the
check against extracted PDF text catches that.

## Layout

| Path | What is in it |
|---|---|
| `code/` | every analysis script; each writes a CSV to `derived/` |
| `derived/` | the panels and the stored result of every table and figure |
| `figures/` | the five manuscript figures (PNG and PDF) plus `make_figures.py` notes |
| `code/md_to_latex.py` | builds the `sn-jnl` LaTeX submission from the Markdown source |

The manuscript sources are added to this repository on acceptance; until then the
verification script takes the manuscript path as an argument.

## Which script produces which result

| Manuscript | Script | Stored output |
|---|---|---|
| §5.1 hospital degradation, both cutoffs | `code/primary_temporal.py` | `primary_temporal{,_summary}.csv` |
| §5.2 transfer vs. contemporaneous | `code/hospital_contemporaneous.py` | `hospital_contemporaneous.csv` |
| §5.3 unlabeled regime distance | `code/regime_distance.py` | `regime_distance{,_summary}.csv` |
| §5.4 electricity, three cutoffs | `code/power_multicutoff_origspec.py` | `power_multicutoff_origspec.csv` |
| §5.4 robustness specification | `code/power_multicutoff.py` | `power_multicutoff.csv` |
| §5.4 electricity contemporaneous | `code/power_contemporaneous.py` | `power_contemporaneous.csv` |
| §5.6 canary power | `code/canary_power.py` | `canary_power.csv` |
| §5.7 window grid | `code/grid_all_windows.py` | `grid_all_windows{,_marginals}.csv` |
| §5.7 exit window | `code/grid_exit_windows.py` | `grid_exit_windows.csv` |
| §5.7 2×2 cells | `code/grid_2x2_cells.py` | `grid_2x2_cells_{imbal,bal}.csv` |
| Figures 1–5 | `code/make_figures.py` | `figures/fig*.pdf` |

Figures 1–3 **read** the stored CSVs rather than re-estimating. An earlier
version of `make_figures.py` re-fit the hospital panel under its own
specification (50/50 entity split, `min_samples_leaf=20`) while the manuscript
tables came from the canonical scripts (70/30, `min_samples_leaf=5`), so the
figures and the tables they illustrated were drawn from different runs. No
classifier specification lives in that file any more.

## Rebuilding the manuscript

```
python3 code/md_to_latex.py                      # Markdown -> sn-jnl LaTeX
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The `.tex` is generated; edit the Markdown, not the LaTeX.

## Data sources

Both datasets are public and neither is redistributed here in raw form.

- **CMS HCRIS** cost report public use files, 2011–2023 — <https://www.cms.gov/data-research>
- **KPX** hourly nationwide system load, 2013–2025 — <https://www.data.go.kr>

`derived/` contains the panels built from them, so the analysis runs without
re-downloading the sources.

## License

Code is released under the MIT License (`LICENSE`). The derived panels are built
from US and Korean public-sector open data and carry those sources' terms.
