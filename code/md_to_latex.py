#!/usr/bin/env python3
"""Port DriftDriver_Manuscript_v1.md to the Springer Nature LaTeX template.

The manuscript is authored in Markdown and the journal wants sn-jnl.cls, so the
port has to be repeatable: any hand-edit of the .tex is lost the next time the
Markdown changes. Everything mechanical happens here.

  markdown body  -> pandoc -> LaTeX fragment
  "(Author, 2020)" / "Author (2020)" -> \\citep / \\citet against refs.bib
  figure anchors  -> float environments with captions
  front matter    -> sn-jnl title block, abstract, keywords, declarations

Run:  python3 md_to_latex.py
Then: cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

verify_manuscript.py runs against the produced .tex as well as the .md, so a
number lost in translation fails a check rather than reaching the journal.
"""
import re
import subprocess
from pathlib import Path

import os
_ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("DRIFT_MANUSCRIPT",
           Path.home() / ".openclaw/workspace/DriftDriver_Manuscript_v1.md"))
OUT = _ROOT / "paper" / "main.tex"

# in-text citation string -> bib key. Every reference in refs.bib appears here;
# an unmapped citation is a hard error below rather than a silent passthrough.
KEYS = {
    "Altman, 1968": "altman1968",
    "Bergmeir and Benítez, 2012": "bergmeir2012",
    "Bifet and Gavaldà, 2007": "bifet2007",
    "Centers for Medicare & Medicaid Services, 2025": "cms2025",
    "Denholm et al., 2015": "denholm2015",
    "European Parliament and Council of the European Union, 2024": "euaiact2024",
    "EU AI Act, 2024": "euaiact2024",
    "Gama et al., 2004": "gama2004",
    "Gama et al., 2014": "gama2014",
    "Hong and Fan, 2016": "hong2016",
    "IEA-PVPS, 2020": "ieapvps2020",
    "Kim and Sun, 2026": "kim2026",
    "Korea Energy Agency, 2025": "kea2025",
    "Lopez-Paz and Oquab, 2017": "lopezpaz2017",
    "Moreno-Torres et al., 2012": "morenotorres2012",
    "NIST, 2023": "nist2023",
    "Page, 1954": "page1954",
    "Rabanser et al., 2019": "rabanser2019",
    "Roberts et al., 2017": "roberts2017",
    "Sculley et al., 2015": "sculley2015",
    "Tashman, 2000": "tashman2000",
    "Vela et al., 2022": "vela2022",
    "Lu et al., 2018": "lu2019",
    "Webb et al., 2016": "webb2016",
    "Ditzler et al., 2015": "ditzler2015",
    "Grzenda et al., 2019": "grzenda2020",
    "Sethi and Kantardzic, 2017": "sethi2017",
    "dos Reis et al., 2016": "dosreis2016",
    "Finlayson et al., 2021": "finlayson2021",
    "Davis et al., 2017": "davis2017",
    "Subbaswamy and Saria, 2019": "subbaswamy2020",
    "Paleyes et al., 2022": "paleyes2022",
    "Kolter and Maloof, 2003": "kolter2003",
    "Elwell and Polikar, 2011": "elwell2011",
    "Ben-David et al., 2009": "bendavid2010",
    "Breck et al., 2017": "breck2017",
    "Barros and Santos, 2018": "barros2018",
    "Gonçalves et al., 2014": "goncalves2014",
    "Frias-Blanco et al., 2015": "friasblanco2015",
    "Žliobaitė et al., 2015": "zliobaite2016",
    "Gama et al., 2012": "gama2013",
    "Bayram et al., 2022": "bayram2022",
    "Sugiyama et al., 2008": "sugiyama2008",
    "Street and Kim, 2001": "street2001",
    "Klinkenberg, 2004": "klinkenberg2004",
    "Karim et al., 2026": "karim2026",
}
# narrative form: "Kim and Sun (2026) showed" -> \citet
NARRATIVE = {
    "Webb et al.": "webb2016",
    "Lu et al.": "lu2019",
    "Sethi and Kantardzic": "sethi2017",
    "Grzenda et al.": "grzenda2020",
    "Bayram et al.": "bayram2022",
    "Žliobaitė et al.": "zliobaite2016",
    "Paleyes et al.": "paleyes2022",
    "Breck et al.": "breck2017",
    "Finlayson et al.": "finlayson2021",
    "Davis et al.": "davis2017",
    "Subbaswamy and Saria": "subbaswamy2020",
    "Kim and Sun": "kim2026",
    "Kim & Sun": "kim2026",
    "Denholm et al.": "denholm2015",
    "Gama et al.": "gama2014",
    "Rabanser et al.": "rabanser2019",
    "Roberts et al.": "roberts2017",
    "Sculley et al.": "sculley2015",
    "Vela et al.": "vela2022",
    "Moreno-Torres et al.": "morenotorres2012",
    "Lopez-Paz and Oquab": "lopezpaz2017",
    "Hong and Fan": "hong2016",
    "Bergmeir and Benítez": "bergmeir2012",
    r"Kim \& Sun": "kim2026",
    r"Centers for Medicare \& Medicaid Services": "cms2025",
    "Tashman": "tashman2000",
    "Altman": "altman1968",
    "Page": "page1954",
}

# Figure placement: the paragraph cue that must already be in the text, the file,
# and the caption. Captions live here because the Markdown has none - the journal
# needs them and figures/README.md is not a submission artifact.
FIGURES = [
    dict(key="fig1", file="fig1_regime_indexed",
         after="The pit does not move when the model moves. It is indexed to the calendar.",
         caption=r"Hospital degradation is indexed to the calendar year, not to model "
                 r"age. $\Delta$ AUC against the training-period unseen-entity baseline "
                 r"for models trained through 2016 (blue) and through 2018 (orange), "
                 r"scored on hospitals never seen in training. 2021 falls by $-0.088$ "
                 r"and $-0.080$ respectively, at model ages of five and three years; "
                 r"the gap between them is smaller than the seed spread "
                 r"($\mathrm{sd} \approx 0.019$). Shifting the training cutoff by two "
                 r"years does not shift the pit."),
    dict(key="fig2", file="fig2_unlabeled_fails",
         after="here it does not merely miss the event, it points the other way.",
         caption=r"Unlabeled monitoring is uncorrelated with realized degradation and "
                 r"mistimed. Each point is one test year, 2019--2023: the horizontal "
                 r"axis is the unlabeled regime distance (discriminator AUC between the "
                 r"training window and the test year), the vertical axis the realized "
                 r"$\Delta$ AUC of the deployed model. Out-of-sample $r = +0.143$ "
                 r"($p = 0.82$). 2020 carries the largest input shift in the panel and "
                 r"no performance loss at all."),
    dict(key="fig3", file="fig3_power_monotone",
         after="the same ordering, the same decline in the ratio, larger if anything.",
         caption=r"Electricity degradation grows with model age at every training "
                 r"cutoff. \textbf{(a)} $\Delta$ MAPE against the training-period "
                 r"baseline by model age, for cutoffs 2016, 2018 and 2020 "
                 r"($r = +0.949$, $+0.979$, $+0.981$); the single reversal, at three "
                 r"years on the 2016 cutoff, is marked rather than smoothed. "
                 r"\textbf{(b)} the same estimates against the calendar. The "
                 r"identification test is the vertical gap at a shared test year: in "
                 r"2021 a five-year-old model loses $1.96\times$ what a one-year-old "
                 r"one loses on identical data. The same manipulation moves nothing in "
                 r"the hospital domain (Fig.~\ref{fig:fig1})."),
    dict(key="fig4", file="fig4_hourly",
         after="against **06:00 +0.31** and 24:00 +0.59.",
         caption=r"The electricity error increase is localized to midday. Increase in "
                 r"MAPE by hour of day between the training window (2013--2018) and "
                 r"2023--2025. The peak is $+8.39$ pp at 13:00 against $+0.31$ pp at "
                 r"06:00. The solar hypothesis committed to this shape in advance."),
    dict(key="fig5", file="fig5_duck_curve",
         after="while 19:00 moved from −0.006 to +0.057.",
         caption=r"The residual signature is the duck curve. Mean residual of the "
                 r"normalized-load model at 13:00 (solar peak) and 19:00 (evening ramp) "
                 r"by year; the shaded span is the training window. Midday residuals "
                 r"move from $-0.029$ to $-0.091$ ($r = -0.933$, $p = 3.2 \times "
                 r"10^{-6}$) while post-sunset residuals move from $-0.006$ to "
                 r"$+0.057$: generation the model cannot see suppresses midday net "
                 r"load and pushes the ramp later."),
]

# The Markdown carries no table captions and pandoc emits captionless longtables,
# which sn-jnl rejects ("No counter 'none' defined") and which a journal will not
# accept anyway. Captions are given here in document order; the count is asserted
# against the number of tables found, so an added table fails loudly.
TABLES = [
    ("tab:ccn", r"External validation of the exit variable. Current Medicare "
     r"enrollment of panel hospitals by when their cost-report observations "
     r"ceased. Retention falls monotonically with how long ago reporting stopped, "
     r"which is why only pre-2020 breaks are treated as genuine exits."),
    ("tab:grid", r"The 2$\times$2 novelty grid. Entities are split into disjoint "
     r"sets A and B and periods into a training and a held-out window; T4 is the "
     r"deployment condition and the only cell that answers the question a "
     r"deployment decision asks."),
    ("tab:cutoffs", r"Hospital degradation at two training cutoffs. $\Delta$ AUC "
     r"against the training-period baseline for models trained through 2016 and "
     r"through 2018, scored on unseen hospitals. The 2021 pit does not move when "
     r"the cutoff moves."),
    ("tab:contemp", r"Hospital transfer versus contemporaneous training. Arm A "
     r"trains on 2011--2018 and transfers to year $t$; arm B trains on year $t$ "
     r"itself with entities disjoint from the evaluation set. Retraining restores "
     r"2021 to normal-period performance, which is the definitional test for "
     r"concept drift."),
    ("tab:powerdeg", r"Electricity degradation against the training-period "
     r"baseline at the 2018 cutoff. $\Delta$ MAPE rises with elapsed time and "
     r"never recovers."),
    ("tab:powerid", r"The electricity identification test. Each row holds the "
     r"calendar year fixed and reads across model ages; in every year the fresher "
     r"model degrades less. The same manipulation moves nothing in the hospital "
     r"domain."),
    ("tab:powercontemp", r"Electricity transfer versus contemporaneous training, "
     r"separated by ISO week parity. Retraining recovers a growing share of the "
     r"loss but never returns to the in-era reference: the driver moves the "
     r"mapping and drains the feature set at once."),
    ("tab:canary", r"Canary detection power by audit-sample size. Each threshold "
     r"$\tau_n$ is calibrated to a 5\% false-alarm rate within the stable "
     r"training window (2,000 bootstrap resamples per cell). Boldface marks the "
     r"two broken years."),
    ("tab:windows", r"Entity cost against period cost by outcome and test window. "
     r"The ordering that survived seeds, model families and preprocessing "
     r"reverses when the test window moves off 2021--22."),
    ("tab:mapping", r"The two domains mapped onto the driver-trajectory account."),
]

# in-text "Table N" -> the label it actually points at. Two further references
# ("IEA-PVPS, 2020, Table 3") are to a table in a cited source, not to ours, and
# are deliberately absent.
TABLE_REFS = {"Table 3": "tab:contemp", "Table 4": "tab:canary",
              "Table 5": "tab:mapping"}

PREAMBLE = r"""%% Generated by code/md_to_latex.py from DriftDriver_Manuscript_v1.md.
%% Do not hand-edit: edit the Markdown and re-run the converter.
\documentclass[pdflatex,sn-basic]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{calc}   %% pandoc's wide tables use \real{} for column widths
\usepackage[title]{appendix}
\usepackage{textcomp}
\usepackage{url}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{float}   %% [H] keeps Algorithm 1 where it is referenced

\theoremstyle{remark}
\newtheorem{remark}{Remark}


\begin{document}

\title[Drift Inherits Its Driver]{Drift Inherits Its Driver: Why Deployed
Models Recover from Policy Shocks but Not from Infrastructure Buildout}

\author*{\fnm{Yongjun} \sur{Kim}}\email{yjkim0123@ajou.ac.kr}

%% sn-jnl's \orcid renders a logo from Orcidlogo.eps, which the template zip does
%% not ship; the identifier goes on the title page as text instead.
\affil*{\orgdiv{Department of Software and Computer Engineering},
\orgname{Ajou University}, \orgaddress{\city{Suwon}, \country{Republic of Korea}}.
ORCID: \url{https://orcid.org/0000-0003-4234-4883}}

\abstract{%(ABSTRACT)s}

\keywords{Concept drift, Model monitoring, Distribution shift, Deployment
evaluation, Temporal validation}

\maketitle
"""

BACKMATTER = r"""
\section*{Declarations}

\bmhead{Funding} No funding was received for conducting this study.

\bmhead{Competing interests} The author declares no competing interests.

\bmhead{Ethics approval and consent to participate} Not applicable. The study
uses only aggregate, publicly released administrative and operational records
and involves no human participants.

\bmhead{Consent for publication} Not applicable.

\bmhead{Materials availability} Not applicable.

\bmhead{Data availability} All data underlying this study are public. The US
hospital panel is derived from the Centers for Medicare \& Medicaid Services
Healthcare Cost Report Information System (HCRIS) public use files for
2011--2023, available at \url{https://www.cms.gov/data-research}. The Korean
electricity load series is the Korea Power Exchange (KPX) hourly nationwide
demand series for 2013--2025, available at \url{https://www.data.go.kr}. The
derived panels and every stored result file reported in this paper are released
with the code (see below).

\bmhead{Code availability} All analysis code, the derived panels, and the stored
result files behind every table and figure are available at
\url{https://github.com/yjkim0123/drift-inherits-its-driver}. Each reported
quantity is reproduced from a stored output by \texttt{verify\_manuscript.py},
which fails if a number in the manuscript is not the number the data support.

\bmhead{Author contributions} Y.K. is the sole author and is responsible for the
study design, data preparation, analysis, and writing.

\bibliography{refs}

\end{document}
"""


# pdflatex silently DROPS characters outside its input encoding: the first port
# of this manuscript rendered "r = -0.430" as "r = 0.430" and "n ~ 400" as "n 400"
# in the PDF while the .tex source still held the right character, so a source-level
# check passed a PDF that had lost 74 minus signs. Every non-ASCII character the
# manuscript uses is mapped to math mode here.
SUPERSCRIPT = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
               "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
UNICODE = {
    "−": "$-$", "×": r"$\times$", "±": r"$\pm$", "≈": r"$\approx$",
    "→": r"$\rightarrow$", "Δ": r"$\Delta$", "≤": r"$\le$", "≥": r"$\ge$",
    "≫": r"$\gg$", "ρ": r"$\rho$", "τ": r"$\tau$", "½": r"$\frac{1}{2}$",
    "σ": r"$\sigma$", "μ": r"$\mu$", "α": r"$\alpha$", "β": r"$\beta$",
    "ç": r"\c{c}", "ã": r"\~{a}", "Ž": r"\v{Z}", "ė": r"\.{e}", "ü": r'\"{u}',
    "é": r"\'{e}", "í": r"\'{i}", "á": r"\'{a}", "ó": r"\'{o}", "ñ": r"\~{n}",
    "š": r"\v{s}", "ř": r"\v{r}", "à": r"\`{a}", "ö": r'\"{o}', "ä": r'\"{a}',
}


def latexify_unicode(tex: str) -> str:
    tex = re.sub("[" + "".join(SUPERSCRIPT) + "]+",
                 lambda m: "$^{" + "".join(SUPERSCRIPT[c] for c in m.group()) + "}$",
                 tex)
    for ch, cmd in UNICODE.items():
        tex = tex.replace(ch, cmd)
    left = sorted({c for c in tex if ord(c) > 127})
    if left:
        raise SystemExit(f"unmapped non-ASCII would be dropped by pdflatex: {left}")
    return tex


def pandoc(md: str) -> str:
    p = subprocess.run(
        ["pandoc", "-f", "markdown+pipe_tables", "-t", "latex",
         "--top-level-division=section", "--wrap=preserve"],
        input=md, capture_output=True, text=True, check=True)
    return p.stdout


def cite(tex: str) -> tuple[str, set]:
    """Rewrite author-year strings as natbib commands. Longest match first so
    'Gama et al., 2014' is never eaten by a prefix of 'Gama et al., 2004'.

    A group whose parts are bare citations becomes \\citep. A part carrying a
    qualifier ('IEA-PVPS, 2020, for the 2013-2019 figures') keeps the qualifier:
    only the author-year prefix becomes \\citealp, inside the original parens.
    Dropping those qualifiers would silently detach each figure from the source
    that actually covers it."""
    used = set()
    # pandoc has already escaped ampersands; match against the source spelling
    lookup = {k.replace("&", r"\&"): v for k, v in KEYS.items()}
    lookup.update(KEYS)
    prefixes = sorted(lookup, key=len, reverse=True)

    def resolve(part):
        """(key, trailing qualifier) or (None, None)."""
        if part in lookup:
            return lookup[part], ""
        for p in prefixes:
            if part.startswith(p + ","):
                return lookup[p], part[len(p) + 1:].strip()
        return None, None

    def paren(m):
        parts = [p.strip() for p in m.group(1).split(";")]
        got = [resolve(p) for p in parts]
        if any(k is None for k, _ in got):
            return m.group(0)  # not a citation group; leave alone
        used.update(k for k, _ in got)
        if all(not q for _, q in got):
            return "\\citep{" + ",".join(k for k, _ in got) + "}"
        return "(" + "; ".join(
            "\\citealp{" + k + "}" + (", " + q if q else "") for k, q in got) + ")"

    tex = re.sub(r"\(([^()]*?\b(?:19|20)\d{2}[a-z]?[^()]*?)\)", paren, tex)

    for name in sorted(NARRATIVE, key=len, reverse=True):
        key = NARRATIVE[name]
        # pandoc ties the year to the name with a non-breaking space: "Webb et al.~(2016)"
        pat = re.escape(name) + r"[~\s]+\(((?:19|20)\d{2})\)"
        if re.search(pat, tex):
            used.add(key)
        tex = re.sub(pat, "\\\\citet{" + key + "}", tex)
    return tex, used


def caption_tables(tex: str) -> str:
    starts = [m.start() for m in re.finditer(r"\\begin\{longtable\}", tex)]
    if len(starts) != len(TABLES):
        raise SystemExit(f"{len(starts)} tables found, {len(TABLES)} captions written")
    for start, (label, cap) in zip(reversed(starts), reversed(TABLES)):
        rule = tex.index(r"\toprule", start)
        tex = tex[:rule] + "\\caption{" + cap + "}\\label{" + label + "}\\\\\n" + tex[rule:]
    return tex


def _braced(tex: str, open_at: int) -> tuple[str, int]:
    """Content of the brace group starting at open_at, and the index past it.
    Column specs nest braces (p{(\\linewidth - 8\\tabcolsep) * \\real{0.2}}), so a
    non-greedy regex stops at the wrong brace and shreds the table."""
    depth, i = 0, open_at
    while i < len(tex):
        if tex[i] == "{":
            depth += 1
        elif tex[i] == "}":
            depth -= 1
            if depth == 0:
                return tex[open_at + 1:i], i + 1
        i += 1
    raise SystemExit("unbalanced braces in column spec")


def longtables_to_floats(tex: str) -> str:
    """sn-jnl wants captioned float tables; pandoc emits longtable, whose caption
    machinery this class does not define."""
    out, pos = [], 0
    while True:
        start = tex.find(r"\begin{longtable}[]{", pos)
        if start < 0:
            out.append(tex[pos:])
            return "".join(out)
        out.append(tex[pos:start])
        spec, after = _braced(tex, tex.index("{", start + len(r"\begin{longtable}")))
        end = tex.index(r"\end{longtable}", after)
        body = tex[after:end]
        cap = re.search(r"\\caption\{.*?\}\\label\{[^}]*\}\\\\\n", body, re.S)
        head = cap.group(0)[:-3] if cap else ""      # keep \caption..\label, drop \\
        if cap:
            body = body[:cap.start()] + body[cap.end():]
        for junk in (r"\endhead", r"\endfirsthead", r"\endlastfoot", r"\noalign{}"):
            body = body.replace(junk, "")
        out.append("\\begin{table}[htbp]\n\\centering\n" + head +
                   "\n\\begin{tabular}{" + spec + "}\n" +
                   "\n".join(l for l in body.strip().split("\n") if l.strip()) +
                   "\n\\end{tabular}\n\\end{table}")
        pos = end + len(r"\end{longtable}")


def place_figures(tex: str) -> str:
    for f in FIGURES:
        anchor = latexify_unicode(pandoc(f["after"]).strip())
        anchor = anchor.split("\n")[0]
        # the anchor sentence survives pandoc with its own markup; locate the
        # end of the paragraph that contains it
        idx = tex.find(anchor)
        if idx < 0:
            raise SystemExit(f"figure anchor not found for {f['key']}: {anchor[:70]}")
        end = tex.find("\n\n", idx)
        end = len(tex) if end < 0 else end
        block = (
            "\n\n\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\textwidth]{{{f['file']}}}\n"
            f"\\caption{{{f['caption']}}}\n"
            f"\\label{{fig:{f['key']}}}\n"
            "\\end{figure}"
        )
        tex = tex[:end] + block + tex[end:]
    return tex


def main():
    md = SRC.read_text(encoding="utf-8")
    body_all, refs = md.split("## References")
    head, body = body_all.split("## 1. Introduction", 1)
    body = "## 1. Introduction" + body
    abstract = head.split("## Abstract")[1].strip().strip("-").strip()

    n = len(abstract.split())
    if not 150 <= n <= 250:
        raise SystemExit(f"abstract is {n} words; the journal allows 150-250")

    body = re.sub(r"\(Figure (\d)([ab]?)\)", r"(Fig.~\\ref{fig:fig\1})", body)
    body = re.sub(r"\bFigure (\d)([ab])\b", r"Fig.~\\ref{fig:fig\1}\2", body)
    body = re.sub(r"\bFigure (\d)\b", r"Fig.~\\ref{fig:fig\1}", body)

    tex = pandoc(body)
    abs_tex = pandoc(abstract).strip()

    tex, used = cite(tex)
    abs_tex, used2 = cite(abs_tex)
    used |= used2
    tex = latexify_unicode(tex)
    abs_tex = latexify_unicode(abs_tex)

    declared = set(re.findall(r"^@\w+\{([^,]+),", OUT.parent.joinpath("refs.bib")
                              .read_text(encoding="utf-8"), re.M))
    if used - declared:
        raise SystemExit(f"cited but not in refs.bib: {sorted(used - declared)}")
    uncited = declared - used
    if uncited:
        print(f"  note: in refs.bib but never cited: {sorted(uncited)}")

    # pandoc reads '##' as a subsection because the Markdown has an H1 title, and
    # the Markdown numbers its own headings. sn-jnl numbers them itself, so the
    # manual numbers have to go or every heading reads "5 5. Results".
    tex = re.sub(r"\\subsubsection\{\d+\.\d+\s+", r"\\subsection{", tex)
    tex = re.sub(r"\\subsection\{\d+\.\s+", r"\\section{", tex)

    # pandoc emits longtable; sn-jnl wants float tables, and a longtable caption
    # trips "No counter 'none' defined" in this class.
    tex = caption_tables(tex)
    tex = longtables_to_floats(tex)
    tex = place_figures(tex)
    for ref, label in TABLE_REFS.items():
        # leave "IEA-PVPS, 2020, Table 3" alone: that table is in a cited source
        tex = re.sub(r"(?<!, )\b" + ref + r"\b", "Table~\\\\ref{" + label + "}", tex)
    # A sentence that continues after a display equation is not a new paragraph;
    # pandoc turns the blank line into one and LaTeX indents it.
    tex = re.sub(r"(\\end\{equation\}\n\n)([a-z])", r"\1\\noindent \2", tex)
    tex = tex.replace(r"\tightlist", "")
    # the Markdown's '---' section separators become printed rules in LaTeX
    tex = tex.replace(r"\begin{center}\rule{0.5\linewidth}{0.5pt}\end{center}", "")

    OUT.write_text((PREAMBLE % {"ABSTRACT": abs_tex}) + "\n" + tex + BACKMATTER,
                   encoding="utf-8")
    print(f"  wrote {OUT}")
    print(f"  abstract {n} words | {len(used)} of {len(declared)} references cited")


if __name__ == "__main__":
    main()
