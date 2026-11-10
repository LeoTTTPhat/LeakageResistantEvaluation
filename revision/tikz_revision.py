#!/usr/bin/env python3
"""TikZ/pgfplots versions of the three figures of make_revision_figures.py (round 4).

The round-4 manuscript draws its figures in TikZ. This script writes one
self-contained ``tikzpicture`` per figure, from the same result files and with
the data-preparation code of ``make_revision_figures.py`` copied verbatim, so
that every plotted value is the one the PNG shows:

* ``threshold_cost``            -- expected cost per delivered message against the
                                   decision threshold; augmented word model, r = 10,
                                   seed-42 reference partition; one panel per
                                   prevalence, one curve per stream, the L3 curve
                                   without the SpamAssassin duplicates
                                   (``figure_threshold_cost``)
* ``distilbert_calibration``    -- reliability curve on the generator-disjoint
                                   corpus and confidence distributions on both
                                   corpora at the pre-registered checkpoint, pooled
                                   over seeds 29, 42, 71 (``figure_calibration``)
* ``distilbert_learning_curve`` -- F1 at maximum length 160 and 320 and ECE at 160
                                   over two epochs, means over seeds 29, 42, 71
                                   (``figure_learning_curve``)

The histograms are binned with the call that ``ax.hist(bins=25, density=True)``
makes for one dataset (``np.histogram`` over that dataset's own range) and drawn
as constant plots of the pre-binned densities. Axis limits follow matplotlib's
default autoscaling (5% margins, in log space on the cost axis, histogram
baseline at 0) and the ticks are those of the PNGs. Data are written inline (no
external data files); colors are the Okabe-Ito names of the manuscript preamble
(oiBlue, oiGreen, oiVermillion), and the matplotlib script's two grays are
defined inside each picture. Line widths, dash patterns, marker sizes and alphas
are matplotlib's (rcParams defaults of make_figures.configure_style()).

``--check`` also runs the matplotlib figure functions themselves (nothing is
saved) and asserts that every plotted coordinate, axis limit and tick equals the
value written here. It imports make_revision_figures.py with a three-path stand-in
for rev_common, whose import of run_experiment would load torch.

Inputs (results/):
  revision_threshold_cost_curves.csv           L1/L2 cost curves, seed-42 partition
  revision_r4_threshold_cost_curves_dedup.csv  L3 curves without the SpamAssassin duplicates
  revision_distilbert_probabilities.csv        DistilBERT test probabilities
  revision_distilbert_learning_curve.csv       DistilBERT checkpoints, every tenth of an epoch

Outputs:
  figures/tikz/tikz_threshold_cost.tex
  figures/tikz/tikz_distilbert_calibration.tex
  figures/tikz/tikz_distilbert_learning_curve.tex
  and a copy of each in the manuscript bundle (--copy-to; default ../../../manuscriptR4,
  skipped when that folder does not exist)

Usage (from artifact/eswa_study/revision/):
    python tikz_revision.py                        # all three figures
    python tikz_revision.py --only threshold_cost  # selected figures
    python tikz_revision.py --copy-to DIR          # copy the .tex files to DIR ("none": no copy)
    python tikz_revision.py --check                # also compare with the matplotlib figures
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
TIKZ_DIR = FIGURES / "tikz"
BUNDLE = STUDY.parent.parent / "manuscriptR4"
NAMES = ("threshold_cost", "distilbert_calibration", "distilbert_learning_curve")

# --- copied verbatim from make_figures.py and make_revision_figures.py --------
COLORS = {
    "Word TF--IDF": "#0072B2",
    "MiniLM": "#009E73",
    "DistilBERT": "#D55E00",
}
TRAIN_STYLE = {"clean-only": (":", "o"), "linked-augmented": ("-", "s")}


def reliability(probability, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    centres, observed, weight = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (
            probability <= high if high == 1.0 else probability < high
        )
        if not mask.any():
            continue
        centres.append(probability[mask].mean())
        observed.append(y[mask].mean())
        weight.append(mask.mean())
    return np.array(centres), np.array(observed), np.array(weight)
# ------------------------------------------------------------------------------

# Okabe-Ito colors defined in the manuscript preamble.
TIKZ_COLOR = {"#0072B2": "oiBlue", "#009E73": "oiGreen", "#D55E00": "oiVermillion"}

# matplotlib defaults behind the PNGs: dashes scale with the line width
# (lines.dashed_pattern 3.7/1.6, lines.dotted_pattern 1/1.65); markers have a
# 1 pt edge; spines 0.7 pt; major/minor ticks 0.8/0.6 pt wide, 3.5/2 pt long.
REF_DASH = "dash pattern=on 2.96pt off 1.28pt"   # "--" at lw 0.8 (reference lines)
DOTTED = "dash pattern=on 1.3pt off 2.145pt"     # ":" at lw 1.3
MARGIN = 0.05                                    # axes.xmargin / axes.ymargin

# Common panel style. Fonts: ticks and legends \scriptsize, axis labels
# \footnotesize, bold titles at the axis' left edge (matplotlib loc="left").
PANEL_STYLE = r"""\pgfplotsset{revision panel/.style={
  scale only axis,
  axis x line*=bottom, axis y line*=left,
  axis line style={line width=0.7pt},
  tick align=outside, major tick length=3.5pt, minor tick length=2pt,
  tick style={black, line width=0.8pt}, minor tick style={line width=0.6pt},
  tick label style={font=\scriptsize},
  label style={font=\footnotesize},
  title style={font=\footnotesize\bfseries\boldmath, at={(0,1)},
    anchor=south west, inner xsep=0pt},
  legend style={font=\scriptsize, draw=none, fill=none},
  legend cell align=left,
  legend image code/.code={\draw[mark repeat=2, mark phase=2, ##1]
    plot coordinates {(0pt,0pt) (7.5pt,0pt) (15pt,0pt)};},
  scaled ticks=false,
},
revision area legend/.style={% matplotlib's legend key for a histogram outline
  legend image code/.code={\draw[##1] (0pt,-2.6pt) rectangle (15pt,2.6pt);}},
}%"""

# Layout. Each picture is 0.96\linewidth wide, as the PNG was in its figure*
# float: axis width = (0.96\linewidth - FIXED) / 3, where FIXED is the room taken
# by the y label and tick labels of panel (a), the two gaps and the overhang of
# the last tick label, measured in the supplement (\linewidth = 522 pt; the
# pictures come out at 501.0-501.1 pt for a target of 501.1 pt). Axis height =
# RATIO * axis width (the PNG panels' aspect), with a floor that only engages at
# a narrower \linewidth (the response letter's quote, 375 pt): there the y label
# still fits along the axis and the titles wrap inside their own panel.
TC_FIXED, TC_SEP, TC_RATIO, TC_MIN_HEIGHT = "129.4pt", "44pt", "0.90", "100pt"
CAL_FIXED, CAL_SEP, CAL_RATIO, CAL_MIN_HEIGHT = "127.3pt", "45pt", "0.89", "100pt"
LC_FIXED, LC_SEP, LC_RATIO, LC_MIN_HEIGHT = "120.9pt", "42pt", "0.88", "90pt"

# A legend below the panels: an empty, hidden axis that only carries a legend,
# centered under everything drawn so far (matplotlib: fig.legend, lower center).
LEGEND_AXIS = r"""\begin{axis}[revision panel, hide axis, width=1pt, height=1pt,
  at={(current bounding box.south)}, anchor=north, yshift=-%(gap)s,
  xmin=0, xmax=1, ymin=0, ymax=1,
  legend columns=%(columns)d,
  legend style={at={(0.5,0)}, anchor=north,
    /tikz/every even column/.append style={column sep=%(colsep)s}}]
%(entries)s
\end{axis}"""


def fmt(value) -> str:
    """Six significant digits, the resolution of pgfplots' floating-point unit."""
    text = f"{float(value):.6g}"
    return "0" if text == "-0" else text


def fmt_list(values) -> str:
    return ",".join(fmt(v) for v in values)


def coordinates(xs, ys, per_line=6) -> str:
    pairs = [f"({fmt(x)},{fmt(y)})" for x, y in zip(xs, ys)]
    lines = [" ".join(pairs[i:i + per_line]) for i in range(0, len(pairs), per_line)]
    return "\n    ".join(lines)


def table(columns: dict[str, np.ndarray]) -> str:
    """Inline table body for \\pgfplotstableread[row sep=\\\\]."""
    names = list(columns)
    rows = [" ".join(names) + r"\\"]
    for values in zip(*columns.values()):
        rows.append(" ".join(fmt(v) for v in values) + r"\\")
    return "\n".join(rows)


def autoscale(low, high, log=False, sticky=None):
    """matplotlib's default view limits for data spanning [low, high]."""
    if log:
        a, b = math.log10(low), math.log10(high)
        delta = (b - a) * MARGIN
        return 10 ** (a - delta), 10 ** (b + delta)
    delta = (high - low) * MARGIN
    lo, hi = low - delta, high + delta
    if sticky is not None and lo < sticky <= low:  # e.g. histogram baseline at 0
        lo = sticky
    return lo, hi


def linear_ticks(step, low, high):
    first, last = math.ceil(low / step - 1e-9), math.floor(high / step + 1e-9)
    return [round(k * step, 10) for k in range(first, last + 1)]


def log_ticks(low, high):
    """Decades (labelled) and 2..9 multiples (minor) inside [low, high]."""
    k0, k1 = math.floor(math.log10(low)), math.floor(math.log10(high))
    major = [10.0 ** k for k in range(k0, k1 + 1) if low <= 10.0 ** k <= high]
    minor = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in range(2, 10)
             if low <= m * 10.0 ** k <= high]
    return major, minor


def number_format(precision: int) -> str:
    return ("/pgf/number format/.cd, fixed, fixed zerofill, precision=%d" % precision
            if precision else "/pgf/number format/.cd, fixed, precision=0")


def header(name: str, label: str, what: str, data: str) -> str:
    return (
        f"% tikz_{name}.tex -- written by artifact/eswa_study/revision/tikz_revision.py;\n"
        f"% edit and rerun the script, not this file. Supplement, \\label{{{label}}}.\n"
        f"% {what}\n"
        f"% Data: {data}\n"
        f"% Width 0.96\\linewidth (as the PNG in the figure* float); no external files."
    )


# ==============================================================================
# Figure: threshold_cost
# ==============================================================================
STREAM_COLOUR = {
    "primary_pipeline": COLORS["Word TF--IDF"],
    "pipeline_disjoint": COLORS["MiniLM"],
    "human_authored": COLORS["DistilBERT"],
}
STREAM_LABEL = {
    "primary_pipeline": "L1 primary",
    "pipeline_disjoint": "L2 gen.-disjoint",
    "human_authored": "L3 human",
}
STREAM_COLUMN = {"primary_pipeline": "L1", "pipeline_disjoint": "L2", "human_authored": "L3"}


def threshold_cost_data():
    """The curves, minima and reference lines of figure_threshold_cost()."""
    # verbatim from figure_threshold_cost()
    frame = pd.read_csv(RESULTS / "revision_threshold_cost_curves.csv")
    dedup = RESULTS / "revision_r4_threshold_cost_curves_dedup.csv"
    if dedup.exists():  # round 4: L3 stream without the SpamAssassin duplicates
        frame = pd.concat([frame[frame["stream"] != "human_authored"], pd.read_csv(dedup)],
                          ignore_index=True)
    frame = frame[frame["model"] == "word_lr_linked_augmented"]
    assert frame["cost_ratio"].unique().tolist() == [10.0]
    panels = []
    for index, prevalence in enumerate(sorted(frame["prevalence"].unique())):
        block = frame[frame["prevalence"] == prevalence]
        curves = {}
        for stream, colour in STREAM_COLOUR.items():
            part = block[block["stream"] == stream].sort_values("threshold")
            if part.empty:
                continue
            best = part.loc[part["expected_cost"].idxmin()]
            curves[stream] = {
                "x": part["threshold"].to_numpy(), "y": part["expected_cost"].to_numpy(),
                "best": (best["threshold"], best["expected_cost"]), "colour": colour,
            }
        reference = prevalence * 10.0  # ax.axhline: cost of flagging nothing, pi * r
        xs = np.concatenate([c["x"] for c in curves.values()])
        ys = np.concatenate([c["y"] for c in curves.values()] + [[reference]])
        xlim = autoscale(xs.min(), xs.max())
        ylim = autoscale(ys.min(), ys.max(), log=True)
        major, minor = log_ticks(*ylim)
        panels.append({
            "dedup": dedup.exists(),
            "letter": chr(97 + index), "prevalence": prevalence, "curves": curves,
            "reference": reference, "xlim": xlim, "ylim": ylim,
            "xticks": linear_ticks(0.25, *xlim), "yticks": major, "yminor": minor,
            # matplotlib: f"({chr(97 + index)}) $\pi$ = {prevalence:.1%}, $r$ = 10"
            "title": f"({chr(97 + index)}) $\\pi = {prevalence:.1%}$, $r = 10$".replace("%", r"\%"),
        })
    return panels


def threshold_cost_tikz(panels) -> str:
    out = [header(
        "threshold_cost", "fig:threshold",
        "Expected cost per delivered message (log scale) against the decision threshold, "
        "augmented word model, r = 10, seed-42 partition.",
        ("results/revision_threshold_cost_curves.csv (L1, L2) and "
         "results/revision_r4_threshold_cost_curves_dedup.csv (L3)" if panels[0]["dedup"] else
         "results/revision_threshold_cost_curves.csv (L1-L3; no dedup file: round-3 L3 stream)")
        + ", as in make_revision_figures.figure_threshold_cost().")]
    out.append(r"\begin{tikzpicture}[color=black]")
    out.append(r"\definecolor{tcGray}{HTML}{888888}% reference line, matplotlib #888888")
    out.append(PANEL_STYLE)
    # matplotlib "v", ms 4: apex 2 pt below the point, base 4 pt wide, 2 pt above
    out.append(r"\pgfdeclareplotmark{tcTriangleDown}{%"
               "\n  \\pgfpathmoveto{\\pgfqpoint{0pt}{-\\pgfplotmarksize}}%"
               "\n  \\pgfpathlineto{\\pgfqpoint{\\pgfplotmarksize}{\\pgfplotmarksize}}%"
               "\n  \\pgfpathlineto{\\pgfqpoint{-\\pgfplotmarksize}{\\pgfplotmarksize}}%"
               "\n  \\pgfpathclose\\pgfusepathqfillstroke}%")
    out.append(r"\pgfmathsetlengthmacro\tcW{(0.96\linewidth-%s)/3}%% axis width" % TC_FIXED)
    out.append(r"\pgfmathsetlengthmacro\tcH{max(%s*\tcW,%s)}%% axis height" % (TC_RATIO, TC_MIN_HEIGHT))
    for panel in panels:
        curves = panel["curves"]
        x0 = next(iter(curves.values()))["x"]
        for curve in curves.values():
            assert np.array_equal(curve["x"], x0), "streams on different threshold grids"
        cols = {"t": x0}
        cols.update({STREAM_COLUMN[s]: c["y"] for s, c in curves.items()})
        out.append(f"% ({panel['letter']}) prevalence {panel['prevalence']:g}: threshold t and "
                   f"expected cost per stream ({len(x0)} thresholds)")
        out.append(r"\pgfplotstableread[row sep=\\]{" + table(cols) + "}"
                   + r"\tcCost" + panel["letter"].upper())
    out.append(r"""\begin{groupplot}[
  group style={group size=3 by 1, horizontal sep=%s},
  revision panel, width=\tcW, height=\tcH, title style={text width=\tcW, align=flush left},
  ymode=log,
  xlabel={Decision threshold}, ylabel={Expected cost per message},
  xticklabel style={%s},
]""" % (TC_SEP, number_format(2)))
    for panel in panels:
        out.append(r"\nextgroupplot[title={%s}," % panel["title"])
        out.append("  xmin=%s, xmax=%s, xtick={%s}," % (fmt(panel["xlim"][0]), fmt(panel["xlim"][1]),
                                                       fmt_list(panel["xticks"])))
        out.append("  ymin=%s, ymax=%s, ytick={%s}," % (fmt(panel["ylim"][0]), fmt(panel["ylim"][1]),
                                                       fmt_list(panel["yticks"])))
        out.append("  minor ytick={%s}]" % fmt_list(panel["yminor"]))
        tab = r"\tcCost" + panel["letter"].upper()
        for stream, curve in panel["curves"].items():
            colour = TIKZ_COLOR[curve["colour"]]
            out.append(r"\addplot[%s, line width=1.4pt, line join=round] table[x=t, y=%s] {%s};"
                       % (colour, STREAM_COLUMN[stream], tab))
            bx, by = curve["best"]
            out.append(r"\addplot[forget plot, only marks, mark=tcTriangleDown, mark size=2pt,"
                       r" mark options={%s, line width=1pt}] coordinates {(%s,%s)};%% minimum"
                       % (colour, fmt(bx), fmt(by)))
        out.append(r"\draw[tcGray, line width=0.8pt, %s] ({rel axis cs:0,0} |- {axis cs:0,%s})"
                   r" -- ({rel axis cs:1,0} |- {axis cs:0,%s});%% pi*r" % (REF_DASH, fmt(panel["reference"]),
                                                                         fmt(panel["reference"])))
    out.append(r"\end{groupplot}")
    entries = "\n".join(
        r"\addlegendimage{%s, line width=1.4pt}\addlegendentry{%s}" % (TIKZ_COLOR[STREAM_COLOUR[s]],
                                                                       STREAM_LABEL[s])
        for s in STREAM_COLOUR)
    out.append(LEGEND_AXIS % {"gap": "3pt", "columns": 3, "colsep": "12pt", "entries": entries})
    out.append(r"\end{tikzpicture}%")
    return "\n".join(out) + "\n"



# ==============================================================================
# Figure: distilbert_calibration
# ==============================================================================
def calibration_data():
    """Reliability curves and histograms of figure_calibration()."""
    # verbatim from figure_calibration()
    frame = pd.read_csv(RESULTS / "revision_distilbert_probabilities.csv")
    frame = frame[(frame["max_length"] == 160) & (frame["epoch"] == 1)]
    curves = {}
    for training, (style, marker) in TRAIN_STYLE.items():
        block = frame[
            (frame["training"] == training)
            & (frame["condition"] == "external_openai")
        ]
        centres, observed, _ = reliability(
            block["probability"].to_numpy(), block["label"].to_numpy()
        )
        curves[training] = {"x": centres, "y": observed, "n": len(block),
                            "seeds": sorted(block["seed"].unique())}
    histograms = {}
    for condition in ("linked_rewrites", "external_openai"):
        for training, (style, _) in TRAIN_STYLE.items():
            block = frame[
                (frame["training"] == training) & (frame["condition"] == condition)
            ]
            # ax.hist(block["probability"], bins=25, density=True) bins one dataset
            # with np.histogram(x, 25, range=None, density=True): 25 equal-width bins
            # over that dataset's own [min, max].
            values = block["probability"].to_numpy()
            density, edges = np.histogram(values, bins=25, weights=None, range=None, density=True)
            counts, _ = np.histogram(values, bins=edges)
            histograms[(condition, training)] = {"edges": edges, "density": density,
                                                 "counts": counts, "n": len(values),
                                                 "seeds": sorted(block["seed"].unique())}
    # view limits as matplotlib autoscales them
    rel_x = np.concatenate([[0, 1]] + [c["x"] for c in curves.values()])
    rel_y = np.concatenate([[0, 1]] + [c["y"] for c in curves.values()])
    limits = {"a": (autoscale(rel_x.min(), rel_x.max()), autoscale(rel_y.min(), rel_y.max()))}
    for letter, condition in (("b", "linked_rewrites"), ("c", "external_openai")):
        parts = [histograms[(condition, t)] for t in TRAIN_STYLE]
        edges = np.concatenate([p["edges"] for p in parts])
        top = max(p["density"].max() for p in parts)
        limits[letter] = (autoscale(edges.min(), edges.max()), autoscale(0.0, top, sticky=0.0))
    return curves, histograms, limits


# tick steps (x, y) per panel, as matplotlib chose them for the PNG
CAL_TICKS = {"a": (0.25, 0.2), "b": (0.25, 2.0), "c": (0.25, 2.5)}


def step_outline(edges, density):
    """Points of a const plot that traces matplotlib's histtype='step' outline."""
    xs = [edges[0]] + list(edges[:-1]) + [edges[-1], edges[-1]]
    ys = [0.0] + list(density) + [density[-1], 0.0]
    return xs, ys


def calibration_tikz(curves, histograms, limits) -> str:
    n_ext = curves["linked-augmented"]["n"]
    n_lnk = histograms[("linked_rewrites", "linked-augmented")]["n"]
    out = [header(
        "distilbert_calibration", "fig:calibration",
        "DistilBERT at the pre-registered checkpoint (length 160, epoch 1), pooled over seeds "
        "29, 42, 71: (a) reliability, generator-disjoint corpus, 15 equal-width bins; "
        "(b)-(c) predicted-probability densities, 25 bins over each dataset's own range "
        f"(n = {n_lnk} linked, {n_ext} generator-disjoint predictions per training regime).",
        "results/revision_distilbert_probabilities.csv, as in "
        "make_revision_figures.figure_calibration().")]
    out.append(r"\begin{tikzpicture}[color=black]")
    out.append(r"\definecolor{calGray}{HTML}{888888}% diagonal, matplotlib #888888")
    out.append(PANEL_STYLE)
    out.append(r"\pgfmathsetlengthmacro\calW{(0.96\linewidth-%s)/3}%% axis width" % CAL_FIXED)
    out.append(r"\pgfmathsetlengthmacro\calH{max(%s*\calW,%s)}%% axis height" % (CAL_RATIO, CAL_MIN_HEIGHT))
    out.append(r"""\begin{groupplot}[
  group style={group size=3 by 1, horizontal sep=%s},
  revision panel, width=\calW, height=\calH, title style={text width=\calW, align=flush left},
  xticklabel style={%s},
]""" % (CAL_SEP, number_format(2)))
    clean_opts = r"oiVermillion, opacity=0.65, line width=1.3pt, %s" % DOTTED  # alpha 0.65
    aug_opts = r"oiVermillion, line width=1.3pt"
    # (a) reliability
    (x0, x1), (y0, y1) = limits["a"]
    out.append(r"\nextgroupplot[title={(a) Reliability, \mbox{gen.-disjoint}},")
    out.append("  xlabel={Mean predicted probability}, ylabel={Observed frequency},")
    out.append("  xmin=%s, xmax=%s, ymin=%s, ymax=%s," % (fmt(x0), fmt(x1), fmt(y0), fmt(y1)))
    out.append("  xtick={%s}, ytick={%s}, yticklabel style={%s}," % (
        fmt_list(linear_ticks(CAL_TICKS["a"][0], x0, x1)), fmt_list(linear_ticks(CAL_TICKS["a"][1], y0, y1)),
        number_format(1)))
    out.append("  legend style={at={(0.03,0.97)}, anchor=north west}]")
    out.append(r"\addplot[forget plot, calGray, line width=0.8pt, %s] coordinates {(0,0) (1,1)};"
               % REF_DASH)
    for training, label, opts, mark in (("clean-only", "Clean", clean_opts, "*"),
                                        ("linked-augmented", "Augmented", aug_opts + ", line join=round",
                                         "square*")):
        c = curves[training]
        out.append(r"\addplot[%s, mark=%s, mark size=1.5pt, mark options={solid, line width=1pt}]"
                   % (opts, mark))
        out.append("  coordinates {%s};" % coordinates(c["x"], c["y"]))
        out.append(r"\addlegendentry{%s}" % label)
    # (b), (c) densities
    area = "revision area legend"
    for letter, condition, title in (("b", "linked_rewrites", "(b) Confidence, linked"),
                                     ("c", "external_openai", r"(c) Confidence, \mbox{gen.-disjoint}")):
        (x0, x1), (y0, y1) = limits[letter]
        xstep, ystep = CAL_TICKS[letter]
        out.append(r"\nextgroupplot[title={%s}," % title)
        out.append("  xlabel={Predicted probability}, ylabel={Density},")
        out.append("  xmin=%s, xmax=%s, ymin=%s, ymax=%s," % (fmt(x0), fmt(x1), fmt(y0), fmt(y1)))
        out.append("  xtick={%s}, ytick={%s}, yticklabel style={%s}%s]"
                   % (fmt_list(linear_ticks(xstep, x0, x1)), fmt_list(linear_ticks(ystep, y0, y1)),
                      number_format(0 if letter == "b" else 1),
                      ",\n  legend style={at={(0.5,0.97)}, anchor=north}" if letter == "b" else ""))
        for training, label, opts in (("clean-only", "Clean", clean_opts),
                                      ("linked-augmented", "Augmented", aug_opts)):
            h = histograms[(condition, training)]
            xs, ys = step_outline(h["edges"], h["density"])
            out.append("%% %s, %s: n = %d; counts per bin %s" % (condition, training, h["n"],
                                                                  " ".join(str(int(v)) for v in h["counts"])))
            out.append(r"\addplot[%s, const plot, %s]" % (opts, area))
            out.append("  coordinates {%s};" % coordinates(xs, ys))
            if letter == "b":
                out.append(r"\addlegendentry{%s}" % label)
    out.append(r"\end{groupplot}")
    out.append(r"\end{tikzpicture}%")
    return "\n".join(out) + "\n"



# ==============================================================================
# Figure: distilbert_learning_curve
# ==============================================================================
CONDITION_COLOUR = {
    "linked_rewrites": COLORS["Word TF--IDF"],
    "external_openai": COLORS["DistilBERT"],
}
CONDITION_COLUMN = {"linked_rewrites": "linked", "external_openai": "gd"}
LC_PANELS = [
    (160, "f1", "(a) F1, max length 160"),
    (320, "f1", "(b) F1, max length 320"),
    (160, "ece", "(c) ECE, max length 160"),
]


def learning_curve_data():
    """Seed-mean curves of figure_learning_curve(), keyed by panel."""
    frame = pd.read_csv(RESULTS / "revision_distilbert_learning_curve.csv")
    panels = []
    for max_length, metric, title in LC_PANELS:
        block = frame[frame["max_length"] == max_length]
        curves = {}
        for condition, colour in CONDITION_COLOUR.items():
            for training, (style, _) in TRAIN_STYLE.items():
                part = block[
                    (block["eval_condition"] == condition)
                    & (block["training"] == training)
                ]
                if part.empty:
                    continue
                curve = part.groupby("epoch_progress")[metric].mean()
                seeds = part.groupby("epoch_progress")["seed"].nunique()
                curves[(condition, training)] = {"x": curve.index.to_numpy(), "y": curve.to_numpy(),
                                                 "seeds_per_point": sorted(seeds.unique())}
        xs = np.concatenate([c["x"] for c in curves.values()] + [[1.0]])  # + ax.axvline(1.0)
        ys = np.concatenate([c["y"] for c in curves.values()])
        panels.append({"max_length": max_length, "metric": metric, "title": title, "curves": curves,
                       "xlim": autoscale(xs.min(), xs.max()), "ylim": autoscale(ys.min(), ys.max())})
    # sharex=True: one x range for all panels
    lo = min(p["xlim"][0] for p in panels)
    hi = max(p["xlim"][1] for p in panels)
    for panel in panels:
        panel["xlim"] = (lo, hi)
    return panels


def learning_curve_tikz(panels) -> str:
    out = [header(
        "distilbert_learning_curve", "fig:learningcurve",
        "DistilBERT learning curves, means over seeds 29, 42, 71, a checkpoint every tenth of an "
        "epoch; solid augmented, dotted clean; dashed line at the one-epoch stopping point.",
        "results/revision_distilbert_learning_curve.csv, as in "
        "make_revision_figures.figure_learning_curve().")]
    out.append(r"\begin{tikzpicture}[color=black]")
    out.append(r"\definecolor{lcGray}{HTML}{888888}% stopping-point line, matplotlib #888888")
    out.append(r"\definecolor{lcDark}{HTML}{444444}% legend keys for the line styles, matplotlib #444444")
    out.append(PANEL_STYLE)
    out.append(r"\pgfmathsetlengthmacro\lcW{(0.96\linewidth-%s)/3}%% axis width" % LC_FIXED)
    out.append(r"\pgfmathsetlengthmacro\lcH{max(%s*\lcW,%s)}%% axis height" % (LC_RATIO, LC_MIN_HEIGHT))
    # one table per training regime: its checkpoints are shared by all panels
    tables = {}
    for training in TRAIN_STYLE:
        cols = {}
        for panel in panels:
            for condition in CONDITION_COLOUR:
                curve = panel["curves"][(condition, training)]
                if "epoch" not in cols:
                    cols["epoch"] = curve["x"]
                assert np.array_equal(cols["epoch"], curve["x"]), "checkpoints differ between curves"
                cols["%s%s%d" % (CONDITION_COLUMN[condition], panel["metric"], panel["max_length"])] = curve["y"]
        name = r"\lcClean" if training == "clean-only" else r"\lcAugmented"
        tables[training] = name
        out.append("%% %s: epochs of training and seed means (linked = linked rewrites, gd = "
                   "generator-disjoint corpus; f1/ece at maximum length 160/320)" % training)
        out.append(r"\pgfplotstableread[row sep=\\]{" + table(cols) + "}" + name)
    x0, x1 = panels[0]["xlim"]
    out.append(r"""\begin{groupplot}[
  group style={group size=3 by 1, horizontal sep=%s},
  revision panel, width=\lcW, height=\lcH, title style={text width=\lcW, align=flush left},
  xmin=%s, xmax=%s, xtick={%s}, xticklabel style={%s},
  yticklabel style={%s},
  xlabel={Epochs of training},
]""" % (LC_SEP, fmt(x0), fmt(x1), fmt_list(linear_ticks(0.5, x0, x1)), number_format(1),
        number_format(2)))
    for panel in panels:
        y0, y1 = panel["ylim"]
        out.append(r"\nextgroupplot[title={%s}, ylabel={%s}," % (panel["title"],
                                                               "F1" if panel["metric"] == "f1" else "ECE"))
        out.append("  ymin=%s, ymax=%s, ytick={%s}]" % (fmt(y0), fmt(y1), fmt_list(linear_ticks(0.05, y0, y1))))
        for condition, colour in CONDITION_COLOUR.items():
            for training in TRAIN_STYLE:
                column = "%s%s%d" % (CONDITION_COLUMN[condition], panel["metric"], panel["max_length"])
                style = ("opacity=0.6, line width=1.3pt, line join=round, " + DOTTED  # alpha 0.6
                         if training == "clean-only" else "line width=1.3pt, line join=round")
                out.append(r"\addplot[%s, %s] table[x=epoch, y=%s] {%s};"
                           % (TIKZ_COLOR[colour], style, column, tables[training]))
        out.append(r"\draw[lcGray, line width=0.8pt, %s] ({axis cs:1,0} |- {rel axis cs:0,0})"
                   r" -- ({axis cs:1,0} |- {rel axis cs:0,1});%% one-epoch stopping point" % REF_DASH)
    out.append(r"\end{groupplot}")
    entries = "\n".join([
        r"\addlegendimage{oiBlue, line width=1.3pt}\addlegendentry{Linked}",
        r"\addlegendimage{oiVermillion, line width=1.3pt}\addlegendentry{Gen.-disjoint}",
        r"\addlegendimage{lcDark, line width=1.3pt}\addlegendentry{Augmented}",
        r"\addlegendimage{lcDark, line width=1.3pt, %s}\addlegendentry{Clean}" % DOTTED,
    ])
    out.append(LEGEND_AXIS % {"gap": "3pt", "columns": 4, "colsep": "12pt", "entries": entries})
    out.append(r"\end{tikzpicture}%")
    return "\n".join(out) + "\n"



# ==============================================================================
# Comparison with the matplotlib figures (--check)
# ==============================================================================
def _matplotlib_figures():
    """Run make_revision_figures' three figure functions and return their figures."""
    import matplotlib

    matplotlib.use("Agg")
    if "rev_common" not in sys.modules:  # the figure script needs only these three paths
        stand_in = types.ModuleType("rev_common")
        stand_in.FIGURES, stand_in.RESULTS, stand_in.STUDY = FIGURES, RESULTS, STUDY
        sys.modules["rev_common"] = stand_in
    sys.path.insert(0, str(HERE))
    import make_revision_figures as mrf

    figures = {}
    mrf.finish = lambda fig, output_dir, stem: figures.__setitem__(stem, fig)
    mrf.configure_style()
    mrf.figure_threshold_cost(None)
    mrf.figure_calibration(None)
    mrf.figure_learning_curve(None)
    return figures


def _same(a, b, what, exact=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = a.shape == b.shape and (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=1e-12, atol=1e-15))
    if not ok:
        raise AssertionError(f"{what}: {a[:5]} ... != {b[:5]} ...")
    return a.size


def _visible(ticks, lim):
    return [t for t in ticks if lim[0] - 1e-12 <= t <= lim[1] + 1e-12]


def check(data) -> None:
    figures = _matplotlib_figures()
    compared = {}
    # threshold_cost
    if "threshold_cost" in data:
        n = 0
        for ax, panel in zip(figures["threshold_cost"].axes, data["threshold_cost"]):
            lines = ax.get_lines()
            for k, (stream, curve) in enumerate(panel["curves"].items()):
                n += _same(lines[2 * k].get_xdata(), curve["x"], f"{stream} x")
                n += _same(lines[2 * k].get_ydata(), curve["y"], f"{stream} y")
                n += _same(lines[2 * k + 1].get_xydata()[0], curve["best"], f"{stream} minimum")
                assert lines[2 * k].get_color().upper() == curve["colour"]
                assert lines[2 * k].get_label() == STREAM_LABEL[stream]
            n += _same(lines[-1].get_ydata(), [panel["reference"]] * 2, "pi*r line")
            n += _same(ax.get_xlim(), panel["xlim"], "xlim", exact=False)
            n += _same(ax.get_ylim(), panel["ylim"], "ylim", exact=False)
            n += _same(_visible(ax.get_xticks(), ax.get_xlim()), panel["xticks"], "xticks", exact=False)
            n += _same(_visible(ax.get_yticks(), ax.get_ylim()), panel["yticks"], "yticks", exact=False)
            minor = ax.yaxis.get_minor_locator()()
            n += _same(_visible(minor, ax.get_ylim()), panel["yminor"], "minor yticks", exact=False)
        compared["threshold_cost"] = n
    # distilbert_calibration
    if "distilbert_calibration" in data:
        curves, histograms, limits = data["distilbert_calibration"]
        axes = figures["distilbert_calibration"].axes
        n = 0
        lines = axes[0].get_lines()
        n += _same(lines[0].get_xydata(), [[0, 0], [1, 1]], "diagonal")
        for line, training in zip(lines[1:], TRAIN_STYLE):
            n += _same(line.get_xdata(), curves[training]["x"], f"reliability x {training}")
            n += _same(line.get_ydata(), curves[training]["y"], f"reliability y {training}")
        for ax, letter, condition in ((axes[0], "a", None), (axes[1], "b", "linked_rewrites"),
                                      (axes[2], "c", "external_openai")):
            n += _same(ax.get_xlim(), limits[letter][0], f"xlim {letter}", exact=False)
            n += _same(ax.get_ylim(), limits[letter][1], f"ylim {letter}", exact=False)
            n += _same(_visible(ax.get_xticks(), ax.get_xlim()),
                       linear_ticks(CAL_TICKS[letter][0], *limits[letter][0]), f"xticks {letter}", exact=False)
            n += _same(_visible(ax.get_yticks(), ax.get_ylim()),
                       linear_ticks(CAL_TICKS[letter][1], *limits[letter][1]), f"yticks {letter}", exact=False)
            if condition is None:
                continue
            for patch, training in zip(ax.patches, TRAIN_STYLE):
                h = histograms[(condition, training)]
                xs, ys = step_outline(h["edges"], h["density"])
                # const plot: horizontal to the next x, then vertical -> explicit outline
                outline = [(xs[0], ys[0])]
                for (xa, ya), (xb, yb) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
                    outline += [(xb, ya), (xb, yb)]
                outline = [p for i, p in enumerate(outline) if i == 0 or p != outline[i - 1]]
                vertices = [tuple(v) for v in patch.get_path().vertices.tolist()]
                vertices = [v for i, v in enumerate(vertices) if i == 0 or v != vertices[i - 1]]
                n += _same(vertices, outline, f"histogram {condition} {training}")
        compared["distilbert_calibration"] = n
    # distilbert_learning_curve
    if "distilbert_learning_curve" in data:
        n = 0
        for ax, panel in zip(figures["distilbert_learning_curve"].axes, data["distilbert_learning_curve"]):
            lines = ax.get_lines()
            for line, curve in zip(lines[:4], panel["curves"].values()):
                n += _same(line.get_xdata(), curve["x"], f"{panel['title']} x")
                n += _same(line.get_ydata(), curve["y"], f"{panel['title']} y")
            n += _same(lines[4].get_xdata(), [1.0, 1.0], "stopping line")
            n += _same(ax.get_xlim(), panel["xlim"], "xlim", exact=False)
            n += _same(ax.get_ylim(), panel["ylim"], "ylim", exact=False)
            n += _same(_visible(ax.get_yticks(), ax.get_ylim()),
                       linear_ticks(0.05, *panel["ylim"]), "yticks", exact=False)
            n += _same(_visible(ax.get_xticks(), ax.get_xlim()),
                       linear_ticks(0.5, *panel["xlim"]), "xticks", exact=False)
        compared["distilbert_learning_curve"] = n
    for name, n in compared.items():
        print(f"  check {name}: {n} values equal to the matplotlib figure")


# ==============================================================================
def summary(name, data) -> None:
    if name == "threshold_cost":
        for panel in data:
            minima = ", ".join(f"{STREAM_COLUMN[s]} {c['best'][0]:.3f} ({c['best'][1]:.4f})"
                               for s, c in panel["curves"].items())
            print(f"  ({panel['letter']}) pi = {panel['prevalence']:.3f}: minima {minima}; pi*r = "
                  f"{panel['reference']:g}")
    elif name == "distilbert_calibration":
        curves, histograms, _ = data
        for training, c in curves.items():
            print(f"  reliability {training}: {len(c['x'])} bins, n = {c['n']}, seeds {c['seeds']}")
        for (condition, training), h in histograms.items():
            print(f"  {condition} {training}: n = {h['n']}, range [{h['edges'][0]:.4f}, "
                  f"{h['edges'][-1]:.4f}], peak density {h['density'].max():.3f}")
    else:
        for panel in data:
            parts = []
            for (condition, training), c in panel["curves"].items():
                at_one = c["y"][np.isclose(c["x"], 1.0)][0]
                parts.append(f"{CONDITION_COLUMN[condition]}/{training[:5]} {at_one:.4f}")
            print(f"  {panel['title']}: value at epoch 1.0: " + ", ".join(parts)
                  + f"; seeds per point {sorted({s for c in panel['curves'].values() for s in c['seeds_per_point']})}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--copy-to", default=str(BUNDLE),
                        help="also copy the .tex files here (default: the manuscript bundle; 'none': no copy)")
    parser.add_argument("--check", action="store_true",
                        help="compare every plotted value with the matplotlib figure functions")
    args = parser.parse_args()
    TIKZ_DIR.mkdir(parents=True, exist_ok=True)
    copy_to = None if args.copy_to.lower() == "none" else Path(args.copy_to)
    if copy_to is not None and not copy_to.is_dir():
        print(f"no folder {copy_to}: the .tex files are not copied")
        copy_to = None
    builders = {
        "threshold_cost": (threshold_cost_data, lambda d: threshold_cost_tikz(d)),
        "distilbert_calibration": (calibration_data, lambda d: calibration_tikz(*d)),
        "distilbert_learning_curve": (learning_curve_data, lambda d: learning_curve_tikz(d)),
    }
    data = {}
    for name in args.only:
        prepare, draw = builders[name]
        data[name] = prepare()
        target = TIKZ_DIR / f"tikz_{name}.tex"
        target.write_text(draw(data[name]))
        print(f"wrote {target.relative_to(STUDY)}")
        summary(name, data[name])
        if copy_to is not None:
            shutil.copy2(target, copy_to / target.name)
            print(f"  copied to {copy_to / target.name}")
    if args.check:
        check(data)


if __name__ == "__main__":
    main()
