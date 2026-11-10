#!/usr/bin/env python3
"""TikZ/pgfplots versions of the evaluation-ladder and era-matched figures.

Writes the two figures of the "ladder_era" group as pgfplots pictures with their
data inline, so that the manuscript can typeset them instead of the PNG files:

* ``evaluation_ladder`` (fig:ladder)
    (a) F1 at threshold 0.5 at L1, L2, L2 cluster-deduplicated and L3, on a
        vertical axis broken between 0.48 and 0.74;
    (b) recall at threshold 0.5 at L1, L2, L3 and on the era-matched Nazario
        corpus (2015-2022 and 2023-2025).
  Same data as figure_evaluation_ladder() in make_r4_revised_figures.py.

* ``era_matched`` (fig:era)
    recall on era-matched real phishing by collection year, (a) at threshold 0.5
    and (b) at the validation cost-minimizing threshold (pi = 1%, r = 10).
  Same data as figure_era_matched() in make_r4_new_figures.py.

Every plotted number comes from the data code of those two scripts: the
helpers of make_r4_revised_figures.py are imported and called exactly as
figure_evaluation_ladder() calls them, and the loading block of
figure_era_matched(), which is not a separate function, is copied verbatim
into era_matched_data() below (make_r4_new_figures._by_year is imported).
Both modules import rev_common, which imports run_experiment, which imports
sentence_transformers (and with it torch) at module level although no figure
code uses it; this script first registers the same placeholder module that
rev_common installs when that package is absent, so neither torch nor a model
is loaded. Nothing is refitted; the script runs in a few seconds.

Inputs (results/):
    evaluation_ladder: revision_r3_multisplit_metrics.csv,
        revision_r4_multisplit100_metrics.csv, revision_distilbert_stream_metrics.csv,
        revision_r3_distilbert_multisplit_metrics.csv,
        revision_r4_distilbert_multisplit_metrics.csv,
        revision_r3_redundancy_sensitivity.csv, revision_r4_era_format_metrics.csv,
        revision_r4_era_format_distilbert_metrics.csv (+ its _run_info.json)
    era_matched: revision_r4_era_by_year.csv, revision_r4_era_format_summary.csv,
        revision_r4_era_format_distilbert_by_year.csv,
        revision_r4_era_format_distilbert_metrics.csv

Outputs:
    figures/tikz/tikz_evaluation_ladder.tex and figures/tikz/tikz_era_matched.tex,
    and a copy of each in --copy-to (default: the manuscript bundle, the folder
    manuscriptR4/ next to artifact/; skipped if it does not exist).
    Each file holds one tikzpicture (no float, caption or label) with its data
    inline. It needs pgfplots (compat 1.18; era_matched also the fillbetween
    library), graphicx, and the Okabe-Ito colors oiBlue, oiGreen, oiVermillion of
    the paper preamble. Its width is 0.96\\linewidth (evaluation_ladder) or
    \\linewidth (era_matched), the widths at which the PNG files were included:
    the picture is laid out for that width, but never narrower than in the
    paper, and \\resizebox scales it to the line, so that in a narrower line
    (the quotes of the response letter: 374.6pt, Computer Modern) it shrinks as
    the PNG did instead of overrunning the margin by 73-95pt with colliding
    labels. In the paper the scale factor is 1.0000. --bare writes the plain
    tikzpicture without the \\resizebox (identical in the paper).

Usage (from artifact/eswa_study/revision/):
    python tikz_ladder_era.py                      # both figures
    python tikz_ladder_era.py --only era_matched   # selected figures
    python tikz_ladder_era.py --copy-to DIR        # copy to DIR instead
    python tikz_ladder_era.py --no-copy            # artifact folder only
    python tikz_ladder_era.py --bare               # without the \\resizebox
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import types
from pathlib import Path

# --------------------------------------------------------------------------
# import the two figure scripts without torch
# --------------------------------------------------------------------------
if "sentence_transformers" not in sys.modules:  # the placeholder of rev_common.py
    _stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # noqa: D401 - placeholder, never instantiated here
        def __init__(self, *args, **kwargs):
            raise RuntimeError("sentence-transformers is not loaded by tikz_ladder_era.py")

    _stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = _stub

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

import make_r4_new_figures as NEW  # noqa: E402  (figure_era_matched)
import make_r4_revised_figures as REV  # noqa: E402  (figure_evaluation_ladder)
from rev_common import FIGURES, RESULTS, STUDY  # noqa: E402

TIKZ_DIR = FIGURES / "tikz"
BUNDLE = STUDY.parent.parent / "manuscriptR4"
GENERATOR = "artifact/eswa_study/revision/tikz_ladder_era.py"

# model keys used for the table names in the .tex files
FAMILY_KEY = {"Word TF-IDF": "w", "MiniLM": "m", "DistilBERT": "d"}
TRAINING_KEY = {"clean": "c", "augmented": "a"}
FAMILY_TEX = {"Word TF-IDF": "Word TF--IDF", "MiniLM": "MiniLM", "DistilBERT": "DistilBERT"}
FAMILY_COLOR = {"Word TF-IDF": "oiBlue", "MiniLM": "oiGreen", "DistilBERT": "oiVermillion"}
ERA_FAMILY_NAME = {"word": "Word TF-IDF", "minilm": "MiniLM", "distilbert": "DistilBERT"}

# the width the PNG had in the paper (elsarticle 5p: \textwidth = 522pt)
PAPER_TEXTWIDTH_PT = 522.0


def check_colors() -> None:
    """The family colors of the figure scripts must be the preamble's Okabe-Ito colors."""
    expected = {"Word TF-IDF": "#0072B2", "MiniLM": "#009E73", "DistilBERT": "#D55E00"}
    for family, hex_code in expected.items():
        if REV.colour(family).upper() != hex_code:
            raise ValueError(f"{family}: color {REV.colour(family)} is not {hex_code}")
    for name, hex_code in NEW.FAMILY.values():
        if hex_code.upper() != expected[name]:
            raise ValueError(f"{name}: color {hex_code} is not {expected[name]}")


def num(value: float, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def table(name: str, header: list[str], rows: list[list[str]], comment: str) -> str:
    """An inline \\pgfplotstableread block (row sep=\\\\) with aligned columns."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    lines = [f"  % {comment}", "  \\pgfplotstableread[row sep=\\\\]{"]
    for cells in [header] + rows:
        lines.append("    " + " ".join(c.rjust(w) for c, w in zip(cells, widths)) + " \\\\")
    lines.append(f"  }}\\{name}%")
    return "\n".join(lines)


def wrapping(width: str, fraction: float, bare: bool) -> dict:
    """Template fields for the width handling (see the module docstring)."""
    if bare:
        return {"open_wrap": "", "close_wrap": "", "layout_width": width, "graphicx": "",
                "width_note": " (laid out for the line; the text does not shrink)"}
    design = f"{fraction * PAPER_TEXTWIDTH_PT:.2f}pt"
    return {"open_wrap": f"\\resizebox{{{width}}}{{!}}{{%\n", "close_wrap": "}%\n",
            "layout_width": f"max({width},{design})", "graphicx": ", graphicx",
            "width_note": f":\n% laid out for that width, but never narrower than in the paper "
                          f"({design}), and\n% scaled to the line by \\resizebox (factor 1.0000 "
                          f"in the paper; in a narrower\n% line, such as a quote of the response "
                          f"letter, it shrinks as the PNG did)"}


# ==========================================================================
# evaluation_ladder
# ==========================================================================
LADDER_LEVELS = ["L1", "L2", "L2dedup", "L3"]
# tick labels: the `names` dict of figure_evaluation_ladder() and the labels of
# RECALL_LEVELS in make_r4_revised_figures.py, in TeX
LADDER_NAMES = {"L1": "L1\\\\linked\\\\rewrites", "L2": "L2\\\\gen.-\\\\disjoint",
                "L2dedup": "L2\\\\dedup.\\\\(cos.\\ 0.90)", "L3": "L3\\\\human\\\\(2000s)"}
RECALL_NAMES = ["L1\\\\linked\\\\rewrites", "L2\\\\gen.-\\\\disjoint", "L3\\\\human\\\\(2000s)",
                "Nazario\\\\2015--2022", "Nazario\\\\2023--2025"]
# axis limits and ticks of figure_evaluation_ladder()
A_TOP = (0.74, 1.0)
A_BOTTOM = (0.0, 0.48)
B_LIMITS = (0.0, 1.02)
A_TOP_TICKS = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
A_BOTTOM_TICKS = [0.0, 0.1, 0.2, 0.3, 0.4]
B_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def ladder_data(allow_partial: bool) -> dict:
    """The numbers of figure_evaluation_ladder(), by the same calls."""
    light, bert = REV.lightweight_metrics(), REV.distilbert_metrics()
    f1 = REV.f1_levels(light, bert)
    era_bert, status = REV.era_distilbert(allow_partial)
    print(f"  {status}")
    recall = REV.recall_levels(REV.era_lightweight(), era_bert)
    if era_bert is None:
        print("  Fig 7b: DistilBERT OMITTED from panel (b), as in the PNG script")
    if len(REV.RECALL_LEVELS) != len(RECALL_NAMES):
        raise ValueError("RECALL_LEVELS of make_r4_revised_figures.py changed")

    levels = LADDER_LEVELS
    offsets = REV.model_offsets(REV.MODELS)
    # K per level (verbatim from figure_evaluation_ladder)
    k_light = {lvl: f1[("Word TF-IDF", "clean")][lvl]["k"] for lvl in levels}
    k_bert = {lvl: f1[("DistilBERT", "clean")].get(lvl, {}).get("k") for lvl in levels}
    for model in REV.MODELS:  # every model of a family must have the same K at a level
        reference = k_bert if model[0] == "DistilBERT" else k_light
        for lvl in levels:
            if model in f1 and lvl in f1[model] and f1[model][lvl]["k"] != reference[lvl]:
                raise ValueError(f"K differs within a level: {model} {lvl}")

    present = [model for model in REV.MODELS if model in recall]
    offsets_b = REV.model_offsets(present, step=0.12 if len(present) == 6 else 0.14)
    k_b_light = {recall[("Word TF-IDF", "clean")][i]["k"] for i in range(len(REV.RECALL_LEVELS))}
    if len(k_b_light) != 1:
        raise ValueError("lightweight K differs between recall levels")
    k_b_light = k_b_light.pop()
    k_b_bert = recall[("DistilBERT", "clean")][0]["k"] if ("DistilBERT", "clean") in recall else None

    # a broken axis hides whatever falls into the gap: every mark and whisker end
    # of (a) must lie inside one of the two parts, and (b) inside its axis
    for model, entry in f1.items():
        for lvl, stats in entry.items():
            values = [stats["mean"]] + [stats[k] for k in ("low", "high") if k in stats]
            inside = [A_TOP[0] <= v <= A_TOP[1] or A_BOTTOM[0] <= v <= A_BOTTOM[1] for v in values]
            if not all(inside):
                raise ValueError(f"{model} {lvl}: {values} falls into the axis break")
    for model, entry in recall.items():
        for stats in entry:
            if not B_LIMITS[0] <= stats["low"] <= stats["high"] <= B_LIMITS[1]:
                raise ValueError(f"{model}: recall interval outside the axis of (b)")
    dedup_k = {f1[model]["L2dedup"]["k"] for model in REV.MODELS if "L2dedup" in f1[model]}
    if len(dedup_k) != 1:
        raise ValueError("the L2 dedup. means rest on different numbers of partitions")
    return {"f1": f1, "recall": recall, "offsets": offsets, "offsets_b": offsets_b,
            "present": present, "k_light": k_light, "k_bert": k_bert,
            "k_b_light": k_b_light, "k_b_bert": k_b_bert, "dedup_k": dedup_k.pop()}


def model_key(model) -> str:
    return FAMILY_KEY[model[0]] + TRAINING_KEY[model[1]]


def interval_rows(points) -> list[list[str]]:
    rows = []
    for x, stats in points:
        rows.append([num(x, 4), num(stats["mean"]), num(stats["low"]), num(stats["high"]),
                     num(stats["high"] - stats["mean"]), num(stats["mean"] - stats["low"])])
    return rows


def ladder_tex(data: dict, bare: bool = False) -> str:
    f1, recall = data["f1"], data["recall"]
    header = ["x", "mean", "low", "high", "up", "down"]
    blocks = []
    # (a): one table per model, the levels with an interval (L1, L2, L3)
    for model in REV.MODELS:
        levels = [lvl for lvl in LADDER_LEVELS if lvl in f1[model] and "low" in f1[model][lvl]]
        points = [(LADDER_LEVELS.index(lvl) + data["offsets"][model], f1[model][lvl])
                  for lvl in levels]
        k_text = ", ".join(f"{lvl} {f1[model][lvl]['k']}" for lvl in levels)
        blocks.append(table(f"lgF{model_key(model)}", header, interval_rows(points),
                            f"(a) {model[0]}, {REV.TRAINING_LABEL[model[1]]}: F1 at "
                            f"{', '.join(levels)} (partitions: {k_text})"))
    # (b): one table per model present
    for model in data["present"]:
        points = [(position + data["offsets_b"][model], stats)
                  for position, stats in enumerate(recall[model])]
        blocks.append(table(f"lgR{model_key(model)}", header, interval_rows(points),
                            f"(b) {model[0]}, {REV.TRAINING_LABEL[model[1]]}: recall at L1, L2, "
                            f"L3, Nazario 2015-2022, 2023-2025 (partitions: {recall[model][0]['k']})"))
    dedup = [model for model in REV.MODELS if "L2dedup" in f1[model]]

    def points_a(model) -> str:
        style = "lg clean" if model[1] == "clean" else "lg aug"
        return (f"    \\addplot[{FAMILY_COLOR[model[0]]}, {style}, lg bars] table[x=x, y=mean, "
                f"y error plus=up, y error minus=down] {{\\lgF{model_key(model)}}};")

    def points_dedup(model) -> str:
        style = "lg dedup clean" if model[1] == "clean" else "lg dedup aug"
        x = LADDER_LEVELS.index("L2dedup") + data["offsets"][model]
        mean = f1[model]["L2dedup"]["mean"]
        return (f"    \\addplot[{FAMILY_COLOR[model[0]]}, {style}] coordinates "
                f"{{({num(x, 4)},{num(mean)})}}; % L2 dedup. F1 of {model[0]}, "
                f"{REV.TRAINING_LABEL[model[1]]}: mean of {data['dedup_k']} partitions")

    def points_b(model) -> str:
        style = "lg clean" if model[1] == "clean" else "lg aug"
        return (f"    \\addplot[{FAMILY_COLOR[model[0]]}, {style}, lg bars] table[x=x, y=mean, "
                f"y error plus=up, y error minus=down] {{\\lgR{model_key(model)}}};")

    def k_pair(level) -> str:
        light, bert = data["k_light"][level], data["k_bert"][level]
        return f"$K = {light}\\,/\\,{bert}$" if bert else f"$K = {light}\\,/$\\,--"

    a_ticklabels = ",\n        ".join(f"{{{LADDER_NAMES[lvl]}\\\\{k_pair(lvl)}}}"
                                     for lvl in LADDER_LEVELS)
    b_ticklabels = ",\n        ".join(f"{{{label}}}" for label in RECALL_NAMES)
    if data["k_b_bert"]:
        k_text_b = (f"$K = {data['k_b_light']}$ partitions (lightweight)\\\\"
                    f"$K = {data['k_b_bert']}$ (DistilBERT)")
    else:
        k_text_b = f"$K = {data['k_b_light']}$ partitions\\\\(lightweight models)"

    # legend: matplotlib fills its 4 x 2 legend column by column; the matrix is row-wise
    key_rows = []
    for training in REV.TRAININGS:
        cells = []
        for family in REV.FAMILIES:
            fill = "white" if training == "clean" else FAMILY_COLOR[family]
            cells.append(f"\\draw[{FAMILY_COLOR[family]}, line width=1.1pt, fill={fill}] "
                         f"(-5.6pt,\\lgKeyY) circle (2.3pt); \\node[lg key] "
                         f"{{{FAMILY_TEX[family]}, {REV.TRAINING_LABEL[training]}}};")
        if training == "clean":
            cells.append("\\path plot[mark=lgX, mark size=2.6pt, mark options={draw=lgKey, "
                         "fill=lgKey, line width=1pt}] coordinates {(-5.6pt,\\lgKeyY)}; "
                         f"\\node[lg key] {{L2 dedup.: mean of {data['dedup_k']} partitions, "
                         "no interval};")
        else:
            cells.append("\\draw[lgKey, line width=0.9pt] (-5.6pt,\\lgKeyY-3.4pt) -- "
                         "(-5.6pt,\\lgKeyY+3.4pt); \\draw[lgKey, line width=0.8pt] "
                         "(-7.2pt,\\lgKeyY-3.4pt) -- (-4pt,\\lgKeyY-3.4pt) (-7.2pt,\\lgKeyY+3.4pt)"
                         " -- (-4pt,\\lgKeyY+3.4pt); \\node[lg key] "
                         "{2.5th--97.5th percentile across partitions};")
        key_rows.append("      " + "\n      \\& ".join(cells) + " \\\\")

    return LADDER_TEMPLATE.format(
        generator=GENERATOR, tables="\n".join(blocks),
        plots_a="\n".join(points_a(model) for model in REV.MODELS),
        plots_dedup="\n".join(points_dedup(model) for model in dedup),
        plots_b="\n".join(points_b(model) for model in data["present"]),
        a_ticklabels=a_ticklabels, b_ticklabels=b_ticklabels, k_text_b=k_text_b,
        legend="\n".join(key_rows), **wrapping("0.96\\linewidth", 0.96, bare),
        a_top_min=f"{A_TOP[0]:g}", a_top_max=f"{A_TOP[1]:g}",
        a_bot_min=f"{A_BOTTOM[0]:g}", a_bot_max=f"{A_BOTTOM[1]:g}",
        b_min=f"{B_LIMITS[0]:g}", b_max=f"{B_LIMITS[1]:g}",
        a_top_ticks=",".join(f"{t:g}" for t in A_TOP_TICKS),
        a_top_labels=",".join(f"{t:.2f}" for t in A_TOP_TICKS),
        a_bottom_ticks=",".join(f"{t:g}" for t in A_BOTTOM_TICKS),
        a_bottom_labels=",".join(f"{t:.1f}" for t in A_BOTTOM_TICKS),
        b_ticks=",".join(f"{t:g}" for t in B_TICKS),
        b_labels=",".join(f"{t:.1f}" for t in B_TICKS),
        break_low=f"{A_BOTTOM[1]:.2f}", break_high=f"{A_TOP[0]:.2f}",
    )


# The picture. Doubled braces are literal braces (str.format). Geometry follows
# figure_evaluation_ladder(): panel widths 1 : 1.06, the two parts of (a) 1.3 : 1.
# \lgL and \lgG hold the label widths that lie outside the axes (measured in the
# paper's Times), so that the picture is exactly 0.96\linewidth wide there.
LADDER_TEMPLATE = r"""% tikz_evaluation_ladder.tex -- figure fig:ladder (evaluation_ladder) in pgfplots.
% Written by {generator} from the data code of
% figure_evaluation_ladder() in make_r4_revised_figures.py; regenerate, do not edit.
% One tikzpicture, 0.96\linewidth wide{width_note}.
% Needs pgfplots (compat 1.18){graphicx} and the colors oiBlue, oiGreen,
% oiVermillion of the preamble.
% Marks: means across partitions (open: clean-only, filled: augmented); whiskers:
% 2.5th-97.5th percentiles across partitions; crosses: L2 dedup. means (no interval).
{open_wrap}\begin{{tikzpicture}}[color=black, font=\scriptsize]
  \definecolor{{lgGrid}}{{HTML}}{{CBD3D8}}%
  \definecolor{{lgSep}}{{HTML}}{{E1E6E9}}%
  \definecolor{{lgShade}}{{HTML}}{{F1F4F6}}%
  \definecolor{{lgNote}}{{HTML}}{{56636B}}%
  \definecolor{{lgKey}}{{HTML}}{{52616B}}%
  \definecolor{{lgInk}}{{HTML}}{{1B1B1B}}%
  % filled x (matplotlib marker "X"): a plus of arm width 1/2, turned by 45 degrees
  \pgfdeclareplotmark{{lgX}}{{%
    \pgfpathmoveto{{\pgfqpoint{{-.5\pgfplotmarksize}}{{-\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{0pt}}{{-.5\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{.5\pgfplotmarksize}}{{-\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{\pgfplotmarksize}}{{-.5\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{.5\pgfplotmarksize}}{{0pt}}}}%
    \pgfpathlineto{{\pgfqpoint{{\pgfplotmarksize}}{{.5\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{.5\pgfplotmarksize}}{{\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{0pt}}{{.5\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{-.5\pgfplotmarksize}}{{\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{-\pgfplotmarksize}}{{.5\pgfplotmarksize}}}}%
    \pgfpathlineto{{\pgfqpoint{{-.5\pgfplotmarksize}}{{0pt}}}}%
    \pgfpathlineto{{\pgfqpoint{{-\pgfplotmarksize}}{{-.5\pgfplotmarksize}}}}%
    \pgfpathclose\pgfusepathqfillstroke}}%
  % ---- data --------------------------------------------------------------------
  % x: level position (L1 0, L2 1, L2 dedup. 2, L3 3 in (a); L1 0, L2 1, L3 2,
  % Nazario 2015-2022 3, 2023-2025 4 in (b)) plus the model's offset; mean, low,
  % high: mean and 2.5th/97.5th percentiles across partitions; up = high - mean,
  % down = mean - low (whisker lengths).
{tables}
  % ---- geometry ------------------------------------------------------------------
  \pgfmathsetlengthmacro\lgW{{{layout_width}}}%
  \pgfmathsetlengthmacro\lgL{{35.3pt}}% (a): y label and tick labels left of the axis
  \pgfmathsetlengthmacro\lgG{{42pt}}%   between the panels: (b) y label and tick labels
  \pgfmathsetlengthmacro\lgAW{{(\lgW-\lgL-\lgG)/2.06}}%
  \pgfmathsetlengthmacro\lgBW{{1.06*\lgAW}}%
  \pgfmathsetlengthmacro\lgTH{{0.168*\lgW}}%  upper part of (a)
  \pgfmathsetlengthmacro\lgBH{{\lgTH/1.3}}%   lower part of (a)
  \pgfmathsetlengthmacro\lgGap{{0.0105*\lgW}}%
  \pgfmathsetlengthmacro\lgFullH{{\lgTH+\lgGap+\lgBH}}%
  \pgfmathsetmacro\lgYlab{{(1-(\lgGap+\lgBH)/\lgTH)/2}}% middle of (a), in the upper part
  \def\lgKeyY{{2.2pt}}%
  \pgfplotsset{{
    lg axis/.style={{
      set layers, scale only axis, anchor=south west, clip=true, clip mode=individual,
      axis y line*=left, xmin=-0.5,
      axis line style={{black, line width=0.7pt}},
      tick style={{black, line width=0.8pt}}, tick align=outside, major tick length=3.5pt,
      major x tick style={{draw=none}},
      ymajorgrids, major grid style={{lgGrid, draw opacity=0.6, line width=0.5pt}},
      tick label style={{font=\scriptsize, black}},
      x tick label style={{align=center}},
      label style={{font=\footnotesize, black}},
      title style={{font=\footnotesize\bfseries, black, at={{(0,1)}}, anchor=south west,
        inner xsep=0pt, yshift=1pt}},
    }},
    lg bars/.style={{error bars/.cd, y dir=both, y explicit,
      error bar style={{line width=0.9pt}},
      error mark options={{rotate=90, mark size=1.6pt, line width=0.8pt}}}},
    lg clean/.style={{only marks, mark=*, mark size=2pt,
      mark options={{line width=1.1pt, fill=white}}}},
    lg aug/.style={{only marks, mark=*, mark size=2pt, mark options={{line width=1.1pt}}}},
    lg dedup clean/.style={{only marks, mark=lgX, mark size=2.6pt,
      mark options={{line width=1pt, fill=white}}}},
    lg dedup aug/.style={{only marks, mark=lgX, mark size=2.6pt,
      mark options={{line width=1pt}}}},
  }}%
  % ---- (a) lower part, {a_bot_min} to {a_bot_max} (both parts draw all marks; each clips to its range)
  \begin{{axis}}[lg axis, name=lgBot, at={{(0,0)}}, width=\lgAW, height=\lgBH,
      xmax=3.5, ymin={a_bot_min}, ymax={a_bot_max}, axis x line*=bottom,
      ytick={{{a_bottom_ticks}}}, yticklabels={{{a_bottom_labels}}},
      xtick={{0,1,2,3}}, xticklabels={{{a_ticklabels}}}]
    \begin{{pgfonlayer}}{{axis background}}
      \draw[lgSep, line width=0.7pt] (axis cs:0.5,{a_bot_min}) -- (axis cs:0.5,{a_bot_max})
        (axis cs:1.5,{a_bot_min}) -- (axis cs:1.5,{a_bot_max}) (axis cs:2.5,{a_bot_min}) -- (axis cs:2.5,{a_bot_max});
    \end{{pgfonlayer}}
{plots_a}
{plots_dedup}
    \node[anchor=north west, align=left, text=lgNote, inner sep=0pt]
      at (rel axis cs:0.015,0.95)
      {{$K$ = partitions: lightweight\,/\,DistilBERT\\vertical axis broken between {break_low} and {break_high}}};
  \end{{axis}}
  % ---- (a) upper part, {a_top_min} to {a_top_max}
  \begin{{axis}}[lg axis, name=lgTop, at={{(0,\lgBH+\lgGap)}}, width=\lgAW, height=\lgTH,
      xmax=3.5, ymin={a_top_min}, ymax={a_top_max}, axis x line=none,
      ytick={{{a_top_ticks}}}, yticklabels={{{a_top_labels}}},
      ylabel={{F1 at threshold 0.5}}, ylabel style={{at={{(ticklabel cs:\lgYlab)}}}},
      title={{(a) F1 at threshold 0.5}}]
    \begin{{pgfonlayer}}{{axis background}}
      \draw[lgSep, line width=0.7pt] (axis cs:0.5,{a_top_min}) -- (axis cs:0.5,{a_top_max})
        (axis cs:1.5,{a_top_min}) -- (axis cs:1.5,{a_top_max}) (axis cs:2.5,{a_top_min}) -- (axis cs:2.5,{a_top_max});
    \end{{pgfonlayer}}
{plots_a}
{plots_dedup}
  \end{{axis}}
  % break marks on the left spine
  \draw[lgInk, line width=0.8pt] ([shift={{(-3.5pt,-1.925pt)}}]lgTop.south west) -- ++(7pt,3.85pt);
  \draw[lgInk, line width=0.8pt] ([shift={{(-3.5pt,-1.925pt)}}]lgBot.north west) -- ++(7pt,3.85pt);
  % ---- (b) recall at threshold 0.5 -------------------------------------------------
  \begin{{axis}}[lg axis, name=lgB, at={{(\lgAW+\lgG,0)}}, width=\lgBW, height=\lgFullH,
      xmax=4.5, ymin={b_min}, ymax={b_max}, axis x line*=bottom,
      ytick={{{b_ticks}}}, yticklabels={{{b_labels}}},
      xtick={{0,1,2,3,4}}, xticklabels={{{b_ticklabels}}},
      ylabel={{Recall at threshold 0.5}}, title={{(b) Recall at threshold 0.5}}]
    \begin{{pgfonlayer}}{{axis background}}
      \draw[lgSep, line width=0.7pt] (axis cs:0.5,{b_min}) -- (axis cs:0.5,{b_max})
        (axis cs:1.5,{b_min}) -- (axis cs:1.5,{b_max}) (axis cs:2.5,{b_min}) -- (axis cs:2.5,{b_max})
        (axis cs:3.5,{b_min}) -- (axis cs:3.5,{b_max});
      \fill[lgShade] (axis cs:2.5,{b_min}) rectangle (axis cs:4.5,{b_max});
    \end{{pgfonlayer}}
{plots_b}
    \node[anchor=south, align=center, font=\scriptsize\itshape, text=lgNote, inner sep=0pt]
      at (axis cs:3.5,0.035) {{era-matched real phishing\\(all messages are phishing)}};
    \node[anchor=south west, align=left, text=lgNote, inner sep=0pt]
      at (rel axis cs:0.015,0.36) {{{k_text_b}}};
  \end{{axis}}
  % ---- shared legend ---------------------------------------------------------------
  \matrix[anchor=north, column sep=11pt, row sep=3.2pt, inner sep=0pt,
      ampersand replacement=\&,
      lg key/.style={{anchor=base west, inner sep=0pt, text=black}}]
    at ([yshift=-7pt]current bounding box.south) {{
{legend}
  }};
\end{{tikzpicture}}%
{close_wrap}"""


# ==========================================================================
# era_matched
# ==========================================================================
def era_matched_data() -> dict:
    """The data block of figure_era_matched() (make_r4_new_figures.py), verbatim."""
    years = pd.read_csv(RESULTS / "revision_r4_era_by_year.csv")
    summary = pd.read_csv(RESULTS / "revision_r4_era_format_summary.csv")
    years = years[(years["format"] == "published") & (years["stream"] == "nazario")]
    l3 = summary[(summary["format"] == "published") & (summary["stream"] == "L3_human")]
    reference = {(row["family"], row["training"]): row["recall_t0.5"] for _, row in l3.iterrows()}
    partitions = {family: years[years["family"] == family]["seed"].nunique()
                  for family in ("word", "minilm")}

    families = ["word", "minilm"]
    bert_path = RESULTS / "revision_r4_era_format_distilbert_by_year.csv"
    bert_metrics_path = RESULTS / "revision_r4_era_format_distilbert_metrics.csv"
    bert_expected = 5
    if bert_path.exists():
        bert = pd.read_csv(bert_path)
        bert = bert[(bert["format"] == "published") & (bert["stream"] == "nazario")]
        if len(bert):
            years = pd.concat([years, bert], ignore_index=True)
            families.append("distilbert")
            partitions["distilbert"] = bert["seed"].nunique()
            if bert_metrics_path.exists():
                metrics = pd.read_csv(bert_metrics_path)
                metrics = metrics[(metrics["format"] == "published")
                                  & (metrics["stream"] == "L3_human")]
                for training, block in metrics.groupby("training"):
                    reference[("distilbert", training)] = float(block["recall_t0.5"].mean())
            status = "complete" if partitions["distilbert"] >= bert_expected else "PARTIAL"
            print(f"  DistilBERT by-year file found: {partitions['distilbert']} of "
                  f"{bert_expected} partitions ({status}); seeds {sorted(bert['seed'].unique())}")
    else:
        print("  DistilBERT by-year file absent: figure drawn without DistilBERT")

    counts = (years[(years["family"] == "word") & (years["training"] == "clean")]
              .groupby("year")["n"].first().sort_index())
    # (end of the verbatim block) the curves exactly as the plotting loop computes them
    curves = {}
    for column, _ in ERA_PANELS:
        for family in families:
            for training in ("clean", "augmented"):
                curve = NEW._by_year(years, family, training, column)
                if not curve.empty:
                    curves[(column, family, training)] = curve
    high = max(value for key, value in reference.items() if key[0] in families)
    return {"curves": curves, "reference": reference, "partitions": partitions,
            "families": families, "counts": counts, "bert_expected": bert_expected,
            "high": high}


ERA_PANELS = [("recall_t0.5", "a"), ("recall_tcost", "b")]
ERA_STYLE = {"clean": ("era clean", 0.09), "augmented": ("era aug", 0.16)}  # line style, band alpha


def era_tex(data: dict, bare: bool = False) -> str:
    curves, families, counts = data["curves"], data["families"], data["counts"]
    blocks, plots = [], {}
    for column, letter in ERA_PANELS:
        bands, lines, refs = [], [], []
        for family in families:
            name = ERA_FAMILY_NAME[family]
            color = FAMILY_COLOR[name]
            for training in ("clean", "augmented"):
                curve = curves.get((column, family, training))
                if curve is None:
                    continue
                table_name = f"era{letter.upper()}{FAMILY_KEY[name]}{TRAINING_KEY[training]}"
                rows = [[str(int(year)), num(row["mean"]), num(row["sd"]),
                         num(row["mean"] - row["sd"]), num(row["mean"] + row["sd"])]
                        for year, row in curve.iterrows()]
                k = int(curve["partitions"].min())
                blocks.append(table(table_name, ["year", "mean", "sd", "lo", "hi"], rows,
                                    f"({letter}) {name}, {REV.TRAINING_LABEL[training]}: "
                                    f"{column} by year, mean and SD over {k} partitions; "
                                    "lo/hi = mean -/+ SD"))
                style, alpha = ERA_STYLE[training]
                if curve["partitions"].min() > 1:  # as the PNG: no band from one partition
                    bands.append(
                        f"    \\addplot[draw=none, name path={table_name}lo, forget plot] "
                        f"table[x=year, y=lo] {{\\{table_name}}};\n"
                        f"    \\addplot[draw=none, name path={table_name}hi, forget plot] "
                        f"table[x=year, y=hi] {{\\{table_name}}};\n"
                        f"    \\addplot[fill={color}, fill opacity={alpha:g}, draw=none, "
                        f"forget plot] fill between[of={table_name}lo and {table_name}hi];")
                lines.append(f"    \\addplot[{color}, {style}] table[x=year, y=mean] "
                             f"{{\\{table_name}}};")
                if column == "recall_t0.5" and (family, training) in data["reference"]:
                    value = data["reference"][(family, training)]
                    refs.append(f"    \\addplot[{color}, era ref {training}] coordinates "
                                f"{{(2014.5,{num(value)}) (2025.5,{num(value)})}}; "
                                f"% {name}, {REV.TRAINING_LABEL[training]}: L3 recall at 0.5")
        plots[letter] = ("\n".join(bands), "\n".join(refs), "\n".join(lines))

    first_year = int(counts.index[0])
    labels = []
    for year, n in counts.items():
        mark = "\\llap{$n$\\hspace{6.5pt}}" if int(year) == first_year else ""
        labels.append(f"{{{int(year)}\\\\[1.6pt]\\textcolor{{eraMuted}}{{{mark}{int(n)}}}}}")
    xticklabels = ",\n        ".join(", ".join(labels[i:i + 4]) for i in range(0, len(labels), 4))
    xticks = ",".join(str(int(year)) for year in counts.index)

    def family_label(family):  # the logic of family_label() in figure_era_matched()
        name = ERA_FAMILY_NAME[family]
        count = data["partitions"][family]
        if family == "distilbert" and count < data["bert_expected"]:
            return f"{FAMILY_TEX[name]} ({count} of {data['bert_expected']} partitions)"
        return f"{FAMILY_TEX[name]} ({count} partitions)"

    family_cells = [f"\\draw[{FAMILY_COLOR[ERA_FAMILY_NAME[f]]}, line width=1.6pt] "
                    f"(-19pt,\\eraKeyY) -- (-5pt,\\eraKeyY); \\node[era key] {{{family_label(f)}}};"
                    for f in families]
    style_cells = [
        "\\draw[eraInk, line width=1.35pt] (-19pt,\\eraKeyY) -- (-5pt,\\eraKeyY); "
        "\\draw[eraInk, line width=0.9pt, fill=eraInk] (-12pt,\\eraKeyY) circle (1.5pt); "
        "\\node[era key] {Augmented};",
        "\\draw[eraInk, line width=1.35pt, era dots] (-19pt,\\eraKeyY) -- (-5pt,\\eraKeyY); "
        "\\draw[eraInk, line width=0.9pt, fill=white] (-12pt,\\eraKeyY) circle (1.5pt); "
        "\\node[era key] {Clean-only};",
        "\\fill[eraInk, fill opacity=0.15] (-19pt,\\eraKeyY-2.6pt) rectangle (-5pt,\\eraKeyY+2.6pt); "
        "\\node[era key] {$\\pm 1$ SD over partitions};",
    ]
    while len(family_cells) < 3:
        family_cells.append("")
    legend = ("      " + "\n      \\& ".join(family_cells) + " \\\\\n"
              "      " + "\n      \\& ".join(style_cells) + " \\\\")

    bands_a, refs_a, lines_a = plots["a"]
    bands_b, _, lines_b = plots["b"]
    return ERA_TEMPLATE.format(
        generator=GENERATOR, tables="\n".join(blocks), xticks=xticks, xticklabels=xticklabels,
        bands_a=bands_a, refs_a=refs_a, lines_a=lines_a, bands_b=bands_b, lines_b=lines_b,
        note_y=num(data["high"] + 0.16), high=num(data["high"]), legend=legend,
        **wrapping("\\linewidth", 1.0, bare),
    )


# Geometry follows figure_era_matched(): two axes of 3.05 x 1.80 in, 0.28 in apart.
# \eraL holds the label widths left of panel (a) (measured in the paper's Times),
# so that the picture is exactly \linewidth wide there.
ERA_TEMPLATE = r"""% tikz_era_matched.tex -- figure fig:era (era_matched) in pgfplots.
% Written by {generator} from the data code of
% figure_era_matched() in make_r4_new_figures.py; regenerate, do not edit.
% One tikzpicture, \linewidth wide{width_note}.
% Needs pgfplots (compat 1.18) with the fillbetween library{graphicx} and the
% colors oiBlue, oiGreen, oiVermillion of the preamble.
% Lines: means over partitions (solid, filled: augmented; dotted, open: clean-only);
% bands: +/-1 SD over partitions; horizontal lines in (a): recall at 0.5 on the 2000s
% L3 human-authored phishing; n: Nazario messages per collection year.
{open_wrap}\begin{{tikzpicture}}[color=black, font=\scriptsize]
  \definecolor{{eraGrid}}{{HTML}}{{CBD3D8}}%
  \definecolor{{eraMuted}}{{HTML}}{{6B7B85}}%
  \definecolor{{eraInk}}{{HTML}}{{33414A}}%
  % ---- data --------------------------------------------------------------------
{tables}
  % ---- geometry ------------------------------------------------------------------
  \pgfmathsetlengthmacro\eraW{{{layout_width}}}%
  \pgfmathsetlengthmacro\eraL{{43pt}}%  (a): y label and tick labels left of the axis
  \pgfmathsetlengthmacro\eraG{{20.5pt}}% between the panels
  \pgfmathsetlengthmacro\eraAW{{(\eraW-\eraL-\eraG)/2}}%
  \pgfmathsetlengthmacro\eraH{{0.59*\eraAW}}%
  \def\eraKeyY{{2.2pt}}%
  \tikzset{{era dots/.style={{dash pattern=on 1.35pt off 2.23pt}}}}%
  \tikzset{{era box/.style={{anchor=north, align=center, fill=white, rounded corners=1.2pt,
    inner xsep=1.6pt, inner ysep=1.2pt, text=eraMuted, font=\scriptsize}}}}%
  \pgfplotsset{{
    era axis/.style={{
      set layers, scale only axis, anchor=south west, clip=true, clip mode=individual,
      width=\eraAW, height=\eraH,
      axis x line*=bottom, axis y line*=left,
      xmin=2014.5, xmax=2025.5, ymin=0, ymax=1.21,
      xtick={{{xticks}}}, ytick={{0,0.2,0.4,0.6,0.8,1}},
      axis line style={{black, line width=0.7pt}},
      tick style={{black, line width=0.8pt}}, tick align=outside,
      major tick length=3.5pt, major x tick style={{/pgfplots/major tick length=2pt}},
      ymajorgrids, major grid style={{eraGrid, draw opacity=0.6, line width=0.5pt}},
      tick label style={{font=\scriptsize, black}},
      x tick label style={{align=center}},
      label style={{font=\footnotesize, black}},
      xlabel={{Collection year (Nazario phishing corpus)}},
      xticklabels={{{xticklabels}}},
    }},
    era clean/.style={{line width=1.35pt, era dots, mark=*, mark size=1.5pt,
      mark options={{solid, line width=0.9pt, fill=white}}}},
    era aug/.style={{line width=1.35pt, mark=*, mark size=1.5pt,
      mark options={{line width=0.9pt}}}},
    era ref clean/.style={{line width=0.9pt, draw opacity=0.85,
      dash pattern=on 0.9pt off 1.49pt, forget plot}},
    era ref augmented/.style={{line width=0.9pt, draw opacity=0.85, forget plot}},
  }}%
  % ---- (a) threshold 0.5 -----------------------------------------------------------
  \begin{{axis}}[era axis, name=eraA, at={{(0,0)}},
      yticklabels={{0.0,0.2,0.4,0.6,0.8,1.0}},
      ylabel={{Recall on era-matched\\real phishing}}, ylabel style={{align=center}}]
    \begin{{pgfonlayer}}{{axis background}}
      \draw[eraMuted, line width=0.9pt, dash pattern=on 3.33pt off 1.44pt]
        (axis cs:2022.5,0) -- (axis cs:2022.5,1.21);
    \end{{pgfonlayer}}
{bands_a}
{refs_a}
{lines_a}
    \node[era box] at (axis cs:2022.5,1.19) {{public release of\\instruction-tuned chat models}};
    \node[anchor=south west, align=left, text=eraInk, inner sep=0pt] (eraNote)
      at (axis cs:2014.85,{note_y}) {{2000s L3 human phishing\\(recall at 0.5, one line per model)}};
    \draw[eraMuted, line width=0.6pt, shorten <=1.5pt] (eraNote) -- (axis cs:2015.6,{high});
  \end{{axis}}
  % ---- (b) cost-minimizing threshold -------------------------------------------------
  \begin{{axis}}[era axis, name=eraB, at={{(\eraAW+\eraG,0)}}, yticklabels={{}}]
    \begin{{pgfonlayer}}{{axis background}}
      \draw[eraMuted, line width=0.9pt, dash pattern=on 3.33pt off 1.44pt]
        (axis cs:2022.5,0) -- (axis cs:2022.5,1.21);
    \end{{pgfonlayer}}
{bands_b}
{lines_b}
    \node[era box] at (axis cs:2022.5,1.19) {{public release of\\instruction-tuned chat models}};
  \end{{axis}}
  % ---- titles ----------------------------------------------------------------------
  \path (eraA.north) ++(0,16pt) coordinate (eraTitle);
  \node[anchor=base west, inner sep=0pt, font=\footnotesize\bfseries]
    at (eraA.outer north west |- eraTitle) {{(a) Threshold 0.5}};
  \node[anchor=base west, inner sep=0pt, font=\footnotesize\bfseries]
    at (eraB.north west |- eraTitle) {{(b) Cost-minimizing threshold}};
  \node[anchor=base west, inner sep=0pt, text=eraMuted]
    at ([yshift=4.5pt]eraB.north west)
    {{validation threshold minimizing expected cost at $\pi = 1\%$, $r = 10$}};
  % ---- legend ----------------------------------------------------------------------
  \matrix[anchor=north, column sep=26pt, row sep=3.2pt, inner sep=0pt,
      ampersand replacement=\&,
      era key/.style={{anchor=base west, inner sep=0pt, text=black}}]
    at ([yshift=-6pt]current bounding box.south) {{
{legend}
  }};
\end{{tikzpicture}}%
{close_wrap}"""


# ==========================================================================
FIGURES_TEX = {
    "evaluation_ladder": lambda args: ladder_tex(ladder_data(args.allow_partial_distilbert_era),
                                                 args.bare),
    "era_matched": lambda args: era_tex(era_matched_data(), args.bare),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES_TEX),
                        help="write only these figures")
    parser.add_argument("--copy-to", type=Path, default=BUNDLE,
                        help="also copy the .tex files to this directory (default: the bundle)")
    parser.add_argument("--no-copy", action="store_true", help="do not copy the .tex files")
    parser.add_argument("--bare", action="store_true",
                        help="write the bare tikzpicture, laid out at the line width, without "
                             "the \\resizebox that lets it shrink in a narrower line")
    parser.add_argument("--allow-partial-distilbert-era", action="store_true",
                        help="as in make_r4_revised_figures.py: draw DistilBERT in the ladder's "
                             "panel (b) from an incomplete era-format run (>= 2 partitions)")
    args = parser.parse_args()
    check_colors()
    TIKZ_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(FIGURES_TEX):
        print(f"[{name}]")
        target = TIKZ_DIR / f"tikz_{name}.tex"
        target.write_text(FIGURES_TEX[name](args))
        print(f"  wrote {target.relative_to(STUDY.parent)}")
        if not args.no_copy:
            if args.copy_to.is_dir():
                shutil.copy2(target, args.copy_to / target.name)
                print(f"  copied to {args.copy_to / target.name}")
            else:
                print(f"  not copied: {args.copy_to} does not exist")


if __name__ == "__main__":
    main()
