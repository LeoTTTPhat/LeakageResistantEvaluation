#!/usr/bin/env python3
"""TikZ/pgfplots versions of the four "limitations" figures (round 4).

The manuscript draws every figure in TikZ. This script writes pgfplots pictures
that show the same data, panels, encodings and annotations as the matplotlib
figures of the same names:

* ``tikz_domain_adaptation.tex`` (fig:adaptation; matplotlib:
  make_limitation_figures.py, domain_adaptation_figure)
    (a) F1 at threshold 0.5 for each training source -> evaluation target,
        Word TF-IDF and MiniLM, bars = mean, error bars = SD over five seeds;
    (b) Word TF-IDF F1 at threshold 0.5 on human-authored mail against the
        number k of human-authored training messages (log axis): target only,
        target + synthetic upweighted, target + synthetic as collected, with
        the full-corpus ceiling and the synthetic-only reference lines.
  Data: results/revision_domain_adaptation_summary.csv,
        results/revision_fewshot_curve.csv

* ``tikz_adaptive_evasion.tex`` (fig:adaptive; make_limitation_figures.py,
  adaptive_figure)
    (a) evasion rate against the edit budget, mean +/- SD over the seeds;
    (b) recall at threshold 0.5 on adversarial messages, rows = attacked
        detector, columns = scoring detector, boxed = the attacked detector.
  Data: results/revision_adaptive_by_seed.csv,
        results/revision_adaptive_transfer.csv

* ``tikz_augmentation_dose.tex`` (fig:dose; make_figures.py, dose_figure)
    F1 by training exposure on (a) the linked rewrites and (b) the
    generator-disjoint corpus, mean +/- SD over the five reference partitions.
  Data: results/rigor_dose_summary.csv

* ``tikz_leakage_and_masking.tex`` (fig:stress; make_figures.py, stress_figure)
    (a) F1 under the source-group and the row-wise split, with the test-group
        overlap inset; (b) F1 before and after masking the 20 largest
        positive-coefficient unigrams; bars = mean, error bars = SD over the
        five reference partitions.
  Data: results/rigor_leakage_summary.csv, results/rigor_cue_mask_summary.csv

The data selection of each panel is copied verbatim from the matplotlib
function named above (make_limitation_figures.py cannot be imported here: its
rev_common import loads run_experiment, sentence-transformers and torch), and
the colors come from make_figures.COLORS, so every plotted number and color is
the one the PNG shows. Axis limits are the PNG's: explicit where the matplotlib
code sets them, matplotlib's default 5% margins otherwise. Each picture is
exactly one tikzpicture with its data inline (Editorial Manager flattens
folders), 0.94\\linewidth wide, Okabe-Ito colors from the paper preamble
(oiBlue, oiGreen, ...) and any other color defined inside the picture.

Outputs:
  figures/tikz/tikz_{domain_adaptation,adaptive_evasion,augmentation_dose,
  leakage_and_masking}.tex, and a copy of each in the manuscript bundle
  (--copy-to; default ../../../manuscriptR4 when that folder exists).

Run from artifact/eswa_study/revision/:
    python tikz_limitations.py [--copy-to DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import textwrap

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY / "results"
TIKZ_DIR = STUDY / "figures" / "tikz"
PROJECT = STUDY.parent.parent
DEFAULT_BUNDLE = PROJECT / "manuscriptR4"

sys.path.insert(0, str(STUDY))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

from make_figures import COLORS, GRID  # noqa: E402  (pandas/matplotlib only)

# Copied verbatim from make_limitation_figures.py.
MODEL_LABEL = {"word_lr": "Word TF-IDF", "minilm_lr": "MiniLM"}
MODEL_COLOR = {"word_lr": COLORS["Word TF--IDF"], "minilm_lr": COLORS["MiniLM"]}
ARM_STYLE = {
    "human_only_k": ("-", "o", "target only"),
    "mixed_balanced": ("--", "s", "target + synthetic (upweighted)"),
    "mixed": (":", "^", "target + synthetic (as collected)"),
}

# Okabe-Ito colors that the paper preamble defines.
PALETTE = {"#0072B2": "oiBlue", "#E69F00": "oiOrange", "#009E73": "oiGreen",
           "#D55E00": "oiVermillion", "#56B4E9": "oiSky", "#CC79A7": "oiPurple"}

GENERATOR = "artifact/eswa_study/revision/tikz_limitations.py"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def oi(hex_color: str) -> str:
    """Preamble name of an Okabe-Ito color used by the matplotlib script."""
    return PALETTE[hex_color.upper()]


def html(hex_color: str) -> str:
    return to_hex(hex_color).lstrip("#").upper()


def blend_on_white(hex_color: str, alpha: float) -> str:
    """The opaque color of ``hex_color`` drawn at ``alpha`` on white."""
    rgb = np.array(to_rgb(hex_color))
    return to_hex(alpha * rgb + (1 - alpha) * np.ones(3)).lstrip("#").upper()


def num(value: float) -> str:
    return f"{float(value):.6f}"


def mpl_margins(lo: float, hi: float, log: bool = False) -> tuple[float, float]:
    """matplotlib's default autoscaled view limits (axes.xmargin = 0.05)."""
    if log:
        a, b = np.log10(lo), np.log10(hi)
        pad = 0.05 * (b - a)
        return float(10 ** (a - pad)), float(10 ** (b + pad))
    pad = 0.05 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def table(columns: list[str], rows: list[list[str]], indent: str = "    ") -> str:
    """Rows of a \\pgfplotstableread[row sep=\\\\] block, aligned."""
    cells = [columns] + rows
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    return "\n".join(indent + " ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() + " \\\\"
                     for row in cells)


def latex_label(text: str) -> str:
    """matplotlib tick text -> pgfplots tick text (line breaks, arrows)."""
    return text.replace("\u2192", "$\\rightarrow$").replace("\n", "\\\\")


def tick_list(labels: list[str]) -> str:
    """{label}, {label}, ... for xticklabels/yticklabels, one per line."""
    return ",\n        ".join("{" + latex_label(label) + "}" for label in labels)


def fill(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace(f"<<{key}>>", value)
    if "<<" in template:
        raise ValueError(f"unfilled placeholder near {template[template.index('<<'):][:40]!r}")
    return template.lstrip("\n")


def header(name: str, label: str, document: str, matplotlib_source: str, inputs: list[str],
           needs: str = "pgfplots (compat 1.18, groupplots)") -> str:
    def wrap(text: str) -> list[str]:
        return textwrap.wrap(text, width=86, initial_indent="% ", subsequent_indent="% ",
                             break_on_hyphens=False)

    lines = [*wrap(f"tikz_{name}.tex -- figure {label} of {document}: {name}.png redrawn "
                   f"with the same data, panels and encodings ({matplotlib_source})."),
             f"% Written by {GENERATOR}; do not edit by hand.",
             "% Data: " + inputs[0], *("%       " + item for item in inputs[1:]),
             *wrap(f"Needs {needs} and the oi* colors of the paper preamble. Width "
                   "0.94\\linewidth; default color black (also inside a colored quote).")]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# data: copied from the matplotlib functions
# ----------------------------------------------------------------------------
def domain_adaptation_data() -> dict:
    """As make_limitation_figures.domain_adaptation_figure selects it."""
    summary = pd.read_csv(RESULTS / "revision_domain_adaptation_summary.csv")
    curve = pd.read_csv(RESULTS / "revision_fewshot_curve.csv")

    conditions = [
        ("synthetic_transfer", "linked_augmented", "synthetic\n\u2192 human"),
        ("human_in_domain", "human_only", "human\n\u2192 human"),
        ("reverse_to_L2", "human_only", "human\n\u2192 L2"),
        ("reverse_to_L1", "human_only", "human\n\u2192 L1"),
    ]
    width = 0.36
    offsets = {"word_lr": -width / 2, "minilm_lr": width / 2}
    bars = {}
    for model, offset in offsets.items():
        heights, errors = [], []
        for condition, training, _ in conditions:
            row = summary[
                (summary["condition"] == condition)
                & (summary["training"] == training)
                & (summary["model"] == model)
            ]
            heights.append(float(row["f1_mean"].iloc[0]))
            errors.append(float(row["f1_std"].iloc[0]))
        bars[model] = {"x": list(np.arange(len(conditions)) + offset), "f1": heights, "sd": errors}

    word = curve[curve["model"] == "word_lr"].copy()
    word["k"] = word["n_human_train_mean"].astype(int)
    arms = {}
    for arm, (style, marker, label) in ARM_STYLE.items():
        block = word[(word["condition"] == arm) & (word["k"] > 0)].sort_values("k")
        if block.empty:
            continue
        arms[arm] = {"k": block["k"].tolist(), "f1": block["f1_mean"].tolist(), "label": label}
    ceiling = float(
        summary.query("condition == 'human_in_domain' and model == 'word_lr'")
        ["f1_mean"].iloc[0]
    )
    floor = float(
        word[(word["condition"] == "mixed") & (word["k"] == 0)]["f1_mean"].iloc[0]
    )
    return {"conditions": conditions, "width": width, "bars": bars, "arms": arms,
            "ceiling": ceiling, "floor": floor}


def adaptive_data() -> dict:
    """As make_limitation_figures.adaptive_figure selects it."""
    by_seed = pd.read_csv(RESULTS / "revision_adaptive_by_seed.csv")
    transfer = pd.read_csv(RESULTS / "revision_adaptive_transfer.csv")

    budgets = [5, 10, 20, 30, 40]
    label = {
        "word_lr_clean_only": ("Word, clean", ":", "o"),
        "word_lr_linked_augmented": ("Word, augmented", "-", "s"),
        "charplusword_lr_linked_augmented": ("Char+word, augmented", "--", "^"),
    }
    curves = {}
    for model, (name, style, marker) in label.items():
        block = by_seed[by_seed["model"] == model]
        if block.empty:
            continue
        means = [block[f"success_at_{b}pct"].mean() for b in budgets]
        errors = [block[f"success_at_{b}pct"].std() for b in budgets]
        curves[model] = {"name": name, "mean": means, "sd": errors,
                         "color": COLORS["Word TF--IDF"] if model.startswith("word")
                         else COLORS["DistilBERT"]}

    order = [
        "word_lr_clean_only", "word_lr_linked_augmented",
        "charplusword_lr_linked_augmented", "minilm_lr_clean_only",
        "minilm_lr_linked_augmented",
    ]
    short = {
        "word_lr_clean_only": "Word\nclean", "word_lr_linked_augmented": "Word\naug.",
        "charplusword_lr_linked_augmented": "Char+word\naug.",
        "minilm_lr_clean_only": "MiniLM\nclean", "minilm_lr_linked_augmented": "MiniLM\naug.",
    }
    attacked = [m for m in order if m in set(transfer["attacked_model"])]
    grid = np.full((len(attacked), len(order)), np.nan)
    for i, source in enumerate(attacked):
        for j, target in enumerate(order):
            block = transfer[
                (transfer["attacked_model"] == source)
                & (transfer["scored_model"] == target)
            ]
            if not block.empty:
                grid[i, j] = block["recall_attacked"].mean()

    # the legend title of the PNG states these two counts; take them from the file
    n_partitions = sorted(set(by_seed.groupby("model")["seed"].nunique()))
    n_targets = sorted(set(by_seed["n_targets"]))
    if len(n_partitions) != 1 or len(n_targets) != 1:
        raise ValueError(f"unequal partitions {n_partitions} or targets {n_targets} per model")
    return {"budgets": budgets, "curves": curves, "order": order, "short": short,
            "attacked": attacked, "grid": grid, "n_partitions": n_partitions[0],
            "n_targets": int(n_targets[0])}


def dose_data() -> dict:
    """As make_figures.dose_figure selects it."""
    frame = pd.read_csv(RESULTS / "rigor_dose_summary.csv")
    conditions = [
        "canonical_only_1x",
        "canonical_repeated_6x",
        "entity_short_2x",
        "length_well_known_3x",
        "full_factorial_6x",
    ]
    labels = ["Canonical\n1x", "Canonical\n6x", "Entity\n2x", "Length\n3x", "Full\n6x"]
    families = [("Word LR", "Word TF--IDF"), ("MiniLM LR", "MiniLM")]
    panels = [
        ("linked_rewrites", "Linked rewrites", (0.925, 0.988)),
        ("external_openai", "Generator-disjoint corpus", (0.776, 0.96)),
    ]
    out = []
    for evaluation, title, ylim in panels:
        series = {}
        for family, display in families:
            subset = (
                frame[
                    frame["family"].eq(family)
                    & frame["evaluation_condition"].eq(evaluation)
                ]
                .set_index("training_condition")
                .loc[conditions]
            )
            series[display] = {"mean": subset["f1_mean"].tolist(), "sd": subset["f1_std"].tolist()}
        out.append({"evaluation": evaluation, "title": title, "ylim": ylim, "series": series})
    return {"conditions": conditions, "labels": labels, "families": families, "panels": out}


def stress_data() -> dict:
    """As make_figures.stress_figure selects it."""
    leakage = pd.read_csv(RESULTS / "rigor_leakage_summary.csv")
    cue = pd.read_csv(RESULTS / "rigor_cue_mask_summary.csv")

    families = [("Word LR", "Word TF--IDF"), ("MiniLM LR", "MiniLM")]
    split_specs = [("source_group", "Source-group", "#8CBFD9"), ("row_stratified", "Row-wise", "#E69F00")]
    width = 0.34
    panel_a = []
    for offset, (strategy, display, color) in zip((-width / 2, width / 2), split_specs):
        means = []
        stds = []
        for family, _ in families:
            row = leakage[
                leakage["family"].eq(family)
                & leakage["split_strategy"].eq(strategy)
            ].iloc[0]
            means.append(row["f1_mean"])
            stds.append(row["f1_std"])
        panel_a.append({"strategy": strategy, "display": display, "color": color,
                        "x": list(np.arange(len(families)) + offset), "mean": means, "sd": stds,
                        "label_y": [m + s + 0.0016 for m, s in zip(means, stds)]})
    # The inset of the PNG prints 0.0% and 99.7%; here they come from the file.
    overlap = {}
    for strategy in ("source_group", "row_stratified"):
        rates = leakage.loc[leakage["split_strategy"].eq(strategy), "test_group_overlap_rate_mean"]
        if rates.max() - rates.min() > 1e-12:
            raise ValueError(f"test-group overlap differs between families for {strategy}")
        overlap[strategy] = f"{100 * rates.iloc[0]:.1f}%"
    if (overlap["source_group"], overlap["row_stratified"]) != ("0.0%", "99.7%"):
        raise ValueError(f"overlap {overlap} differs from the 0.0% / 99.7% printed in the PNG")

    evaluations = [
        ("linked_rewrites", "Linked"),
        ("external_openai", "Gen.-disjoint"),
    ]
    input_specs = [
        ("original", "Original", "#009E73"),
        ("top20_positive_unigrams_masked", "Top-20 masked", "#D55E00"),
    ]
    panel_b = []
    for offset, (input_condition, display, color) in zip((-width / 2, width / 2), input_specs):
        means = []
        stds = []
        for evaluation, _ in evaluations:
            row = cue[
                cue["evaluation_condition"].eq(evaluation)
                & cue["input_condition"].eq(input_condition)
            ].iloc[0]
            means.append(row["f1_mean"])
            stds.append(row["f1_std"])
        panel_b.append({"input": input_condition, "display": display, "color": color,
                        "x": list(np.arange(len(evaluations)) + offset), "mean": means, "sd": stds,
                        "label_y": [m + s + 0.004 for m, s in zip(means, stds)]})
    return {"families": families, "width": width, "panel_a": panel_a, "overlap": overlap,
            "evaluations": evaluations, "panel_b": panel_b}


# ----------------------------------------------------------------------------
# shared pgfplots settings (written into every picture)
# ----------------------------------------------------------------------------
COMMON_AXIS = r"""
      scale only axis,
      axis lines*=left, tick align=outside, tick pos=left,
      major tick length=3pt, minor tick length=1.8pt,
      every tick/.style={black, line width=0.5pt},
      axis line style={line width=0.6pt},
      tick label style={font=\scriptsize},
      label style={font=\footnotesize},
      title style={font=\footnotesize\bfseries, at={(0,1)}, anchor=south west,
        inner xsep=0pt},
      legend style={font=\scriptsize, draw=none, fill=none},
      legend cell align=left,"""

# Error bars in the matplotlib style: solid lines, caps of half-width ``cap_half``.
# ``error mark options`` replaces pgfplots' default {rotate=90}, which turns the
# default "-" mark across the bar, so rotate=90 is repeated. ``color`` None = the
# plot's color (matplotlib errorbar); bar plots pass black (ax.bar yerr).
def error_bars(line_width: float, cap_half: float, cap_width: float, color: str | None = None) -> str:
    ink = f"{color}, " if color else ""
    return (f"error bars/y dir=both, error bars/y explicit,\n"
            f"        error bars/error bar style={{{ink}solid, line width={line_width}pt}},\n"
            f"        error bars/error mark options={{{ink}solid, line width={cap_width}pt, "
            f"mark size={cap_half}pt, rotate=90}}")


# legend handle of a matplotlib errorbar: line, error bar with caps, marker on top
def errorbar_legend(length: float, half_height: float, cap_half: float, bar_width: float,
                    cap_width: float) -> str:
    mid = length / 2
    return (r"legend image code/.code={%" "\n"
            f"        \\draw[#1, mark=none] (0pt,0pt) -- ({length}pt,0pt);\n"
            f"        \\draw[#1, mark=none, solid, line width={bar_width}pt] ({mid}pt,-{half_height}pt) -- "
            f"({mid}pt,{half_height}pt);\n"
            f"        \\draw[#1, mark=none, solid, line width={cap_width}pt]\n"
            f"          ({mid - cap_half}pt,-{half_height}pt) -- ({mid + cap_half}pt,-{half_height}pt)\n"
            f"          ({mid - cap_half}pt,{half_height}pt) -- ({mid + cap_half}pt,{half_height}pt);\n"
            f"        \\draw[#1] plot coordinates {{({mid}pt,0pt)}};}}")


# ----------------------------------------------------------------------------
# figure 1: domain_adaptation
# ----------------------------------------------------------------------------
DOMAIN_TEMPLATE = r"""
<<HEADER>>\begin{tikzpicture}[black]
  \definecolor{daGrid}{HTML}{<<GRID>>}% #<<GRIDRAW>> at opacity 0.6, as in the PNG
  \definecolor{daRef}{HTML}{888888}% reference lines
  \definecolor{daRefText}{HTML}{555555}% their labels
  % (a) F1 at threshold 0.5 by training source -> evaluation target: mean and SD
  % over five seeds; x = group index -/+ half the bar width (bar center)
  \pgfplotstableread[row sep=\\]{
<<TABLE_WORD>>
  }\daWord
  \pgfplotstableread[row sep=\\]{
<<TABLE_MINILM>>
  }\daMiniLM
  % (b) Word TF-IDF F1 at threshold 0.5 on human-authored mail against k
  \pgfplotstableread[row sep=\\]{
<<TABLE_CURVE>>
  }\daCurve
  % axis boxes: 0.94\linewidth minus <<OVERHEAD>>pt, the width of the tick labels, axis
  % labels and panel gap in the paper's fonts (picture = 0.94\linewidth)
  \pgfmathsetlengthmacro\daAxisWidth{(0.94\linewidth-<<OVERHEAD>>pt)/2}%
  \begin{groupplot}[
      group style={group size=2 by 1, horizontal sep=<<HSEP>>pt},
      width=\daAxisWidth, height=<<HEIGHT>>pt,<<COMMON>>
      grid style={draw=daGrid, line width=0.5pt},
      ylabel={F1 at threshold 0.5},
      yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1},
    ]
    \nextgroupplot[
      title={(a) Training source {\boldmath$\rightarrow$} evaluation target},
      xmin=<<A_XMIN>>, xmax=<<A_XMAX>>, ymin=0, ymax=1.2,
      xtick={<<A_XTICK>>},
      xticklabels={
        <<A_XTICKLABELS>>},
      xticklabel style={align=center},
      ytick={0,0.2,0.4,0.6,0.8,1.0,1.2}, ymajorgrids,
      legend columns=2,
      legend style={at={(0,1)}, anchor=north west, xshift=2pt, yshift=-2pt,
        /tikz/every even column/.append style={column sep=8pt}},
      legend image code/.code={\fill[#1] (0pt,-2.45pt) rectangle (9.8pt,2.45pt);},
    ]
    \addplot[ybar, bar width=<<BAR_WIDTH>>, fill=<<WORD_COLOR>>, draw=white, line width=0.5pt,
        <<ERR_A>>]
      table[x=x, y=f1, y error=sd] {\daWord};
    \addlegendentry{<<WORD_LABEL>>}
    \addplot[ybar, bar width=<<BAR_WIDTH>>, fill=<<MINILM_COLOR>>, draw=white, line width=0.5pt,
        <<ERR_A>>]
      table[x=x, y=f1, y error=sd] {\daMiniLM};
    \addlegendentry{<<MINILM_LABEL>>}

    \nextgroupplot[
      title={(b) Word TF--IDF: target-data exchange rate},
      xmode=log, xmin=<<B_XMIN>>, xmax=<<B_XMAX>>, ymin=0, ymax=1.05,
      ytick={0,0.2,0.4,0.6,0.8,1.0},
      grid=major,
      xlabel={human-authored training messages ($k$)},
      legend style={at={(1,0)}, anchor=south east, xshift=-2pt, yshift=2pt,
        fill=white, draw=none, inner sep=2pt},
    ]
    % reference lines: full-corpus ceiling (human -> human, Word TF-IDF) and
    % synthetic only (k = 0, as collected)
    \addplot[daRef, line width=0.8pt, dash pattern=on 2.96pt off 1.28pt, forget plot]
      coordinates {(<<B_XMIN>>,<<CEILING>>) (<<B_XMAX>>,<<CEILING>>)};
    \addplot[daRef, line width=0.8pt, dash pattern=on 5.12pt off 1.28pt on 0.8pt off 1.28pt,
        forget plot]
      coordinates {(<<B_XMIN>>,<<FLOOR>>) (<<B_XMAX>>,<<FLOOR>>)};
    \node[anchor=base west, inner sep=0pt, font=\scriptsize, text=daRefText]
      at (axis cs:26,<<CEILING_LABEL_Y>>) {full-corpus ceiling};
    \node[anchor=north west, inner sep=0pt, font=\scriptsize, text=daRefText]
      at (axis cs:26,<<FLOOR_LABEL_Y>>) {synthetic only};
    \addplot[<<WORD_COLOR>>, line width=1.2pt, mark=*, mark size=1.6pt,
        mark options={solid, line width=0.8pt}]
      table[x=k, y=<<ARM1>>] {\daCurve};
    \addlegendentry{<<ARM1_LABEL>>}
    \addplot[<<WORD_COLOR>>, line width=1.2pt, dash pattern=on 4.44pt off 1.92pt,
        mark=square*, mark size=1.6pt, mark options={solid, fill=white, line width=0.8pt}]
      table[x=k, y=<<ARM2>>] {\daCurve};
    \addlegendentry{<<ARM2_LABEL>>}
    \addplot[<<WORD_COLOR>>, line width=1.2pt, dash pattern=on 1.2pt off 1.98pt,
        mark=triangle*, mark size=2pt, mark options={solid, fill=white, line width=0.8pt}]
      table[x=k, y=<<ARM3>>] {\daCurve};
    \addlegendentry{<<ARM3_LABEL>>}
  \end{groupplot}
\end{tikzpicture}
"""


def domain_adaptation_tex() -> str:
    data = domain_adaptation_data()
    bars, arms = data["bars"], data["arms"]
    width = data["width"]
    groups = np.arange(len(data["conditions"]))
    a_lim = mpl_margins(float(groups.min() - width), float(groups.max() + width))
    ks = [arm["k"] for arm in arms.values()]
    if any(k != ks[0] for k in ks):
        raise ValueError("the arms of panel (b) were measured at different k")
    b_lim = mpl_margins(min(ks[0]), max(ks[0]), log=True)
    column = {"human_only_k": "target", "mixed_balanced": "upweighted", "mixed": "collected"}
    if list(arms) != list(column):
        raise ValueError(f"unexpected arms {list(arms)}")

    def bar_table(model):
        return table(["x", "f1", "sd"], [[num(x), num(f), num(s)] for x, f, s in
                                         zip(bars[model]["x"], bars[model]["f1"], bars[model]["sd"])])

    curve_rows = [[str(k)] + [num(arms[arm]["f1"][i]) for arm in column] for i, k in enumerate(ks[0])]
    values = {
        "HEADER": header("domain_adaptation", "fig:adaptation", "paper.tex",
                         "make_limitation_figures.py, domain_adaptation_figure",
                         ["results/revision_domain_adaptation_summary.csv",
                          "results/revision_fewshot_curve.csv"]),
        "GRID": blend_on_white(GRID, 0.6), "GRIDRAW": html(GRID),
        "TABLE_WORD": bar_table("word_lr"), "TABLE_MINILM": bar_table("minilm_lr"),
        "TABLE_CURVE": table(["k"] + list(column.values()), curve_rows),
        "OVERHEAD": "75.78", "HSEP": "42", "HEIGHT": "128",
        "COMMON": COMMON_AXIS,
        "A_XMIN": num(a_lim[0]), "A_XMAX": num(a_lim[1]),
        "A_XTICK": ",".join(str(g) for g in groups),
        "A_XTICKLABELS": tick_list([label for _, _, label in data["conditions"]]),
        "BAR_WIDTH": f"{width}",
        "WORD_COLOR": oi(MODEL_COLOR["word_lr"]), "MINILM_COLOR": oi(MODEL_COLOR["minilm_lr"]),
        # "TF--IDF": the paper's en dash (matplotlib prints a hyphen)
        "WORD_LABEL": MODEL_LABEL["word_lr"].replace("TF-IDF", "TF--IDF"),
        "MINILM_LABEL": MODEL_LABEL["minilm_lr"],
        "ERR_A": error_bars(0.7, 2.0, 1.0, "black"),
        "B_XMIN": f"{b_lim[0]:.3f}", "B_XMAX": f"{b_lim[1]:.3f}",
        "CEILING": num(data["ceiling"]), "FLOOR": num(data["floor"]),
        "CEILING_LABEL_Y": num(data["ceiling"] - 0.06), "FLOOR_LABEL_Y": num(data["floor"] - 0.025),
    }
    for index, arm in enumerate(column, start=1):
        values[f"ARM{index}"] = column[arm]
        values[f"ARM{index}_LABEL"] = arms[arm]["label"]
    return fill(DOMAIN_TEMPLATE, values)


# ----------------------------------------------------------------------------
# figure 2: adaptive_evasion
# ----------------------------------------------------------------------------
ADAPTIVE_TEMPLATE = r"""
<<HEADER>>\begin{tikzpicture}[black]
  \definecolor{aeGrid}{HTML}{<<GRID>>}% #<<GRIDRAW>> at opacity 0.6, as in the PNG
  \definecolor{aeBox}{HTML}{1B1B1B}% boxed cells
  \definecolor{aeNote}{HTML}{3D4A52}% note under panel (b)
  % (a) evasion rate (share of initially flagged phishing driven below 0.5) by
  % edit budget: mean and SD over the partitions
  \pgfplotstableread[row sep=\\]{
<<TABLE_CURVE>>
  }\aeCurve
  % axis boxes: 0.94\linewidth minus <<OVERHEAD>>pt, the width of the tick labels, axis
  % labels and panel gap and colorbar in the paper's fonts (picture = 0.94\linewidth)
  \pgfmathsetlengthmacro\aeInner{0.94\linewidth-<<OVERHEAD>>pt}%
  \pgfmathsetlengthmacro\aeWidthA{0.47*\aeInner}% width ratio 0.94 : 1.06, as in the PNG
  \pgfmathsetlengthmacro\aeWidthB{0.53*\aeInner}%
  \begin{groupplot}[
      group style={group size=2 by 1, horizontal sep=<<HSEP>>pt},
      height=<<HEIGHT>>pt,<<COMMON>>
      grid style={draw=aeGrid, line width=0.5pt},
    ]
    \nextgroupplot[
      width=\aeWidthA,
      title={(a) Evasion rate vs.\ edit budget},
      xmin=<<A_XMIN>>, xmax=<<A_XMAX>>, ymin=0, ymax=1.05,
      xtick={<<BUDGETS>>}, ytick={0,0.2,0.4,0.6,0.8,1.0},
      yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1},
      grid=major,
      xlabel={share of content words changed (\%)},
      ylabel={evasion rate},
      legend style={at={(0,1)}, anchor=north west, xshift=1pt, yshift=-10.5pt,
        row sep=-0.6pt},
      <<LEGEND_IMAGE>>,
    ]
    \node[anchor=north west, inner sep=0pt, font=\scriptsize]
      at ([xshift=4.5pt, yshift=-3pt]rel axis cs:0,1) {<<LEGEND_TITLE>>};
<<CURVES>>
    \nextgroupplot[
      width=\aeWidthB,
      title={(b) Recall on adversarial messages},
      xmin=-0.5, xmax=<<B_XMAX>>, ymin=-0.5, ymax=<<B_YMAX>>, y dir=reverse,
      enlargelimits=false, axis on top, clip=false,
      xtick={<<B_XTICK>>}, xticklabels={
        <<B_XTICKLABELS>>},
      ytick={<<B_YTICK>>}, yticklabels={
        <<B_YTICKLABELS>>},
      xticklabel style={align=center}, yticklabel style={align=right},
      major tick length=0pt,
      xlabel={scored by}, ylabel={attacked},
      point meta min=0, point meta max=1,
      % viridis_r sampled at 33 points (only the colorbar uses it)
      colormap={aeViridisR}{
        <<COLORMAP>>},
      colorbar,
      colorbar style={at={(parent axis.north east)}, anchor=north west, xshift=6pt, width=7pt,
        ytick={0,0.2,0.4,0.6,0.8,1.0},
        yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1},
        major tick length=2.5pt, tick align=outside,
        axis line style={line width=0.4pt},
        axis lines*=box, tick pos=right, yticklabel pos=right, ylabel near ticks,
        ylabel={recall at threshold 0.5}, ylabel style={font=\footnotesize}},
    ]
    % cells: recall at threshold 0.5 of the column detector on the messages
    % crafted against the row detector, mean over the partitions; colors are
    % matplotlib's viridis_r at these values (vmin 0, vmax 1)
<<CELLS>>
    % boxed: scored by the detector the messages were crafted against
<<BOXES>>
<<LABELS>>
  \end{groupplot}
  \node[anchor=north, inner sep=0pt, font=\scriptsize, text=aeNote]
    at ([yshift=-2pt]group c2r1.south |- group c2r1.outer south)
    {boxed: scored by the detector the messages were crafted against};
\end{tikzpicture}
"""

CURVE_STYLE = {  # matplotlib: ls, marker; open marker for the clean model
    ":": "dash pattern=on 1.2pt off 1.98pt",
    "-": "solid",
    "--": "dash pattern=on 4.44pt off 1.92pt",
}
MARK = {"o": ("*", 1.6), "s": ("square*", 1.6), "^": ("triangle*", 2.0)}


def adaptive_tex() -> str:
    data = adaptive_data()
    curves, grid = data["curves"], data["grid"]
    budgets = data["budgets"]
    key = {"word_lr_clean_only": "wordclean", "word_lr_linked_augmented": "wordaug",
           "charplusword_lr_linked_augmented": "charwordaug"}
    columns = ["budget"]
    for model in curves:
        columns += [key[model], key[model] + "_sd"]
    rows = []
    for index, budget in enumerate(budgets):
        row = [str(budget)]
        for model in curves:
            row += [num(curves[model]["mean"][index]), num(curves[model]["sd"][index])]
        rows.append(row)

    label = {  # verbatim from adaptive_figure: name, ls, marker
        "word_lr_clean_only": ("Word, clean", ":", "o"),
        "word_lr_linked_augmented": ("Word, augmented", "-", "s"),
        "charplusword_lr_linked_augmented": ("Char+word, augmented", "--", "^"),
    }
    plots = []
    for model, curve in curves.items():
        name, style, marker = label[model]
        mark, size = MARK[marker]
        face = ", fill=white" if "clean" in model else ""
        plots.append(
            f"    \\addplot[{oi(curve['color'])}, line width=1.2pt, {CURVE_STYLE[style]},\n"
            f"        mark={mark}, mark size={size}pt, mark options={{solid{face}, line width=0.8pt}},\n"
            f"        {error_bars(0.7, 2.0, 0.8)}]\n"
            f"      table[x=budget, y={key[model]}, y error={key[model]}_sd] {{\\aeCurve}};\n"
            f"    \\addlegendentry{{{name}}}")

    order, attacked = data["order"], data["attacked"]
    cmap = colormaps["viridis_r"]
    cells, boxes, labels = [], [], []
    for i, source in enumerate(attacked):
        for j, _ in enumerate(order):
            value = grid[i, j]
            if np.isnan(value):
                continue
            r, g, b, _ = (np.array(cmap(float(value))) * 255).round().astype(int)
            cells.append(f"    \\filldraw[color={{rgb,255:red,{r};green,{g};blue,{b}}}, line width=0.25pt]"
                         f" % {value:.6f}\n"
                         f"      (axis cs:{j - 0.5},{i - 0.5}) rectangle (axis cs:{j + 0.5},{i + 0.5});")
            text = "white" if value > 0.55 else "black"
            labels.append(f"    \\node[font=\\scriptsize, text={text}] at (axis cs:{j},{i}) {{{value:.2f}}};")
        j = order.index(source)
        boxes.append(f"    \\draw[aeBox, line width=1.3pt] (axis cs:{j - 0.5},{i - 0.5}) rectangle "
                     f"(axis cs:{j + 0.5},{i + 0.5});")
    samples = np.linspace(0, 1, 33)
    entries = ["rgb255=({},{},{})".format(*(np.array(cmap(float(s))[:3]) * 255).round().astype(int))
               for s in samples]
    colormap = "\n        ".join(" ".join(entries[i:i + 5]) for i in range(0, len(entries), 5))

    a_lim = mpl_margins(min(budgets), max(budgets))
    values = {
        "HEADER": header("adaptive_evasion", "fig:adaptive", "paper.tex",
                         "make_limitation_figures.py, adaptive_figure",
                         ["results/revision_adaptive_by_seed.csv",
                          "results/revision_adaptive_transfer.csv"]),
        "GRID": blend_on_white(GRID, 0.6), "GRIDRAW": html(GRID),
        "TABLE_CURVE": table(columns, rows),
        "OVERHEAD": "149.49", "HSEP": "64", "HEIGHT": "132",
        "COMMON": COMMON_AXIS,
        "A_XMIN": num(a_lim[0]), "A_XMAX": num(a_lim[1]),
        "BUDGETS": ",".join(str(b) for b in budgets),
        "LEGEND_IMAGE": errorbar_legend(13.0, 3.0, 2.0, 0.7, 0.8),
        "LEGEND_TITLE": (f"mean $\\pm$ SD over {data['n_partitions']} partitions, "
                         f"{data['n_targets']} messages each"),
        "CURVES": "\n".join(plots),
        "B_XMAX": f"{len(order) - 0.5}", "B_YMAX": f"{len(attacked) - 0.5}",
        "B_XTICK": ",".join(str(j) for j in range(len(order))),
        "B_XTICKLABELS": tick_list([data["short"][m] for m in order]),
        "B_YTICK": ",".join(str(i) for i in range(len(attacked))),
        "B_YTICKLABELS": tick_list([data["short"][m] for m in attacked]),
        "COLORMAP": colormap,
        "CELLS": "\n".join(cells), "BOXES": "\n".join(boxes), "LABELS": "\n".join(labels),
    }
    return fill(ADAPTIVE_TEMPLATE, values)


# ----------------------------------------------------------------------------
# figure 3: augmentation_dose
# ----------------------------------------------------------------------------
DOSE_TEMPLATE = r"""
<<HEADER>>\begin{tikzpicture}[black]
  \definecolor{adGrid}{HTML}{<<GRID>>}%
  % F1 by training exposure: mean and SD over the five reference partitions;
  % x = 0..4: <<CONDITIONS>>
<<TABLES>>
  % axis boxes: 0.94\linewidth minus <<OVERHEAD>>pt, the width of the tick labels, axis
  % labels and panel gap in the paper's fonts (picture = 0.94\linewidth)
  \pgfmathsetlengthmacro\adAxisWidth{(0.94\linewidth-<<OVERHEAD>>pt)/2}%
  \begin{groupplot}[
      group style={group size=2 by 1, horizontal sep=<<HSEP>>pt},
      width=\adAxisWidth, height=<<HEIGHT>>pt,<<COMMON>>
      grid style={draw=adGrid, line width=0.6pt, dash pattern=on 0.6pt off 0.99pt},
      xmin=<<XMIN>>, xmax=<<XMAX>>, xtick={<<XTICK>>},
      xticklabels={
        <<XTICKLABELS>>},
      xticklabel style={align=center}, xtick style={draw=none},
      ymajorgrids, ylabel={F1},
      <<LEGEND_IMAGE>>,
    ]
<<PANELS>>
  \end{groupplot}
  \node[anchor=north, inner sep=0pt, font=\footnotesize]
    at ([yshift=-3pt]$(group c1r1.outer south west)!0.5!(group c2r1.outer south east)$)
    {Training exposure (variants per source group)};
\end{tikzpicture}
"""


def dose_tex() -> str:
    data = dose_data()
    colors = {display: oi(COLORS[display]) for _, display in data["families"]}
    ticks = [  # matplotlib's automatic ticks inside the limits set by dose_figure
        ([0.93, 0.94, 0.95, 0.96, 0.97, 0.98], 2),
        ([0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950], 3),
    ]
    tables, panels = [], []
    names = ["adLinked", "adExternal"]
    for index, (panel, (tick, digits)) in enumerate(zip(data["panels"], ticks)):
        columns = ["x"]
        for _, display in data["families"]:
            short = "word" if display.startswith("Word") else "minilm"
            columns += [short, short + "_sd"]
        rows = []
        for x in range(len(data["conditions"])):
            row = [str(x)]
            for _, display in data["families"]:
                series = panel["series"][display]
                row += [num(series["mean"][x]), num(series["sd"][x])]
            rows.append(row)
        tables.append(f"  % ({chr(97 + index)}) {panel['evaluation']}\n"
                      f"  \\pgfplotstableread[row sep=\\\\]{{\n{table(columns, rows)}\n  }}\\{names[index]}")
        lo, hi = panel["ylim"]
        if min(tick) < lo or max(tick) > hi:
            raise ValueError("tick outside the y limits")
        plots = []
        for _, display in data["families"]:
            short = "word" if display.startswith("Word") else "minilm"
            plots.append(
                f"    \\addplot[{colors[display]}, line width=1.7pt, mark=*, mark size=2.1pt,\n"
                f"        mark options={{solid, line width=0.8pt}},\n"
                f"        {error_bars(1.7, 2.2, 0.8)}]\n"
                f"      table[x=x, y={short}, y error={short}_sd] {{\\{names[index]}}};"
                + (f"\n    \\addlegendentry{{{display}}}" if index == 0 else ""))
        legend = (",\n      legend style={at={(1,0)}, anchor=south east, xshift=-2pt, yshift=2pt,"
                  " row sep=-0.6pt}" if index == 0 else "")
        panels.append(
            f"    \\nextgroupplot[\n"
            f"      title={{({chr(97 + index)}) {panel['title']}}},\n"
            f"      ymin={lo}, ymax={hi}, ytick={{{','.join(f'{t:.3f}' for t in tick)}}},\n"
            f"      yticklabel style={{/pgf/number format/.cd, fixed, fixed zerofill, precision={digits}}}"
            f"{legend},\n"
            f"    ]\n" + "\n".join(plots))
    xlim = mpl_margins(0, len(data["conditions"]) - 1)
    values = {
        "HEADER": header("augmentation_dose", "fig:dose", "supplement.tex",
                         "make_figures.py, dose_figure", ["results/rigor_dose_summary.csv"]),
        "GRID": html(GRID),
        "CONDITIONS": ",\n  %   ".join(", ".join(data["conditions"][i:i + 3])
                                       for i in range(0, len(data["conditions"]), 3)),
        "TABLES": "\n".join(tables),
        "OVERHEAD": "82.18", "HSEP": "45", "HEIGHT": "128",
        "COMMON": COMMON_AXIS,
        "XMIN": num(xlim[0]), "XMAX": num(xlim[1]),
        "XTICK": ",".join(str(x) for x in range(len(data["conditions"]))),
        "XTICKLABELS": tick_list(data["labels"]),
        "LEGEND_IMAGE": errorbar_legend(14.0, 3.2, 2.2, 1.7, 0.8),
        "PANELS": "\n\n".join(panels),
    }
    return fill(DOSE_TEMPLATE, values)


# ----------------------------------------------------------------------------
# figure 4: leakage_and_masking
# ----------------------------------------------------------------------------
STRESS_TEMPLATE = r"""
<<HEADER>>\begin{tikzpicture}[black]
  \definecolor{lmGrid}{HTML}{<<GRID>>}%
  \definecolor{lmSourceGroup}{HTML}{<<SOURCE_COLOR>>}% source-group bars (light blue)
  \definecolor{lmTitle}{HTML}{5A6B75}% inset: heading
  \definecolor{lmSourceInk}{HTML}{2F7DB0}% inset: source-group share
  \definecolor{lmRowInk}{HTML}{B5760A}% inset: row-wise share
  \definecolor{lmSub}{HTML}{7C8A93}% inset: subtitle
  \definecolor{lmArrow}{HTML}{8A99A2}% inset: arrow
  \definecolor{lmCard}{HTML}{F5F8FA}% inset: fill
  \definecolor{lmCardEdge}{HTML}{D3DCE2}% inset: edge
  % bars: F1 mean and SD over the five reference partitions; x = bar center
<<TABLES>>
  % axis boxes: 0.94\linewidth minus <<OVERHEAD>>pt, the width of the tick labels, axis
  % labels and panel gap in the paper's fonts (picture = 0.94\linewidth)
  \pgfmathsetlengthmacro\lmAxisWidth{(0.94\linewidth-<<OVERHEAD>>pt)/2}%
  \begin{groupplot}[
      group style={group size=2 by 1, horizontal sep=<<HSEP>>pt},
      width=\lmAxisWidth, height=<<HEIGHT>>pt,<<COMMON>>
      grid style={draw=lmGrid, line width=0.6pt, dash pattern=on 0.6pt off 0.99pt},
      xmin=<<XMIN>>, xmax=<<XMAX>>, xtick={0,1}, xtick style={black},
      ymajorgrids, ylabel={F1},
      yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=2},
      legend image code/.code={\fill[#1] (0pt,-2.6pt) rectangle (14pt,2.6pt);},
      /tikz/lm value/.style={anchor=south, inner sep=0pt, font=\scriptsize},
    ]
    \nextgroupplot[
      title={(a) Split leakage},
      ymin=0.94, ymax=0.995, ytick={0.94,0.95,0.96,0.97,0.98,0.99},
      xticklabels={
        <<A_XTICKLABELS>>},
    ]
<<A_PLOTS>>
    \coordinate (lmInset) at (axis description cs:0.758,0.792);% center of the inset

    \nextgroupplot[
      title={(b) Lexical-cue masking},
      ymin=0.84, ymax=1.005, ytick={0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.00},
      xticklabels={
        <<B_XTICKLABELS>>},
      legend style={at={(1,1)}, anchor=north east, xshift=-2pt, yshift=-2pt},
    ]
<<B_PLOTS>>
  \end{groupplot}
  % inset of panel (a): share of test rows whose source group also occurs in
  % training. A one-column matrix (outside the axis, so cell coordinates are
  % canvas points): its card is drawn before the cells, i.e. under the text and
  % over the bars and the grid, as in the PNG.
  \matrix[draw=lmCardEdge, fill=lmCard, line width=0.6pt, rounded corners=3pt,
      inner xsep=5pt, inner ysep=3pt, row sep=1.6pt,
      nodes={inner sep=0pt, anchor=center}] at (lmInset) {
    \node[font=\scriptsize\bfseries, text=lmTitle] {TEST-GROUP OVERLAP}; \\
    \node[font=\large\bfseries, text=lmSourceInk] at (-25pt,0) {<<OVERLAP_SOURCE>>};
    \draw[lmArrow, line width=1.3pt, -{Triangle[length=3.4pt, width=3.8pt]}]
      (-10pt,0) -- (5.5pt,0);
    \node[font=\large\bfseries, text=lmRowInk] at (25pt,0) {<<OVERLAP_ROW>>}; \\
    \node[font=\scriptsize, text=lmSub] {source-group\hspace{0.7em}vs\hspace{0.7em}row-wise}; \\
  };
\end{tikzpicture}
"""


def stress_tex() -> str:
    data = stress_data()
    names = {"source_group": "lmSourceGroupBars", "row_stratified": "lmRowWiseBars",
             "original": "lmOriginalBars", "top20_positive_unigrams_masked": "lmMaskedBars"}
    tables = []
    for spec in data["panel_a"] + data["panel_b"]:
        key = spec.get("strategy") or spec.get("input")
        rows = [[num(x), num(m), num(s)] for x, m, s in zip(spec["x"], spec["mean"], spec["sd"])]
        tables.append(f"  % {key} ({spec['display']})\n"
                      f"  \\pgfplotstableread[row sep=\\\\]{{\n{table(['x', 'f1', 'sd'], rows)}\n"
                      f"  }}\\{names[key]}")

    def plots(specs, legend):
        out = []
        for spec in specs:
            key = spec.get("strategy") or spec.get("input")
            color = "lmSourceGroup" if spec["color"] == "#8CBFD9" else oi(spec["color"])
            out.append(
                f"    \\addplot[ybar, bar width={data['width']}, fill={color}, draw=white, line width=0.7pt,\n"
                f"        {error_bars(1.5, 2.0, 1.0, 'black')}]\n"
                f"      table[x=x, y=f1, y error=sd] {{\\{names[key]}}};"
                + (f"\n    \\addlegendentry{{{spec['display']}}}" if legend else ""))
        for spec in specs:  # value labels at mean + SD + offset, as in the PNG
            for x, y, m in zip(spec["x"], spec["label_y"], spec["mean"]):
                out.append(f"    \\node[lm value] at (axis cs:{num(x)},{num(y)}) {{{m:.3f}}};")
        return "\n".join(out)

    # tick labels: "Word TF--IDF" typesets the paper's en dash (the PNG prints
    # "TF-IDF"); the generator-disjoint corpus is spelled out (PNG: "Gen.-disjoint")
    tick_b = {"Linked": "Linked rewrites", "Gen.-disjoint": "Generator-disjoint"}
    xlim = mpl_margins(-data["width"], 1 + data["width"])
    values = {
        "HEADER": header("leakage_and_masking", "fig:stress", "supplement.tex",
                         "make_figures.py, stress_figure",
                         ["results/rigor_leakage_summary.csv",
                          "results/rigor_cue_mask_summary.csv"],
                         needs="pgfplots (compat 1.18, groupplots), TikZ arrows.meta"),
        "GRID": html(GRID), "SOURCE_COLOR": html(data["panel_a"][0]["color"]),
        "TABLES": "\n".join(tables),
        "OVERHEAD": "82.18", "HSEP": "45", "HEIGHT": "140",
        "COMMON": COMMON_AXIS,
        "XMIN": num(xlim[0]), "XMAX": num(xlim[1]),
        "A_XTICKLABELS": tick_list([display for _, display in data["families"]]),
        "A_PLOTS": plots(data["panel_a"], legend=False),
        "OVERLAP_SOURCE": data["overlap"]["source_group"].replace("%", "\\%"),
        "OVERLAP_ROW": data["overlap"]["row_stratified"].replace("%", "\\%"),
        "B_XTICKLABELS": tick_list([tick_b[display] for _, display in data["evaluations"]]),
        "B_PLOTS": plots(data["panel_b"], legend=True),
    }
    return fill(STRESS_TEMPLATE, values)


# ----------------------------------------------------------------------------
FIGURES = {
    "domain_adaptation": domain_adaptation_tex,
    "adaptive_evasion": adaptive_tex,
    "augmentation_dose": dose_tex,
    "leakage_and_masking": stress_tex,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--copy-to", type=pathlib.Path,
                        default=DEFAULT_BUNDLE if DEFAULT_BUNDLE.is_dir() else None,
                        help="also copy every tikz_NAME.tex here (default: the manuscript "
                             "bundle, %(default)s)")
    args = parser.parse_args()
    TIKZ_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES.items():
        path = TIKZ_DIR / f"tikz_{name}.tex"
        path.write_text(build())
        print(f"wrote {path.relative_to(STUDY)}")
        if args.copy_to is not None:
            shutil.copy(path, args.copy_to / path.name)
            print(f"  copied to {args.copy_to / path.name}")
    if args.copy_to is None:
        print(f"no copy made: {DEFAULT_BUNDLE} does not exist (use --copy-to)")


if __name__ == "__main__":
    main()
