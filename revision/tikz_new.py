#!/usr/bin/env python3
r"""TikZ/pgfplots versions of three round-4 figures (figure group "new").

The manuscript's figures are redrawn in TikZ so that they are typeset in the
document's own fonts. For each figure NAME this script writes one
self-contained ``tikzpicture`` (data inline, width relative to \linewidth, no
external files) to figures/tikz/tikz_NAME.tex and copies it to the manuscript
bundle, where the float reads it with \input{tikz_NAME.tex}:

* ``pipeline_signature``    -- Figure fig:signature, from figure_pipeline_signature()
* ``l3_explained``          -- Figure fig:l3why, from figure_l3_explained()
* ``deployment_robustness`` -- Figure fig:deploy, from figure_deployment_robustness()

All three functions are in make_r4_new_figures.py, which draws the PNG versions.
Every plotted number comes from the same code path: the constants and label
helpers (stream and corpus lists, estimator and procedure labels, ``signed``,
``_topic_label``, ``_value_label``, FLOOR, the colors) are imported from that
script, and the data-preparation lines of each figure function, which are
interleaved with plotting calls there and so cannot be imported, are copied
verbatim into the ``*_data`` functions below. Box-plot statistics come from
matplotlib's own ``cbook.boxplot_stats`` (what ``Axes.boxplot`` draws), and
the heat-map colors from matplotlib's GnBu and truncated Oranges colormaps
with the same PowerNorm. Values are written with six significant digits. The
layout follows the matplotlib figure: one picture unit is one inch of the
matplotlib figure, scaled so that the picture has the PNG's width in the
paper. Nothing is refitted and matplotlib draws nothing.

Inputs (results/):
  revision_r4_pipeline_vocab.csv, revision_r4_pipeline_erasure.csv,
  revision_r4_pipeline_dann.csv                    (rev_s_pipeline_signature.py)
  revision_r4_proximity_recall.csv                 (rev_u3_proximity.py)
  revision_r4_l3_topics.csv                        (rev_t_human_error_analysis.py)
  revision_r4_deploy_prevalence_estimation.csv, revision_r4_deploy_rule_validation.csv,
  revision_r4_deploy_score_sensitivity.csv         (rev_w_deployment_robustness.py)

Outputs:
  figures/tikz/tikz_pipeline_signature.tex, figures/tikz/tikz_l3_explained.tex,
  figures/tikz/tikz_deployment_robustness.tex, each also copied to --copy-to
  (default: the manuscript bundle, manuscriptR4/).

Usage (from artifact/eswa_study/revision/):
    python tikz_new.py                               # all three figures
    python tikz_new.py --only l3_explained           # selected figures
    python tikz_new.py --copy-to DIR                 # copy the .tex files to DIR
"""

from __future__ import annotations

import argparse
import shutil
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cbook import boxplot_stats  # noqa: E402
from matplotlib.colors import ListedColormap, PowerNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
TIKZ = FIGURES / "tikz"


def _bundle() -> Path:
    """The manuscript bundle (holds paper.tex), also when run from its artifact copy."""
    root = STUDY.parent.parent
    for candidate in (root / "manuscriptR4", root):
        if (candidate / "paper.tex").exists():
            return candidate
    return root / "manuscriptR4"


def _figure_script():
    """Import make_r4_new_figures.py for its constants, without the model stack.

    The figure script takes only the paths FIGURES, RESULTS and STUDY from
    rev_common, whose import also loads run_experiment and sentence-transformers
    (torch). A stand-in module with those three paths is used for the import.
    """
    for path in (str(HERE), str(STUDY)):
        if path not in sys.path:
            sys.path.insert(0, path)
    inserted = "rev_common" not in sys.modules
    if inserted:
        stub = types.ModuleType("rev_common")
        stub.FIGURES, stub.RESULTS, stub.STUDY = FIGURES, RESULTS, STUDY
        sys.modules["rev_common"] = stub
    try:
        import make_r4_new_figures as module
    finally:
        if inserted:
            del sys.modules["rev_common"]
    return module


SRC = _figure_script()

# The figure script's hex codes, mapped to the paper's Okabe-Ito names; the
# other colors are defined inside each tikzpicture.
OKABE_ITO = {"#0072B2": "oiBlue", "#E69F00": "oiOrange", "#009E73": "oiGreen",
             "#D55E00": "oiVermillion", "#56B4E9": "oiSky", "#CC79A7": "oiPurple"}
LOCAL_COLORS = {"nwInk": SRC.INK, "nwMuted": SRC.MUTED, "nwGrid": SRC.GRID,
                "nwZero": SRC.ZERO, "nwNazarioEarly": SRC.NAZARIO_EARLY,
                "nwBand": "#E8EDF0", "nwFloor": "#F3F3F3", "nwBaseline": "#EEF2F4"}


def color(hex_code: str) -> str:
    """TeX color name for one of the script's hex codes."""
    code = hex_code.upper()
    if code in OKABE_ITO:
        return OKABE_ITO[code]
    for name, value in LOCAL_COLORS.items():
        if value.upper() == code:
            return name
    raise KeyError(hex_code)


# ==========================================================================
# helpers for writing TeX
# ==========================================================================
def num(value: float) -> str:
    """Six significant digits (what every plotted value is written with)."""
    text = f"{float(value):.6g}"
    return "0" if text == "-0" else text


def tex(label: str) -> str:
    """A matplotlib label (mathtext, Unicode, newlines) as LaTeX."""
    for old, new in (("%", r"\%"), ("–", "--"), ("−", r"\(-\)"), ("≤", r"\(\le\)"),
                     ("×", r"\(\times\)"), ("Δ", r"\(\Delta\)"), ("π", r"\(\pi\)"),
                     ("\n", r"\\")):
        label = label.replace(old, new)
    return label


def table(macro: str, header: list[str], rows: list[list], note: str = "") -> str:
    """An inline \\pgfplotstableread block (columns aligned for reading)."""
    cells = [header] + [[cell if isinstance(cell, str) else num(cell) for cell in row]
                        for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    lines = [f"% {line}" for line in note.split("\n")] if note else []
    lines.append("\\pgfplotstableread{")
    lines += ["  " + " ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
              for row in cells]
    lines.append("}" + macro)
    return "\n".join(lines)


def place(name: str, x: float, y: float) -> str:
    """A named coordinate in picture units for an axis's ``at`` key.

    pgfplots reads ``at={(x,y)}`` in its own default units (cm), not in the
    picture's x/y unit, so each axis is anchored to a coordinate defined outside it.
    """
    return f"\\coordinate ({name}) at ({num(x)},{num(y)});"


def coordinates(points) -> str:
    return " ".join(f"({num(x)},{num(y)})" for x, y in points)


def lw(fraction: float) -> str:
    """A length as a fraction of \\linewidth."""
    return f"{fraction:.6f}\\linewidth"


def braced_list(items, indent: str = "    ") -> str:
    """{a},{b},... one item per line (for long tick-label lists)."""
    return ",\n".join(f"{indent}{{{item}}}" for item in items)


# matplotlib marker -> pgfplots mark at markersize 4: 'o' radius 2pt, 's' half side 2pt,
# 'D' the same square turned by 45 degrees, '^' a triangle about 4pt wide.
MARKERS = {"o": ("*", 2.0, ""), "s": ("square*", 2.0, ""),
           "D": ("square*", 2.0, "rotate=45"), "^": ("triangle*", 2.4, "")}


def mark(marker: str, *options: str, scale: float = 1.0) -> str:
    """pgfplots mark keys for a matplotlib marker (mark options set once: they replace)."""
    shape, size, rotate = MARKERS[marker]
    extra = ", ".join(o for o in (rotate, *options) if o)
    return f"mark={shape}, mark size={num(size * scale)}pt, mark options={{{extra}}}"


def preamble(name: str, sources: str, width: str, unit: float, colors: list[str]) -> list[str]:
    """Header comment, picture options, local colors and the shared styles."""
    lines = [
        f"% tikz_{name}.tex -- generated by artifact/eswa_study/revision/tikz_new.py",
        f"% from {sources};",
        "% regenerate it rather than edit it. One tikzpicture, data inline, total width",
        f"% {width}. Picture unit: 1 = {unit:.6f}\\linewidth = one inch of the matplotlib figure.",
        f"\\begin{{tikzpicture}}[x={lw(unit)}, y={lw(unit)}, font=\\scriptsize]",
    ]
    lines += [f"\\definecolor{{{c}}}{{HTML}}{{{LOCAL_COLORS[c].lstrip('#').upper()}}}"
              for c in colors]
    lines += [
        "\\def\\nwSmall{\\fontsize{6}{7}\\selectfont}% the smallest annotations",
        "\\pgfplotsset{",
        "  nw/.style={scale only axis, clip=false, axis line style={line width=0.6pt},",
        "    tick style={line width=0.6pt, black}, tick align=outside, major tick length=2.5pt,",
        "    minor tick length=0pt, tick label style={font=\\scriptsize},",
        "    label style={font=\\footnotesize}, legend cell align=left,",
        "    legend style={font=\\scriptsize, draw=none, fill=none, inner sep=1pt},",
        "    grid style={line width=0.5pt, draw=nwGrid, draw opacity=0.6}},",
        "}",
        "\\tikzset{",
        "  nw grid line/.style={line width=0.5pt, draw=nwGrid, draw opacity=0.6},",
        "  nw title/.style={anchor=base west, inner sep=0pt, font=\\footnotesize\\bfseries},",
        "}",
    ]
    return lines


def write(name: str, text: str, output_dir: Path, copy_to: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"tikz_{name}.tex"
    path.write_text(text)
    print(f"  wrote {path}")
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy_to / path.name)
        print(f"  copied to {copy_to / path.name}")


# ==========================================================================
# 1. pipeline signature (Figure fig:signature)
# ==========================================================================
def pipeline_signature_data() -> dict:
    """The numbers figure_pipeline_signature() plots (its data lines, verbatim)."""
    vocab = pd.read_csv(RESULTS / "revision_r4_pipeline_vocab.csv")
    erasure = pd.read_csv(RESULTS / "revision_r4_pipeline_erasure.csv")
    dann = pd.read_csv(RESULTS / "revision_r4_pipeline_dann.csv")

    cv_column = "origin_auc_cv_on_erased_heldout"
    if cv_column in erasure.columns and erasure[cv_column].notna().all():
        erasure_auc_column = cv_column
    else:
        erasure_auc_column = "origin_auc_after_erasure"
        print(f"  WARNING: {cv_column} missing; falling back to origin_auc_after_erasure")

    # ---- (a) vocabulary neutralization
    ks = np.sort(vocab["removed_ngrams"].unique())
    by_k = vocab.groupby("removed_ngrams")
    n_partitions = int(by_k.size().min())

    def quantile(column, level):
        return by_k[column].quantile(level).reindex(ks).to_numpy()

    curves = {name: [quantile(column, level) for level in (0.05, 0.5, 0.95)]
              for name, column in (("origin", "origin_auc_after_removal"),
                                   ("clean", "clean_f1"), ("augmented", "augmented_f1"))}
    ticks = [0, 2500, 10000, 20000, 30000, 40000]
    delta = by_k["delta_f1"].mean().reindex(ks)
    vocabulary = float(vocab["union_vocabulary"].mean())
    shares = [100 * k / vocabulary for k in ticks]
    share_labels = ["0%"] + [f"{s:.1f}%" if s < 10 else f"{s:.0f}%" for s in shares[1:]]
    delta_labels = [SRC.signed(delta[k]).replace("0.", ".", 1) for k in ticks]

    # ---- (b) linear erasure and domain-adversarial training (MiniLM)
    pivot = dann.pivot_table(index=["seed", "adversarial"], columns="training",
                             values=["f1", "domain_probe_auc", "domain_head_auc"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot["delta"] = pivot["f1_augmented"] - pivot["f1_clean"]
    pivot["probe"] = (pivot["domain_probe_auc_augmented"] + pivot["domain_probe_auc_clean"]) / 2
    pivot["head"] = (pivot["domain_head_auc_augmented"] + pivot["domain_head_auc_clean"]) / 2

    entries = []
    for method, label in (("none", "No erasure"), ("leace", "LEACE"),
                          ("inlp_converged", "INLP to\nconvergence")):
        rows = erasure[erasure["method"] == method]
        entries.append((label, rows["delta_f1"].to_numpy(),
                        float(rows[erasure_auc_column].mean()), None,
                        float(rows["directions_removed"].mean())))
    for adversarial, label in ((False, "Without\nadversary"), (True, "DANN")):
        rows = pivot[pivot["adversarial"] == adversarial]
        head = float(rows["head"].mean()) if adversarial else None
        entries.append((label, rows["delta"].to_numpy(), float(rows["probe"].mean()), head,
                        None))
    positions = np.array([0.0, 1.0, 2.0, 3.45, 4.45])
    boxes = []
    for position, (label, values, auc, head, directions) in zip(positions, entries):
        order = np.argsort(np.argsort(values))
        jitter = np.linspace(-0.17, 0.17, len(values))[order]
        boxes.append({"label": label, "position": float(position), "values": values,
                      "x": position + jitter, "stats": boxplot_stats(values, whis=(5, 95))[0],
                      "mean": float(values.mean()), "auc": auc, "head": head})
    return {"ks": ks, "curves": curves, "ticks": ticks, "share_labels": share_labels,
            "delta_labels": delta_labels, "vocabulary": vocabulary,
            "n_partitions": n_partitions, "boxes": boxes, "auc_column": erasure_auc_column}


def pipeline_signature_tex(d: dict) -> str:
    unit = 0.96 / 7.1
    a_left, b_left, bottom, height = 0.58, 4.30, 0.80, 1.62
    a_width, b_width = 2.50, 2.72
    lines = preamble("pipeline_signature", "results/revision_r4_pipeline_{vocab,erasure,dann}.csv",
                     "0.96\\linewidth", unit, ["nwInk", "nwMuted", "nwGrid", "nwZero"])
    lines += [
        "\\pgfplotsset{",
        "  nw sqrt k/.style={% square-root scale for k (in thousands), as in the PNG",
        "    x coord trafo/.code={\\pgfmathparse{sign(##1)*sqrt(abs(##1))}},",
        "    x coord inv trafo/.code={\\pgfmathparse{(##1)*abs(##1)}},",
        "    xmin=-1.5, xmax=42.5, xtick={0,2.5,10,20,30,40}},",
        f"  nw a/.style={{nw, nw sqrt k, at={{(nwA)}}, width={lw(a_width * unit)},",
        f"    height={lw(height * unit)}}},",
        "  nw curve/.style={line width=1.4pt, mark size=1.7pt},",
        "  nw dotted/.style={dash pattern=on 1.4pt off 2.3pt},",
        "  nw dashed/.style={dash pattern=on 4pt off 1.8pt},",
        "}",
        f"\\path (0,{bottom}) (7.1,{bottom});% fixes the width at 0.96\\linewidth",
        place("nwA", a_left, bottom),
        place("nwB", b_left, bottom),
    ]
    rows = []
    for i, k in enumerate(d["ks"]):
        row = [k / 1000]
        for name in ("augmented", "clean", "origin"):
            row += [d["curves"][name][level][i] for level in (0, 1, 2)]
        rows.append(row)
    lines.append(table("\\nwVocab", ["k", "aug_lo", "aug_med", "aug_hi", "clean_lo", "clean_med",
                                     "clean_hi", "origin_lo", "origin_med", "origin_hi"], rows,
                       f"(a) k = n-grams removed (thousands); 5th/50th/95th percentile over "
                       f"{d['n_partitions']} partitions\nof L2 F1 (augmented, clean-only) and of "
                       f"the held-out origin-probe ROC-AUC"))
    tick_labels = ",".join(["0"] + [f"{num(k / 1000)}k" for k in d["ticks"][1:]])
    lines += [
        "% ---- (a) vocabulary neutralization: origin probe on the right axis (drawn first, below)",
        "\\begin{axis}[nw a, axis x line=none, axis y line*=right, ymin=0.70, ymax=1.012,",
        "  ytick={0.7,0.75,0.8,0.85,0.9,0.95,1}, yticklabels={0.70,0.75,0.80,0.85,0.90,0.95,1.00},",
        "  y axis line style={draw=nwMuted}, ytick style={draw=nwMuted},",
        "  yticklabel style={text=nwMuted}, ylabel={Held-out origin-probe ROC--AUC},",
        "  ylabel style={text=nwMuted}]",
        "\\addplot[draw=none, name path=olo] table[x=k, y=origin_lo] \\nwVocab;",
        "\\addplot[draw=none, name path=ohi] table[x=k, y=origin_hi] \\nwVocab;",
        "\\addplot[fill=nwMuted, fill opacity=0.18, draw=none] fill between[of=olo and ohi];",
        "\\addplot[nwMuted, nw dashed, line width=1.1pt, mark=square*, mark size=1.5pt,",
        "  mark options={solid, line width=1pt}] table[x=k, y=origin_med] \\nwVocab;",
        "\\end{axis}",
        "% F1 of the word model (left axis)",
        "\\begin{axis}[nw a, axis x line*=bottom, axis y line*=left, ymin=0.85, ymax=0.962,",
        "  ytick={0.86,0.88,0.9,0.92,0.94,0.96}, yticklabels={0.86,0.88,0.90,0.92,0.94,0.96},",
        f"  ymajorgrids, xticklabels={{{tick_labels}}},",
        "  xlabel={Most corpus-specific n-grams removed (\\(k\\))}, xlabel style={yshift=-9pt},",
        "  ylabel={L2 F1 (word TF--IDF)},",
        "  legend style={at={(0.015,0.015)}, anchor=south west, row sep=-0.6pt}]",
        "\\addplot[forget plot, draw=none, name path=clo] table[x=k, y=clean_lo] \\nwVocab;",
        "\\addplot[forget plot, draw=none, name path=chi] table[x=k, y=clean_hi] \\nwVocab;",
        "\\addplot[forget plot, fill=oiBlue, fill opacity=0.11, draw=none] fill between[of=clo and chi];",
        "\\addplot[forget plot, draw=none, name path=alo] table[x=k, y=aug_lo] \\nwVocab;",
        "\\addplot[forget plot, draw=none, name path=ahi] table[x=k, y=aug_hi] \\nwVocab;",
        "\\addplot[forget plot, fill=oiBlue, fill opacity=0.20, draw=none] fill between[of=alo and ahi];",
        "\\addplot[forget plot, oiBlue, nw curve, nw dotted, mark=*,",
        "  mark options={solid, line width=1pt, fill=white}] table[x=k, y=clean_med] \\nwVocab;",
        "\\addplot[forget plot, oiBlue, nw curve, mark=*, mark options={line width=1pt}]",
        "  table[x=k, y=aug_med] \\nwVocab;",
        "\\addlegendimage{oiBlue, nw curve, mark=*, mark options={line width=1pt}}",
        "\\addlegendentry{Augmented}",
        "\\addlegendimage{oiBlue, nw curve, nw dotted, mark=*,",
        "  mark options={solid, line width=1pt, fill=white}}",
        "\\addlegendentry{Clean-only}",
        "\\addlegendimage{nwMuted, nw dashed, line width=1.1pt, mark=square*, mark size=1.5pt,",
        "  mark options={solid, line width=1pt}}",
        "\\addlegendentry{Origin probe (right axis)}",
        "\\addlegendimage{legend image code/.code={\\path[fill=oiBlue, fill opacity=0.2, draw=none]",
        "  (0cm,-0.09cm) rectangle (0.6cm,0.09cm);}}",
        f"\\addlegendentry{{5th--95th percentile ({d['n_partitions']} partitions)}}",
        "% mean paired difference (augmented - clean) under the tick labels",
    ]
    for k, label in zip(d["ticks"], d["delta_labels"]):
        lines.append(f"\\node[anchor=north, text=oiBlue] at ([yshift=-10.5pt]axis cs:"
                     f"{num(k / 1000)},0.85) {{{tex(label)}}};")
    lines += [
        "\\node[anchor=north east, text=oiBlue] at ([xshift=-1pt, yshift=-10.5pt]rel axis cs:0,0)",
        "  {mean \\(\\Delta\\)F1};",
        "\\end{axis}",
        "% share of the vocabulary removed (top axis)",
        "\\begin{axis}[nw a, axis y line=none, axis x line*=top, ymin=0, ymax=1,",
        f"  xticklabels={{{','.join(tex(s) for s in d['share_labels'])}}},",
        "  x axis line style={draw=nwMuted}, xtick style={draw=nwMuted},",
        "  xticklabel style={text=nwMuted},",
        f"  xlabel={{Share of the {d['vocabulary'] / 1000:.1f}k-term vocabulary removed}},",
        "  xlabel style={text=nwMuted, font=\\scriptsize}]",
        "\\end{axis}",
        "\\node[nw title] at (0.05,2.90) {(a) Vocabulary neutralization (word TF--IDF)};",
    ]

    # ---- (b)
    y_lo, y_hi = -0.022, 0.275

    def axes_y(fraction):  # data y at a fraction of the axis height (linear axis)
        return y_lo + fraction * (y_hi - y_lo)

    lines += [
        "% ---- (b) linear erasure and DANN (MiniLM): box = quartiles, whiskers = the data points",
        "% nearest inside the 5th/95th percentiles (matplotlib's boxplot_stats), dots = partitions,",
        "% diamonds = means",
        f"\\begin{{axis}}[nw, at={{(nwB)}}, width={lw(b_width * unit)}, height={lw(height * unit)},",
        f"  axis x line*=bottom, axis y line*=left, xmin=-0.55, xmax=5, ymin={y_lo}, ymax={y_hi},",
        "  ytick={0,0.05,0.1,0.15,0.2}, yticklabels={0.00,0.05,0.10,0.15,0.20}, ymajorgrids,",
        "  xtick={" + ",".join(num(b["position"]) for b in d["boxes"]) + "},",
        "  xticklabels={" + ",".join("{" + tex(b["label"]) + "}" for b in d["boxes"]) + "},",
        "  xticklabel style={align=center, yshift=1pt}, xtick style={draw=none},",
        "  ylabel={L2 \\(\\Delta\\)F1 (augmented \\(-\\) clean)}]",
        "\\draw[nwZero, line width=0.8pt] (axis cs:-0.55,0) -- (axis cs:5,0);",
        f"\\draw[nwGrid, line width=0.8pt, dash pattern=on 0.8pt off 1.3pt] (axis cs:2.725,{y_lo})"
        f" -- (axis cs:2.725,{num(axes_y(0.80))});",
    ]
    half, cap = 0.26, 0.13
    for b in d["boxes"]:
        s, x = b["stats"], b["position"]
        left, right = num(x - half), num(x + half)
        lines += [
            f"% {b['label'].replace(chr(10), ' ')}: q1 {num(s['q1'])}, median {num(s['med'])},"
            f" q3 {num(s['q3'])}, whiskers {num(s['whislo'])} / {num(s['whishi'])}",
            f"\\fill[oiGreen, fill opacity=0.18] (axis cs:{left},{num(s['q1'])}) rectangle"
            f" (axis cs:{right},{num(s['q3'])});",
            f"\\draw[oiGreen, line width=0.9pt] (axis cs:{left},{num(s['q1'])}) rectangle"
            f" (axis cs:{right},{num(s['q3'])})",
            f"  (axis cs:{num(x)},{num(s['q1'])}) -- (axis cs:{num(x)},{num(s['whislo'])})"
            f" (axis cs:{num(x - cap)},{num(s['whislo'])}) -- (axis cs:{num(x + cap)},{num(s['whislo'])})",
            f"  (axis cs:{num(x)},{num(s['q3'])}) -- (axis cs:{num(x)},{num(s['whishi'])})"
            f" (axis cs:{num(x - cap)},{num(s['whishi'])}) -- (axis cs:{num(x + cap)},{num(s['whishi'])});",
            f"\\draw[oiGreen, line width=1.5pt] (axis cs:{left},{num(s['med'])}) --"
            f" (axis cs:{right},{num(s['med'])});",
        ]
    lines += [
        f"% one dot per partition: x = position + rank-ordered jitter, y = L2 delta F1;",
        f"% {len(d['boxes'][0]['values'])} rows per group, groups in axis order",
        "\\addplot[only marks, mark=*, mark size=1.32pt, oiGreen,",
        "  mark options={draw=none, fill opacity=0.75}] table {",
        "  x y",
    ]
    lines += [f"  {num(x)} {num(y)}" for b in d["boxes"] for x, y in zip(b["x"], b["values"])]
    lines += [
        "  };",
        "\\addplot[only marks, mark=square*, mark size=1.8pt, mark options={rotate=45, fill=white,",
        "  draw=nwInk, line width=0.8pt}] coordinates {"
        + coordinates((b["position"], b["mean"]) for b in d["boxes"]) + "};",
        "% held-out origin-probe ROC-AUC of each representation",
        "\\node[anchor=south, text=nwMuted] at (rel axis cs:0.5,0.955)",
        "  {held-out origin-probe ROC--AUC of the representation};",
    ]
    for b in d["boxes"]:
        lines.append(f"\\node[anchor=south, text=nwInk] at (axis cs:{num(b['position'])},"
                     f"{num(axes_y(0.865))}) {{{b['auc']:.2f}}};")
        if b["head"] is not None:
            lines.append(f"\\node[anchor=south, text=nwMuted, font=\\nwSmall] at (axis cs:"
                         f"{num(b['position'])},{num(axes_y(0.795))}) {{(own head {b['head']:.2f})}};")
    lines.append("% groups")
    for (start, stop), text in (((0.0, 2.0), "linear concept erasure\\\\(MiniLM + LR)"),
                                ((3.45, 4.45), "domain-adversarial\\\\(MiniLM MLP)")):
        lines += [
            f"\\draw[nwMuted, line width=0.7pt] ([yshift=-24pt]axis cs:{num(start - 0.25)},{y_lo})"
            f" -- ([yshift=-24pt]axis cs:{num(stop + 0.25)},{y_lo});",
            f"\\node[anchor=north, align=center, text=nwMuted, inner sep=1pt]",
            f"  at ([yshift=-25.5pt]axis cs:{num((start + stop) / 2)},{y_lo}) {{{text}}};",
        ]
    lines += [
        "\\end{axis}",
        "\\node[nw title] at (3.78,2.90) {(b) Linear erasure and DANN (MiniLM)};",
        "\\end{tikzpicture}%",
    ]
    return "\n".join(lines) + "\n"


# ==========================================================================
# 2. why L3 fails: proximity and topics (Figure fig:l3why)
# ==========================================================================
def l3_explained_data() -> dict:
    """The numbers figure_l3_explained() plots (its data lines, verbatim)."""
    proximity = pd.read_csv(RESULTS / "revision_r4_proximity_recall.csv",
                            dtype={"quartile": str})
    topics = pd.read_csv(RESULTS / "revision_r4_l3_topics.csv")

    # ---- (a) recall by similarity quartile
    n_partitions = proximity["seed"].nunique()
    curves = []
    for corpus, label, colour, marker in SRC.PROXIMITY_CORPORA:
        block = proximity[proximity["corpus"] == corpus]
        quart = block[block["quartile"] != "all"].groupby("quartile")
        x = quart["similarity_median"].mean().sort_index()
        y = quart["recall_0.5"].mean().sort_index()
        low = y - quart["recall_0.5"].min().sort_index()
        high = quart["recall_0.5"].max().sort_index() - y
        overall = block[block["quartile"] == "all"]
        n_all = int(overall["n"].iloc[0])
        curves.append({"corpus": corpus, "label": f"{label} ($n$ = {n_all:,})",
                       "color": colour, "marker": marker, "quartiles": list(x.index),
                       "x": x.to_numpy(), "y": y.to_numpy(), "low": low.to_numpy(),
                       "high": high.to_numpy()})

    # ---- (b) topic shares and per-topic recall
    ordered = topics.sort_values("share_L3 human phishing", ascending=False).reset_index(drop=True)
    labels = [SRC._topic_label(terms) for terms in ordered["top_terms"]]
    shares = ordered[[column for column, _ in SRC.TOPIC_SHARES]].to_numpy()
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)
    share_cells = []
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            value = 100 * shares[i, j]
            text = "0" if value == 0 else (f"{value:.0f}" if value >= 9.95 else f"{value:.1f}")
            share_cells.append({"row": i, "col": j, "share": shares[i, j],
                                "meta": float(norm(shares[i, j])), "text": text,
                                "white": bool(norm(shares[i, j]) > 0.62)})
    recall = ordered[["recall_human_at_0.5", "recall_nazario_at_0.5"]].to_numpy()
    counts = ordered[["n_human", "n_nazario"]].to_numpy()
    recall_cmap = ListedColormap(plt.get_cmap("Oranges")(np.linspace(0.0, 0.85, 256)))
    recall_cells = []
    for i in range(recall.shape[0]):
        for j in range(recall.shape[1]):
            dark = recall[i, j] > 0.66
            recall_cells.append({"row": i, "col": j, "recall": recall[i, j],
                                 "text": f"{recall[i, j]:.2f}", "n": f"{counts[i, j]:,}",
                                 "white": bool(dark)})
    return {"n_partitions": n_partitions, "curves": curves, "labels": labels,
            "share_cells": share_cells, "recall_cells": recall_cells, "rows": len(ordered),
            "share_cmap": plt.get_cmap("GnBu"), "recall_cmap": recall_cmap,
            "n_human": int(ordered["n_human"].sum()), "n_nazario": int(ordered["n_nazario"].sum())}


def colormap_spec(name: str, cmap, note: str, samples: int = 17) -> list[str]:
    """A pgfplots colormap sampled at equal steps from a matplotlib colormap."""
    rgb = [tuple(int(round(255 * c)) for c in cmap(i / (samples - 1))[:3]) for i in range(samples)]
    entries = [f"rgb255=({r},{g},{b})" for r, g, b in rgb]
    return ([f"  colormap={{{name}}}{{% {note}"]
            + ["    " + " ".join(entries[i:i + 4]) for i in range(0, samples, 4)]
            + ["  },"])


def l3_explained_tex(d: dict) -> str:
    unit = 1.0 / 7.1
    top = 2.80
    rows = d["rows"]
    cell_h = 0.208
    share_w, recall_w = 0.43, 0.48
    bottom = top - rows * cell_h
    left_shares = 3.76
    n_share_cols = len(SRC.TOPIC_SHARES)
    left_recall = left_shares + n_share_cols * share_w + 0.07
    lines = preamble("l3_explained", "results/revision_r4_proximity_recall.csv and "
                     "results/revision_r4_l3_topics.csv", "\\linewidth", unit,
                     ["nwInk", "nwMuted", "nwGrid", "nwNazarioEarly"])
    lines += [
        "\\pgfplotsset{",
        *colormap_spec("nwGnBu", d["share_cmap"], "matplotlib GnBu"),
        *colormap_spec("nwOranges", d["recall_cmap"], "matplotlib Oranges from 0 to 0.85"),
        "  nw heat/.style={scale only axis, clip=false, enlargelimits=false, y dir=reverse,",
        f"    ymin=-0.5, ymax={rows - 0.5}, axis line style={{draw=none}}, tick style={{draw=none}},",
        "    xticklabel pos=top, tick label style={font=\\scriptsize},",
        "    xticklabel style={align=center, inner sep=1.5pt, text depth=2pt},",
        f"    height={lw(rows * cell_h * unit)}, point meta min=0, point meta max=1}},",
        "  nw cells/.style={matrix plot*, point meta=explicit, draw=white, line width=0.9pt},",
        "  nw cell text/.style={only marks, mark=none, point meta=explicit symbolic,",
        "    nodes near coords, nodes near coords style={anchor=center, text=nwInk, inner sep=0pt}},",
        "}",
        "\\path (0,0.66) (7.1,0.66);% fixes the width at \\linewidth",
    ]
    # ---- (a)
    for index, curve in enumerate(d["curves"]):
        rows_a = [[q, x, y, lo, hi] for q, x, y, lo, hi in
                  zip(curve["quartiles"], curve["x"], curve["y"], curve["low"], curve["high"])]
        lines.append(table(f"\\nwProx{'ABCD'[index]}", ["quartile", "similarity", "recall",
                                                        "below", "above"], rows_a,
                           f"(a) {curve['corpus']}: means over partitions; below/above = "
                           f"distance to the partition minimum/maximum"))
    lines += [
        "% ---- (a) recall against proximity, by quartile",
        place("nwA", 0.50, 0.66),
        f"\\begin{{axis}}[nw, at={{(nwA)}}, width={lw(1.98 * unit)}, height={lw((top - 0.66) * unit)},",
        "  axis x line*=bottom, axis y line*=left, xmin=0.31, xmax=0.855, ymin=-0.03, ymax=1.05,",
        "  xtick={0.4,0.5,0.6,0.7,0.8}, xticklabels={0.4,0.5,0.6,0.7,0.8},",
        "  ytick={0,0.2,0.4,0.6,0.8,1}, yticklabels={0.0,0.2,0.4,0.6,0.8,1.0}, grid=major,",
        "  xlabel={Max.\\ cosine similarity to training set\\\\(MiniLM; median within quartile)},",
        "  xlabel style={align=center}, ylabel={Recall at threshold 0.5},",
        "  legend style={at={(-0.03,1.01)}, anchor=south west, row sep=-0.8pt},",
        "  legend image post style={xscale=0.8}]",
    ]
    for index, curve in enumerate(d["curves"]):
        style = f"{color(curve['color'])}, line width=1.4pt, {mark(curve['marker'], 'solid', 'line width=1pt')}"
        lines += [
            f"\\addplot[forget plot, {style},",
            "  error bars/.cd, y dir=both, y explicit, error bar style={line width=0.8pt},",
            "  error mark options={rotate=90, mark size=1.8pt, line width=0.8pt}]",
            f"  table[x=similarity, y=recall, y error plus=above, y error minus=below]"
            f" \\nwProx{'ABCD'[index]};",
        ]
    lines.append("% legend: line, marker and error bar, as matplotlib draws an errorbar handle")
    for curve in d["curves"]:
        name = color(curve["color"])
        lines += [
            f"\\addlegendimage{{legend image code/.code={{\\draw[{name}, line width=0.8pt]",
            "  (0.3cm,-0.1cm) -- (0.3cm,0.1cm) (0.25cm,-0.1cm) -- (0.35cm,-0.1cm)"
            " (0.25cm,0.1cm) -- (0.35cm,0.1cm);",
            f"  \\draw[{name}, line width=1.4pt, {mark(curve['marker'], 'solid', 'line width=1pt')},",
            "  mark repeat=2, mark phase=2] plot coordinates {(0cm,0cm) (0.3cm,0cm) (0.6cm,0cm)};}}",
            f"\\addlegendentry{{{tex(curve['label'])}}}",
        ]
    l3 = [c for c in d["curves"] if c["corpus"].startswith("L3")][0]
    for q, x, y in zip(l3["quartiles"], l3["x"], l3["y"]):
        if q == "4":
            lines.append(f"\\node[anchor=west, text=nwMuted, inner sep=1pt] at "
                         f"([xshift=6pt, yshift=-1pt]axis cs:{num(x)},{num(y)}) {{Q{q}}};")
        else:
            lines.append(f"\\node[anchor=south, text=nwMuted, inner sep=1pt] at "
                         f"([yshift=4.5pt]axis cs:{num(x)},{num(y)}) {{Q{q}}};")
    lines += [
        "\\end{axis}",
        "\\node[nw title] at (0.02,3.68) {(a) Recall vs proximity};",
        f"\\node[anchor=base west, text=nwMuted] at (0.02,3.52) {{augmented word TF--IDF, "
        f"{d['n_partitions']} reference partitions}};",
    ]
    # ---- (b)
    share_rows = []
    for cell in d["share_cells"]:
        text = cell["text"]
        share_rows.append([cell["col"], cell["row"], cell["share"], cell["meta"],
                           "{\\color{white}" + text + "}" if cell["white"] else text])
    recall_rows = []
    for cell in d["recall_cells"]:
        value = "{\\color{white}" + cell["text"] + "}" if cell["white"] else cell["text"]
        count = ("{\\color{white}" if cell["white"] else "{\\color{nwMuted}") + "n~=~" + cell["n"] + "}"
        recall_rows.append([cell["col"], cell["row"], cell["recall"], cell["row"] - 0.15,
                            cell["row"] + 0.27, value, count])
    lines += [
        table("\\nwShares", ["col", "row", "share", "meta", "label"], share_rows,
              "(b) share of each corpus's phishing whose dominant topic is the row's topic;\n"
              "meta = PowerNorm(gamma 0.45) of the share, colored by GnBu"),
        table("\\nwRecall", ["col", "row", "recall", "label_y", "count_y", "label", "count"],
              recall_rows, "(b) recall at 0.5 per topic (columns: L3 human phishing, Nazario "
              "2015-2025), colored by\nOranges from 0 to 0.85; label_y/count_y place the two "
              "lines of text in each cell"),
        "% ---- (b) topic shares (left) and per-topic recall (right)",
        place("nwShares", left_shares, bottom),
        place("nwRecall", left_recall, bottom),
        f"\\begin{{axis}}[nw heat, colormap name=nwGnBu, at={{(nwShares)}},"
        f" width={lw(n_share_cols * share_w * unit)},",
        f"  xmin=-0.5, xmax={n_share_cols - 0.5}, xtick={{0,...,{n_share_cols - 1}}},",
        "  xticklabels={" + ",".join("{" + tex(lab) + "}" for _, lab in SRC.TOPIC_SHARES) + "},",
        f"  ytick={{0,...,{rows - 1}}}, yticklabel style={{align=right, inner sep=2pt}},",
        "  yticklabels={",
        braced_list(tex(label) for label in d["labels"]),
        "  }]",
        f"\\addplot[nw cells, mesh/cols={n_share_cols}, mesh/rows={rows}]",
        "  table[x=col, y=row, meta=meta] \\nwShares;",
        "\\addplot[nw cell text] table[x=col, y=row, meta=label] \\nwShares;",
        "\\end{axis}",
        f"\\begin{{axis}}[nw heat, colormap name=nwOranges, at={{(nwRecall)}},"
        f" width={lw(2 * recall_w * unit)},",
        "  xmin=-0.5, xmax=1.5, xtick={0,1}, ytick=\\empty,",
        "  xticklabels={{L3 human\\\\phishing},{Nazario\\\\2015--25}}]",
        f"\\addplot[nw cells, mesh/cols=2, mesh/rows={rows}]",
        "  table[x=col, y=row, meta=recall] \\nwRecall;",
        "\\addplot[nw cell text] table[x=col, y=label_y, meta=label] \\nwRecall;",
        "\\addplot[nw cell text, nodes near coords style={font=\\nwSmall}]",
        "  table[x=col, y=count_y, meta=count] \\nwRecall;",
        "\\end{axis}",
    ]
    header_y = top + 0.36
    share_width = n_share_cols * share_w
    lines += [
        f"\\node[anchor=south, text=nwInk] at ({num(left_shares + share_width / 2)},{num(header_y)})",
        "  {Share of each corpus's phishing (\\%)};",
        f"\\node[anchor=south, text=nwInk] at ({num(left_recall + recall_w)},{num(header_y)})",
        "  {Recall at 0.5 (word, aug.)};",
    ]
    for x0, width in ((left_shares, share_width), (left_recall, 2 * recall_w)):
        lines.append(f"\\draw[nwMuted, line width=0.6pt] ({num(x0 + 0.03)},{num(header_y - 0.035)})"
                     f" -- ({num(x0 + width - 0.03)},{num(header_y - 0.035)});")
    lines += [
        "\\node[nw title] at (2.72,3.68) {(b) Topic shares and per-topic recall};",
        f"\\node[anchor=base west, text=nwMuted] at (2.72,3.52) {{{rows} NMF topics, rows labeled "
        "by their first three terms;",
        "  sorted by share of L3 human phishing};",
        "\\end{tikzpicture}%",
    ]
    return "\n".join(lines) + "\n"


# ==========================================================================
# 3. deployment robustness (Figure fig:deploy)
# ==========================================================================
def deployment_robustness_data() -> dict:
    """The numbers figure_deployment_robustness() plots (its data lines, verbatim)."""
    estimation = pd.read_csv(RESULTS / "revision_r4_deploy_prevalence_estimation.csv")
    validation = pd.read_csv(RESULTS / "revision_r4_deploy_rule_validation.csv")
    sensitivity = pd.read_csv(RESULTS / "revision_r4_deploy_score_sensitivity.csv")
    stream_keys = [key for key, _, _ in SRC.STREAMS]
    FLOOR = SRC.FLOOR

    # ---- (a) prevalence estimation
    medians = (estimation.groupby(["stream", "estimator", "true_prevalence"])["ratio_to_truth"]
               .median())
    offsets = np.linspace(-0.28, 0.28, len(SRC.ESTIMATORS))
    prevalences = sorted(estimation["true_prevalence"].unique())
    points_a = []
    for index, prevalence in enumerate(prevalences):
        for s_index, stream in enumerate(stream_keys):
            for offset, (estimator, _, marker) in zip(offsets, SRC.ESTIMATORS):
                value = float(medians[(stream, estimator, prevalence)])
                clipped = value < FLOOR
                points_a.append({"panel": index, "prevalence": prevalence, "stream": stream,
                                 "estimator": estimator, "marker": marker,
                                 "x": s_index + offset, "y": max(value, FLOOR),
                                 "median": value, "open": clipped})

    # ---- (b) held-out rule validation
    table_b = validation.groupby(["rule", "stream"])[["agreement", "mean_excess"]].mean()
    y_offsets = np.linspace(-0.27, 0.27, len(stream_keys))
    points_b = []
    for r_index, (rule, _) in enumerate(SRC.PROCEDURES):
        for offset, stream in zip(y_offsets, stream_keys):
            if (rule, stream) not in table_b.index:
                continue
            row = table_b.loc[(rule, stream)]
            points_b.append({"rule": rule, "stream": stream, "y": r_index + offset,
                             "mean_excess": float(row["mean_excess"]),
                             "agreement": float(row["agreement"])})

    # ---- (c) score-distribution sensitivity
    means = sensitivity.groupby(["scope", "perturbation", "value", "stream"])[
        ["pi_r_purity", "mean_excess"]].mean()
    conditions = [("none", "none", 0.0, 0.0)]
    position_x = 1.4
    kind_spans, scope_spans = [], []
    for scope in ("validation and target", "target only"):
        scope_start = position_x
        for kind, values, label in SRC.PERTURBATIONS:
            kind_start = position_x
            for value in values:
                conditions.append((scope, kind, value, position_x))
                position_x += 1.0
            kind_spans.append((label, kind_start, position_x - 1.0))
            position_x += 0.55
        scope_spans.append((scope, scope_start, position_x - 1.55))
        position_x += 0.75
    x_offsets = np.linspace(-0.24, 0.24, len(stream_keys))
    groups = []  # one entry per (scope, kind): the points the rule panel connects
    for scope in ("none", "validation and target", "target only"):
        for kind in ("none", "temperature", "shift", "noise"):
            members = [(x, {stream: means.loc[(sc, kd, val, stream)] for stream in stream_keys
                            if (sc, kd, val, stream) in means.index})
                       for sc, kd, val, x in conditions if sc == scope and kd == kind]
            if members:
                groups.append({"scope": scope, "kind": kind, "members": members})
    tick_labels = ["none"] + [SRC._value_label(kind, value) for _, kind, value, _ in conditions[1:]]
    return {"stream_keys": stream_keys, "prevalences": prevalences, "points_a": points_a,
            "points_b": points_b, "conditions": conditions, "kind_spans": kind_spans,
            "scope_spans": scope_spans, "x_offsets": x_offsets, "groups": groups,
            "tick_labels": tick_labels, "xmax": position_x - 1.2, "floor": FLOOR}


def deployment_robustness_tex(d: dict) -> str:
    unit = 0.96 / 7.1
    stream_keys = d["stream_keys"]
    names = {key: color(hex_code) for key, _, hex_code in SRC.STREAMS}
    estimator_style = {estimator: "nw " + estimator.replace("_t", " t").replace(".", "")
                       for estimator, _, _ in SRC.ESTIMATORS}
    lines = preamble("deployment_robustness", "results/revision_r4_deploy_{prevalence_estimation,"
                     "rule_validation,score_sensitivity}.csv", "0.96\\linewidth", unit,
                     ["nwInk", "nwMuted", "nwGrid", "nwBand", "nwFloor", "nwBaseline"])
    lines.append("\\pgfplotsset{% estimator marks of panel (a); 'open': median at the floor")
    for estimator, _, marker in SRC.ESTIMATORS:
        style = estimator_style[estimator]
        lines += [f"  {style}/.style={{only marks, {mark(marker, 'line width=0.9pt')}}},",
                  f"  {style} open/.style={{only marks, {mark(marker, 'line width=0.9pt', 'fill=white')}}},"]
    lines += [
        f"  nw dot/.style={{only marks, {mark('o', 'line width=1pt', scale=0.95)}}},",
        f"  nw small dot/.style={{only marks, {mark('o', 'line width=1pt', scale=0.8)}}},",
        "  nw legend/.style={hide axis, scale only axis, width=1pt, height=1pt, xmin=0, xmax=1,",
        "    ymin=0, ymax=1, legend style={at={(0.5,0.5)}, anchor=center, draw=none, fill=none,",
        "    font=\\scriptsize}},",
        "}",
        "\\path (0,0.62) (7.1,0.62);% fixes the width at 0.96\\linewidth",
        "% streams (legend across the top)",
        place("nwStreams", 3.55, 5.74),
        "\\begin{axis}[nw legend, at={(nwStreams)}, legend columns=4,",
        "  legend style={/tikz/every even column/.append style={column sep=9pt}}]",
    ]
    for key, label, _ in SRC.STREAMS:
        lines.append(f"\\addlegendimage{{only marks, {names[key]}, {mark('o', 'line width=1pt', scale=1.05)}}}")
        lines.append(f"\\addlegendentry{{{tex(label)}}}")
    lines.append("\\end{axis}")

    # ---- (a)
    a_bottom, a_height, a_left, a_width, a_gap = 3.62, 1.38, 0.66, 1.08, 0.07
    x_hi = num(len(stream_keys) - 0.45)
    short = ",".join(f"\\textcolor{{{names[s]}}}{{\\textbf{{{tex(SRC.STREAM_SHORT[s])}}}}}"
                     for s in stream_keys)
    for index, prevalence in enumerate(d["prevalences"]):
        left = a_left + index * (a_width + a_gap)
        if index == 0:
            y_axis = ["  ytick={0.01,0.1,1,10,100,1000}, yticklabels={\\(\\le\\)0.01,0.1,1,10,100,1000},",
                      "  ylabel={Estimated / true prevalence\\\\(median over models \\(\\times\\) partitions)},",
                      "  ylabel style={align=center, inner sep=1pt},"]
        else:
            y_axis = ["  ytick={0.01,0.1,1,10,100,1000}, yticklabels={}, ytick style={draw=none},"]
        lines += [
            f"% ---- (a) true prevalence {100 * prevalence:g}%: median estimate / truth, drawn at "
            f"{num(d['floor'])} when below it",
            place(f"nwA{index}", left, a_bottom),
            f"\\begin{{axis}}[nw, at={{(nwA{index})}}, width={lw(a_width * unit)},"
            f" height={lw(a_height * unit)},",
            "  axis x line*=bottom, axis y line*=left, ymode=log, ymin=6e-3, ymax=2.5e3,",
            f"  xmin=-0.55, xmax={x_hi}, xtick={{0,...,{len(stream_keys) - 1}}},",
            f"  xticklabels={{{short}}},",
            "  xtick style={draw=none}, xticklabel shift=-2pt,",
            *y_axis,
            f"  title={{true \\(\\pi\\) = {100 * prevalence:g}\\%}}, title style={{font=\\scriptsize, yshift=-5pt}}]",
            f"\\fill[nwFloor] (axis cs:-0.55,6e-3) rectangle (axis cs:{x_hi},1.5e-2);",
            f"\\fill[nwBand] (axis cs:-0.55,0.5) rectangle (axis cs:{x_hi},2);",
        ]
        for tick in (0.01, 0.1, 1, 10, 100, 1000):
            lines.append(f"\\draw[nw grid line] (axis cs:-0.55,{tick}) -- (axis cs:{x_hi},{tick});")
        lines.append(f"\\draw[nwMuted, line width=0.8pt] (axis cs:-0.55,1) -- (axis cs:{x_hi},1);")
        for p in d["points_a"]:
            if p["panel"] != index:
                continue
            style = estimator_style[p["estimator"]] + (" open" if p["open"] else "")
            lines.append(f"\\addplot[{style}, {names[p['stream']]}] coordinates "
                         f"{{({num(p['x'])},{num(p['y'])})}};"
                         f" % {SRC.STREAM_SHORT[p['stream']]} {p['estimator']}: median {num(p['median'])}")
        lines.append("\\end{axis}")
    # estimator legend
    lines += [
        place("nwEstimators", 2.23, 5.255),
        "\\begin{axis}[nw legend, at={(nwEstimators)}, legend columns=5,",
        "  legend style={/tikz/every even column/.append style={column sep=8pt}}]",
    ]
    for estimator, label, _ in SRC.ESTIMATORS:
        lines.append(f"\\addlegendimage{{{estimator_style[estimator]}, nwInk}}")
        lines.append(f"\\addlegendentry{{{tex(label)}}}")
    lines += [
        f"\\addlegendimage{{{estimator_style[SRC.ESTIMATORS[0][0]]} open, nwInk}}",
        "\\addlegendentry{median \\(\\le\\) 0.01 (at floor)}",
        "\\end{axis}",
        "\\node[nw title] at (0.02,5.42) {(a) Label-free prevalence estimates};",
    ]

    # ---- (b)
    b_left = 5.02
    n_rules = len(SRC.PROCEDURES)
    for axis_name, left, width, options, column in (
            ("excess", b_left, 1.05,
             ["  xmode=log, xmin=4e-2, xmax=25, xtick={0.1,1,10}, xticklabels={10\\%,100\\%,1000\\%},",
              "  xlabel={Mean excess cost\\\\over best rule},",
              f"  ytick={{0,...,{n_rules - 1}}}, yticklabel style={{align=right}}, ytick style={{draw=none}},",
              "  yticklabels={",
              braced_list(tex(label) for _, label in SRC.PROCEDURES),
              "  }]"],
             "mean_excess"),
            ("agree", b_left + 1.05 + 0.10, 0.82,
             ["  xmin=-0.04, xmax=1.04, xtick={0,0.5,1}, xticklabels={0,0.5,1},",
              "  xlabel={Agreement with\\\\best rule}, ytick=\\empty]"],
             "agreement")):
        x_lo, x_hi = (4e-2, 25) if axis_name == "excess" else (-0.04, 1.04)
        lines += [
            f"% ---- (b) {column.replace('_', ' ')} (equal-weight mean over models), one row per "
            f"procedure",
            place(f"nwB{axis_name}", left, a_bottom),
            f"\\begin{{axis}}[nw, at={{(nwB{axis_name})}}, width={lw(width * unit)},"
            f" height={lw(a_height * unit)},",
            "  axis x line*=bottom, axis y line*=left, y dir=reverse,",
            f"  ymin=-0.5, ymax={n_rules - 0.5}, xmajorgrids, xlabel style={{align=center}},",
            *options,
        ]
        for r_index in range(n_rules - 1):
            lines.append(f"\\draw[nwGrid, line width=0.5pt] (axis cs:{num(x_lo)},{num(r_index + 0.5)})"
                         f" -- (axis cs:{num(x_hi)},{num(r_index + 0.5)});")
        for stream in stream_keys:
            points = [(p[column], p["y"]) for p in d["points_b"] if p["stream"] == stream]
            lines.append(f"\\addplot[nw dot, {names[stream]}] coordinates {{{coordinates(points)}}};")
        lines.append("\\end{axis}")
    lines.append("\\node[nw title] at (3.98,5.42) {(b) Held-out rule validation (all models)};")

    # ---- (c)
    c_left, c_width = 0.66, 6.36
    xmin, xmax = -0.75, d["xmax"]
    n_streams = len(stream_keys)
    header = ([f"x{i}" for i in range(n_streams)] + [f"purity{i}" for i in range(n_streams)]
              + [f"excess{i}" for i in range(n_streams)])
    rows = []
    for g_index, group in enumerate(d["groups"]):
        if g_index:
            rows.append(["nan"] * len(header))  # breaks the connecting lines between groups
        for x, cells in group["members"]:
            rows.append([x + offset for offset in d["x_offsets"]]
                        + [cells[s]["pi_r_purity"] for s in stream_keys]
                        + [cells[s]["mean_excess"] for s in stream_keys])
    stream_note = ", ".join(f"{i} = {SRC.STREAM_SHORT[s]}" for i, s in enumerate(stream_keys))
    lines.append(table("\\nwSens", header, rows,
                       "(c) one row per condition: x position (condition + stream offset), pi*r "
                       "purity and the three-band\nrule's mean excess cost (mean over models); "
                       f"streams {stream_note}; nan rows separate the groups"))
    ticks = ",".join(num(x) for _, _, _, x in d["conditions"])
    tick_labels = ",".join(tex(label) for label in d["tick_labels"])
    scope_lines = [num(start - 0.6) for _, start, _ in d["scope_spans"]]
    for axis_name, bottom, height in (("purity", 1.80, 0.92), ("rule", 0.62, 1.08)):
        if axis_name == "purity":
            y_lo, y_hi = 0.825, 1.015
            y_axis = ["  ymin=0.825, ymax=1.015, ytick={0.85,0.9,0.95,1}, yticklabels={0.85,0.90,0.95,1.00},",
                      "  ylabel={\\(\\pi r\\) purity}, ylabel style={inner sep=1pt}, xtick=\\empty]"]
            y_ticks = (0.85, 0.9, 0.95, 1.0)
        else:
            y_lo, y_hi = 0.05, 6.5
            y_axis = ["  ymode=log, ymin=0.05, ymax=6.5, ytick={0.1,0.3,1,3},"
                      " yticklabels={10\\%,30\\%,100\\%,300\\%},",
                      "  ylabel={Three-band rule:\\\\mean excess cost},"
                      " ylabel style={align=center, inner sep=1pt},",
                      f"  xtick={{{ticks}}},",
                      f"  xticklabels={{{tick_labels}}},",
                      "  major tick length=2pt, xticklabel style={inner sep=1.5pt},"
                      " typeset ticklabels with strut]"]
            y_ticks = (0.1, 0.3, 1, 3)
        lines += [
            f"% ---- (c) {'pi*r purity' if axis_name == 'purity' else 'three-band rule: mean excess cost'}",
            place(f"nwC{axis_name}", c_left, bottom),
            f"\\begin{{axis}}[nw, at={{(nwC{axis_name})}}, width={lw(c_width * unit)},"
            f" height={lw(height * unit)},",
            f"  axis x line*=bottom, axis y line*=left, xmin={num(xmin)}, xmax={num(xmax)},",
            *y_axis,
            "% shaded column: the unperturbed baseline",
            f"\\fill[nwBaseline] (axis cs:-0.6,{num(y_lo)}) rectangle (axis cs:0.6,{num(y_hi)});",
        ]
        for tick in y_ticks:
            lines.append(f"\\draw[nw grid line] (axis cs:{num(xmin)},{num(tick)}) -- "
                         f"(axis cs:{num(xmax)},{num(tick)});")
        for x in scope_lines:
            lines.append(f"\\draw[nwMuted, line width=0.6pt, dash pattern=on 0.6pt off 1pt]"
                         f" (axis cs:{x},{num(y_lo)}) -- (axis cs:{x},{num(y_hi)});")
        column = "purity" if axis_name == "purity" else "excess"
        if axis_name == "rule":
            for i, stream in enumerate(stream_keys):
                lines.append(f"\\addplot[{names[stream]}, line width=0.6pt, opacity=0.4,"
                             f" unbounded coords=jump] table[x=x{i}, y={column}{i}] \\nwSens;")
        for i, stream in enumerate(stream_keys):
            lines.append(f"\\addplot[nw small dot, {names[stream]}] table[x=x{i}, y={column}{i}] \\nwSens;")
        if axis_name == "purity":
            for scope, start, stop in d["scope_spans"]:
                text = ("validation and target perturbed (a detector with another score distribution)"
                        if scope == "validation and target"
                        else "target only perturbed (score shift at deployment)")
                lines.append(f"\\node[anchor=south, text=nwInk, inner sep=1pt] at ([yshift=3pt]axis cs:"
                             f"{num((start + stop) / 2)},{num(y_hi)})")
                lines.append(f"  {{{text}}};")
        else:
            for label, start, stop in d["kind_spans"]:
                lines += [
                    f"\\draw[nwMuted, line width=0.7pt] ([yshift=-14pt]axis cs:{num(start - 0.3)},{num(y_lo)})"
                    f" -- ([yshift=-14pt]axis cs:{num(stop + 0.3)},{num(y_lo)});",
                    f"\\node[anchor=base, text=nwMuted, inner sep=0pt] at ([yshift=-21.5pt]axis cs:"
                    f"{num((start + stop) / 2)},{num(y_lo)}) {{{tex(label)}}};",
                ]
            lines.append(f"\\node[anchor=base, text=nwMuted, inner sep=0pt] at ([yshift=-21.5pt]axis cs:"
                         f"0,{num(y_lo)}) {{baseline}};")
        lines.append("\\end{axis}")
    lines += [
        "\\node[nw title] at (0.02,3.03) {(c) Sensitivity to the score distribution};",
        "\\node[anchor=base west, text=nwMuted] at (3.33,3.03)",
        "  {word TF--IDF and MiniLM, augmented; all partitions};",
        "\\end{tikzpicture}%",
    ]
    return "\n".join(lines) + "\n"


# ==========================================================================
FIGURES_TEX = {
    "pipeline_signature": (pipeline_signature_data, pipeline_signature_tex),
    "l3_explained": (l3_explained_data, l3_explained_tex),
    "deployment_robustness": (deployment_robustness_data, deployment_robustness_tex),
}


def report(name: str, d: dict) -> None:
    """Print a few plotted values, for checking against the text and the PNG script."""
    if name == "pipeline_signature":
        print(f"  (a) origin-probe AUC column: {d['auc_column']}; partitions {d['n_partitions']}")
        for k, label in zip(d["ticks"], d["delta_labels"]):
            i = list(d["ks"]).index(k)
            print(f"      k {k:5d}: aug F1 median {d['curves']['augmented'][1][i]:.4f}, clean "
                  f"{d['curves']['clean'][1][i]:.4f}, origin AUC {d['curves']['origin'][1][i]:.4f},"
                  f" mean delta {label}")
        for b in d["boxes"]:
            print(f"  (b) {b['label'].replace(chr(10), ' '):24s} mean {b['mean']:+.4f} median "
                  f"{b['stats']['med']:+.4f} origin AUC {b['auc']:.4f}"
                  + (f" head {b['head']:.4f}" if b["head"] is not None else ""))
    elif name == "l3_explained":
        for c in d["curves"]:
            print(f"  (a) {c['label']:44s} recall {np.round(c['y'], 3)} sim {np.round(c['x'], 3)}")
        print(f"  (b) {d['rows']} topics; L3 messages {d['n_human']}, Nazario {d['n_nazario']}; "
              f"first row {d['labels'][0]!r}: shares "
              + " ".join(c["text"] for c in d["share_cells"] if c["row"] == 0)
              + "; recall " + " ".join(c["text"] for c in d["recall_cells"] if c["row"] == 0))
    else:
        open_points = sum(p["open"] for p in d["points_a"])
        print(f"  (a) {len(d['points_a'])} medians, {open_points} below the floor (open markers)")
        first = [p for p in d["points_b"] if p["rule"] == SRC.PROCEDURES[0][0]]
        print("  (b) three-band rule: " + ", ".join(
            f"{SRC.STREAM_SHORT[p['stream']]} excess {p['mean_excess']:.3f} agreement "
            f"{p['agreement']:.3f}" for p in first))
        print(f"  (c) {len(d['conditions'])} conditions in {len(d['groups'])} groups")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES_TEX),
                        help="write only these figures")
    parser.add_argument("--output-dir", type=Path, default=TIKZ)
    parser.add_argument("--copy-to", type=Path, default=_bundle(),
                        help="also copy the .tex files here (default: the manuscript bundle)")
    args = parser.parse_args()
    for name in args.only or list(FIGURES_TEX):
        print(f"[{name}]")
        data_function, tex_function = FIGURES_TEX[name]
        data = data_function()
        report(name, data)
        write(name, tex_function(data), args.output_dir, args.copy_to)


if __name__ == "__main__":
    main()
