#!/usr/bin/env python3
"""TikZ/pgfplots versions of Figure 3 (multisplit_inference) and Figure 6 (tradeoff).

Round-4 redraw of two figures that make_r4_revised_figures.py draws with
matplotlib (figure group "inference" of manuscriptR4/drafts/TIKZ_BRIEF.md). Each
figure is written as one self-contained tikzpicture with its data inline, which
the manuscript \\inputs in place of the PNG:

* tikz_multisplit_inference.tex (Figure 3; 0.96\\linewidth, figure* float)
    (a) paired effect (augmented - clean) on linked-rewrite F1 for every
        partition, sorted, with that partition's own 95% grouped-bootstrap
        interval; all 100 + 100 + 20 partitions are drawn (nothing is thinned);
    (b) 95% intervals for the mean effect and for one further partition, by scope;
    (c) between-partition SD from K partitions relative to its value over all
        partitions: median and 5th-95th percentile band over the random
        orderings, and the trajectory in the order the partitions were run;
    (d) corrected resampled-t half-width relative to its large-K floor.
* tikz_tradeoff.tex (Figure 6; \\linewidth, one-column figure float)
    generator-disjoint (L2) F1 against spam hard-negative FPR, both at threshold
    0.5; one arrow per family from the clean-only to the augmented mean; whiskers
    +/-1 SD across partitions; lightly shaded "preferable" lower-right region.

Every plotted number comes from the code path of the PNG figures. The loaders and
summaries (lightweight_metrics, distilbert_metrics, partition_values, summarize,
scope_row, SCOPES, FAMILIES, TRAININGS, TEST_GROUPS, the color map) are imported
from make_r4_revised_figures.py; the per-panel preparation written inline in its
figure_multisplit() and figure_tradeoff() is copied verbatim (marked "verbatim"
below), as are the fixed axis limits, text positions and the label placement
table; the two autoscaled ranges ((a) y, (b) x) follow matplotlib's rule.
make_r4_revised_figures imports rev_common, which imports sentence_transformers
and with it torch; nothing here needs either, so a stub module of the kind
rev_common installs when the package is missing is registered first.

Text follows the paper: tick labels, legends and notes \\scriptsize, axis labels
\\footnotesize, panel titles \\footnotesize bold, Times inherited. Colors are
the preamble's oiBlue / oiGreen / oiVermillion (the script's #0072B2 / #009E73 /
#D55E00); the script's grays are defined inside each picture, which is also set
in black so that a surrounding \\color (the response letter quotes revised text
in red) does not reach it. Widths: the picture is laid out at the line width
(0.96\\linewidth for Figure 3) but never narrower than in the paper (501.12pt and
252pt); on a narrower line, such as the response letter's, the picture is laid
out at the paper's width and scaled to the line, as the PNG was. Heights scale
with the axis width. Needs pgfplots (compat 1.18) with fillbetween and the TikZ
arrows.meta library, all loaded by the paper preamble.

Inputs (results/):
  revision_r4_inference.csv, revision_r4_stabilization.csv     (rev_r_r4_inference.py)
  revision_r3_multisplit_metrics.csv, revision_r4_multisplit100_metrics.csv
  revision_distilbert_stream_metrics.csv, revision_r3_distilbert_multisplit_metrics.csv,
  revision_r4_distilbert_multisplit_metrics.csv
Outputs:
  figures/tikz/tikz_multisplit_inference.tex, figures/tikz/tikz_tradeoff.tex,
  and a copy of each in the manuscript bundle (--copy-to; default manuscriptR4/).

Run from artifact/eswa_study/revision/:
    python tikz_inference.py [--copy-to DIR]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import types

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent
OUT_DIR = STUDY / "figures" / "tikz"
TAGS = {"Word TF-IDF": "Word", "MiniLM": "Mini", "DistilBERT": "Bert"}  # TeX macro suffixes
OKABE_ITO = {"#0072B2": "oiBlue", "#009E73": "oiGreen", "#D55E00": "oiVermillion"}


def default_bundle() -> pathlib.Path:
    """manuscriptR4/ next to artifact/, or the bundle that holds this artifact copy."""
    project = STUDY.parent.parent
    if (project / "manuscriptR4").is_dir():
        return project / "manuscriptR4"
    if (project / "paper.tex").exists():  # running inside manuscriptR4/artifact/
        return project
    return project / "manuscriptR4"


def load_figure_module():
    """Import make_r4_revised_figures without loading torch."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "sentence_transformers" not in sys.modules:
        stub = types.ModuleType("sentence_transformers")

        class SentenceTransformer:  # noqa: D401 - placeholder, never called here
            def __init__(self, *args, **kwargs):
                raise RuntimeError("sentence-transformers is not needed for the figures")

        stub.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = stub
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import make_r4_revised_figures as figures  # noqa: E402

    return figures


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def num(value: float, digits: int = 7) -> str:
    """A plotted value with 7 significant digits (beyond pgfplots' own precision)."""
    text = f"{float(value):.{digits}g}"
    return "0" if text in ("-0", "0") else text


def fill(template: str, **values) -> str:
    """Replace <<key>> placeholders (TeX is full of braces, so no str.format)."""
    for key, value in values.items():
        template = template.replace(f"<<{key}>>", str(value))
    if "<<" in template:
        raise ValueError(f"unfilled placeholder near {template[template.index('<<'):][:40]!r}")
    return template


def table(name: str, columns: list[str], rows) -> str:
    """An inline \\pgfplotstableread with row sep=\\\\ (robust inside macro arguments)."""
    lines = [" ".join(columns) + r"\\"] + [" ".join(cells) + r"\\" for cells in rows]
    body = "\n  ".join(lines)
    return f"\\pgfplotstableread[row sep=\\\\]{{\n  {body}\n}}{name}\n"


def auto_limits(values, margin: float = 0.05) -> tuple[float, float]:
    """matplotlib's default autoscaling: data range plus 5% on each side.

    The PNG panels that are autoscaled ((a) in y, (b) in x) add their zero line
    last with axhline/axvline, which read the current bounds first and rescale
    only when the line lies outside them; it lies inside, so the zero line does
    not enter the limits and is left out of ``values`` here as well.
    """
    low, high = float(np.min(values)), float(np.max(values))
    pad = margin * (high - low)
    return low - pad, high + pad


def tex_family(family: str) -> str:
    return family.replace("TF-IDF", "TF--IDF")


def color_name(fig, family: str) -> str:
    """The preamble's Okabe-Ito name for the color the PNG script uses."""
    code = fig.colour(family).upper()
    if code not in OKABE_ITO:
        raise ValueError(f"{family}: color {code} is not one of the preamble's oi* colors")
    return OKABE_ITO[code]


def wrapper(frame: str, minimum: float, name: str) -> tuple[str, str]:
    """Code around the tikzpicture: lay out at max(frame, minimum), shrink to frame if wider.

    The picture is typeset in a box, in black whatever color surrounds it. In the
    paper the box is exactly ``frame`` wide and is used as it is. On a narrower
    line (the response letter's) it is laid out at the paper's width and scaled
    to the line, as the PNG was; a font wider than Times is absorbed the same way.
    """
    head = (
        "\\begingroup\\leavevmode\n"
        f"\\pgfmathsetlengthmacro{{\\{name}Frame}}{{max({frame},{minimum}pt)}}% layout width\n"
        "\\setbox0=\\hbox{\\color{black}%\n"
    )
    tail = (
        "}%\n"
        f"\\ifdim\\wd0>\\dimexpr{frame}+0.5pt\\relax\n"
        f"  \\resizebox{{{frame}}}{{!}}{{\\box0}}% e.g. the response letter: the paper's layout, scaled\n"
        "\\else\n"
        "  \\box0\n"
        "\\fi\n"
        "\\endgroup\n"
    )
    return head, tail


# --------------------------------------------------------------------------
# Figure 3: data (verbatim from figure_multisplit)
# --------------------------------------------------------------------------
def multisplit_data(fig) -> dict:
    inference = pd.read_csv(fig.RESULTS / "revision_r4_inference.csv")
    stabilization = pd.read_csv(fig.RESULTS / "revision_r4_stabilization.csv")
    single = inference[inference["scope"] == "single split (grouped bootstrap)"]
    FAMILIES, SCOPES = fig.FAMILIES, fig.SCOPES

    # (a) -- verbatim: sorting, positions, density switch, mean
    panel_a, counts = [], {}
    for index, family in enumerate(FAMILIES):
        block = single[single["family"] == family].sort_values("delta")
        k = len(block)
        counts[family] = k
        x = index + np.linspace(-0.38, 0.38, k)
        dense = k > 50
        mean = float(block["delta"].mean())
        panel_a.append({
            "family": family, "index": index, "k": k, "x": x, "dense": dense, "mean": mean,
            "delta": block["delta"].to_numpy(float), "low": block["ci_low"].to_numpy(float),
            "high": block["ci_high"].to_numpy(float), "seed": block["seed"].to_numpy(int),
            "reach_zero": int((block["ci_low"] <= 0).sum()),
        })
    # autoscaled y range of the PNG panel: bars, points and mean lines
    a_values = np.concatenate([np.r_[p["low"], p["high"], p["delta"], p["mean"]] for p in panel_a])
    a_ylim = auto_limits(a_values)

    # (b) -- verbatim: row offsets, one row per scope and family
    offsets = {"Word TF-IDF": -0.24, "MiniLM": 0.0, "DistilBERT": 0.24}
    panel_b = []
    for row, (scope, label) in enumerate(SCOPES):
        for family in FAMILIES:
            entry = fig.scope_row(inference, family, scope)
            y = row + offsets[family]
            panel_b.append({"row": row, "label": label, "family": family, "y": y,
                            "low": float(entry["ci_low"]), "high": float(entry["ci_high"]),
                            "delta": float(entry["delta"]), "scope": entry["scope"]})
    order_cov = {
        family: fig.scope_row(inference, family, SCOPES[-1][0])["scope"].split("coverage ")[-1].rstrip(")")
        for family in FAMILIES
    }
    # autoscaled x range: interval ends and mean marks
    b_values = np.r_[[e["low"] for e in panel_b], [e["high"] for e in panel_b],
                     [e["delta"] for e in panel_b]]
    b_xlim = auto_limits(b_values)

    # (c) -- verbatim: SD(K)/SD(full), random-order quantiles closed at K_full
    panel_c = []
    for family in FAMILIES:
        block = stabilization[stabilization["family"] == family]
        as_run = block[block["ordering"] == "as run"].sort_values("k")
        k_full = int(as_run["k"].max())
        sd_full = float(as_run.loc[as_run["k"] == k_full, "sd"].iloc[0])
        random = (
            block[(block["ordering"] == "random") & block["quantile"].isin(["p05", "median", "p95"])]
            .pivot(index="k", columns="quantile", values="sd")
            .sort_index()
        )
        ks = np.r_[random.index.to_numpy(), k_full]
        lower = np.r_[random["p05"].to_numpy() / sd_full, 1.0]
        median = np.r_[random["median"].to_numpy() / sd_full, 1.0]
        upper = np.r_[random["p95"].to_numpy() / sd_full, 1.0]
        panel_c.append({
            "family": family, "k_full": k_full, "sd_full": sd_full,
            "ks": ks, "lower": lower, "median": median, "upper": upper,
            "run_k": as_run["k"].to_numpy(int), "run_ratio": (as_run["sd"] / sd_full).to_numpy(float),
        })

    # (d) -- verbatim: one ratio series, checked to be the same for every family
    ratio = stabilization[stabilization["ordering"] == "as run"].pivot(
        index="k", columns="family", values="corrected_to_floor_ratio"
    )
    shared = ratio.dropna()
    if not np.allclose(shared.to_numpy(), shared.to_numpy()[:, :1]):
        raise ValueError("corrected-to-floor ratio differs between families")
    series = ratio["Word TF-IDF"].dropna().sort_index()

    return {"panel_a": panel_a, "a_ylim": a_ylim, "counts": counts, "panel_b": panel_b,
            "b_xlim": b_xlim, "order_cov": order_cov, "panel_c": panel_c, "series": series}


# --------------------------------------------------------------------------
# Figure 3: TikZ
# --------------------------------------------------------------------------
# Layout (pt, measured with the paper's fonts). The four axis boxes are placed
# explicitly; their common width is what is left of the layout width after the
# fixed label space, so the picture is exactly that wide:
#   MS_LEFT  (a)/(c) y tick labels and y label, left of the (a)/(c) axis boxes
#   MS_GAP   between the (a)/(c) and (b)/(d) axis boxes ((b)'s scope labels)
#   MS_RIGHT (b)'s last x tick label, right of the (b)/(d) axis boxes
# Heights (paper) give the PNG's 1.12 : 1 rows and its printed height, 342.5pt.
MS_FRAME = 0.96 * 522  # the paper's 0.96\linewidth in a figure* float
MS_LEFT, MS_GAP, MS_RIGHT, MS_VGAP = 39.67, 66.0, 9.08, 47.0
MS_HEIGHT_TOP, MS_HEIGHT_BOTTOM = 135.04, 120.57
MS_AXIS = (MS_FRAME - MS_LEFT - MS_GAP - MS_RIGHT) / 2

MS_HEADER = r"""% tikz_multisplit_inference.tex -- Figure 3 (fig:multisplit), drawn with pgfplots.
% Written by artifact/eswa_study/revision/tikz_inference.py from
% results/revision_r4_inference.csv and results/revision_r4_stabilization.csv
% (the data and layout of figure_multisplit() in make_r4_revised_figures.py);
% do not edit by hand. Width 0.96\linewidth (figure* float); on a line narrower
% than the paper's the same layout is scaled to the line. Needs pgfplots
% (compat=1.18) with the fillbetween library and the preamble's oi* colors.
"""

MS_PREAMBLE = r"""\begin{tikzpicture}[font=\scriptsize]
% grays of make_r4_revised_figures.py, local to this picture
\definecolor{msInk}{HTML}{1B1B1B}%     mean lines and marks
\definecolor{msZero}{HTML}{8A99A2}%    zero / reference lines, leader lines
\definecolor{msRule}{HTML}{B8C2C8}%    dashed divider in (b)
\definecolor{msNoteA}{HTML}{3D4A52}%   key text in (a)
\definecolor{msNote}{HTML}{56636B}%    notes and legend keys
\definecolor{msBand}{HTML}{E4E9EC}%    +/-10% band in (c)
\definecolor{msCurve}{HTML}{2F3B42}%   curve in (d)
\definecolor{msGrid}{HTML}{CBD3D8}%    grid (drawn at 55% opacity)
\pgfmathsetlengthmacro{\msW}{(\msFrame-<<fixed>>pt)/2}% width of each axis box
\pgfmathsetlengthmacro{\msHtop}{<<rtop>>*\msW}% heights of the axis boxes
\pgfmathsetlengthmacro{\msHbottom}{<<rbottom>>*\msW}
\pgfmathsetlengthmacro{\msX}{\msW+<<gap>>pt}% left edge of (b) and (d)
\pgfmathsetlengthmacro{\msY}{\msHbottom+<<vgap>>pt}% bottom edge of (a) and (b)
\pgfplotsset{
  msPanel/.style={
    scale only axis, width=\msW,
    axis x line*=bottom, axis y line*=left,
    axis line style={line width=0.7pt},
    tick align=outside, major tick length=3pt,
    every tick/.append style={line width=0.7pt, black},
    minor tick length=0pt,
    tick label style={font=\scriptsize},
    label style={font=\footnotesize},
    every axis title/.style={at={(0,1)}, anchor=south west, inner sep=0pt,
      yshift=4.5pt, font=\footnotesize\bfseries},
    major grid style={line width=0.5pt, draw=msGrid, opacity=0.55},
    legend style={draw=none, fill=none, font=\scriptsize, inner sep=0pt,
      nodes={inner sep=1.1pt, text depth=0.15em}, row sep=-0.6pt, column sep=1.6pt,
      at={(1,1)}, xshift=-1.3pt, yshift=-1.3pt, anchor=north east},
    legend cell align=left,
  },
  msBars/.style={% plain error bars: no end-mark node per bar (same output, faster)
    error bars/draw error bar/.code 2 args={\draw ##1 -- ##2;}},
  msLogK/.style={% used with xmode=log, which pgfplots wants in the axis options
    xmin=2.7, xmax=112, xtick={3,5,10,20,50,100},
    xticklabels={3,5,10,20,50,100}, xminorticks=false, xmajorgrids, ymajorgrids,
    xlabel={Partitions used, $K$},
  },
}
"""


def multisplit_tex(fig, data: dict) -> str:
    head, tail = wrapper(r"0.96\linewidth", f"{MS_FRAME:.2f}", "ms")
    out = [MS_HEADER, head, fill(
        MS_PREAMBLE, fixed=num(MS_LEFT + MS_GAP + MS_RIGHT, 6), gap=num(MS_GAP, 6),
        vgap=num(MS_VGAP, 6), rtop=f"{MS_HEIGHT_TOP / MS_AXIS:.5f}",
        rbottom=f"{MS_HEIGHT_BOTTOM / MS_AXIS:.5f}")]
    names = {f: color_name(fig, f) for f in fig.FAMILIES}
    counts = data["counts"]

    # ---------------------------------------------------------------- (a)
    out.append("% ---- (a) effect on each partition: sorted gain (point) and its 95%\n"
               "%      within-partition grouped-bootstrap interval (bar), every partition;\n"
               "%      up/down = interval ends relative to the gain (pgfplots error bars)\n")
    for p in data["panel_a"]:
        rows = [(num(x), num(d), num(lo), num(hi), num(hi - d), num(d - lo), str(seed))
                for x, d, lo, hi, seed in zip(p["x"], p["delta"], p["low"], p["high"], p["seed"])]
        out.append(f"% {p['family']}: K = {p['k']} partitions, mean gain {p['mean']:.6f}\n")
        out.append(table(rf"\msA{TAGS[p['family']]}", ["x", "gain", "low", "high", "up", "down", "seed"], rows))
    ymin, ymax = data["a_ylim"]
    ticklabels = ",".join(
        "{" + rf"\textcolor{{{names[f]}}}{{{tex_family(f)}}}\\\textcolor{{{names[f]}}}{{$K = {counts[f]}$}}" + "}"
        for f in fig.FAMILIES
    )
    out.append(fill(r"""\begin{axis}[msPanel, name=msA, at={(0pt,\msY)}, anchor=south west,
  height=\msHtop, xmin=-0.55, xmax=<<xmax>>, ymin=<<ymin>>, ymax=<<ymax>>,
  xtick={0,1,2}, xticklabels={<<ticklabels>>},
  xticklabel style={align=center, font=\scriptsize\bfseries\boldmath},
  x tick style={draw=none}, xtick align=inside,
  ytick={0,0.025,0.05,0.075,0.1,0.125,0.15}, scaled y ticks=false,
  yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=3},
  ymajorgrids,
  ylabel={$\Delta$ linked-rewrite F1 (aug.\ $-$ clean)},
  title={(a) Effect on each partition},
]
\draw[msZero, line width=0.8pt] (axis cs:-0.55,0) -- (axis cs:<<xmax>>,0);
""", xmax=num(len(fig.FAMILIES) - 0.45), ymin=num(ymin), ymax=num(ymax), ticklabels=ticklabels))
    for p in data["panel_a"]:  # bars below points (zorder 2 < 3)
        lw, alpha = (0.45, 0.38) if p["dense"] else (0.85, 0.55)
        out.append(
            rf"\addplot[draw=none, mark=none, forget plot, msBars, error bars/.cd, y dir=both, y explicit,"
            rf" error mark=none, error bar style={{line width={lw}pt, draw={names[p['family']]},"
            rf" opacity={alpha}}}] table[x=x, y=gain, y error plus=up, y error minus=down]"
            rf" {{\msA{TAGS[p['family']]}}};" + "\n")
    for p in data["panel_a"]:
        # matplotlib scatter: s = 3 (dense) or 9 pt^2, no edge -> diameter sqrt(s);
        # pgfplots strokes every mark, so a 0.2pt same-color edge comes off the radius
        radius = np.sqrt(3.0 if p["dense"] else 9.0) / 2 - 0.1
        out.append(
            rf"\addplot[only marks, mark=*, mark size={radius:.3f}pt, color={names[p['family']]},"
            rf" mark options={{line width=0.2pt}}, forget plot] table[x=x, y=gain] {{\msA{TAGS[p['family']]}}};" + "\n")
    for p in data["panel_a"]:  # mean over partitions, with a white halo (zorder 4)
        out.append(
            rf"\draw[msInk, line width=1.1pt, preaction={{draw=white, line width=2.4pt}}]"
            rf" (axis cs:{num(p['index'] - 0.43)},{num(p['mean'])}) -- (axis cs:{num(p['index'] + 0.43)},{num(p['mean'])});"
            + "\n")
    out.append(fill(r"""\node[anchor=north west, align=left, inner sep=0pt, text=msNoteA] at (rel axis cs:0.02,0.98)
  {points: partitions, sorted by effect\\
   bars: 95\% grouped-bootstrap interval within\\
   the partition (<<groups>> test groups per partition)\\
   black line: mean over partitions};
\end{axis}
""", groups=fig.TEST_GROUPS))

    # ---------------------------------------------------------------- (b)
    out.append("% ---- (b) 95% intervals by scope; rows top to bottom, y = -(row + family offset)\n")
    for family in fig.FAMILIES:
        rows = [(str(e["row"]), num(-e["y"]), num(e["low"]), num(e["high"]), num(e["delta"]),
                 num(e["high"] - e["delta"]), num(e["delta"] - e["low"]))
                for e in data["panel_b"] if e["family"] == family]
        out.append(table(rf"\msB{TAGS[family]}", ["row", "y", "low", "high", "mean", "up", "down"], rows))
    xmin, xmax = data["b_xlim"]
    scope_labels = []
    for _, label in fig.SCOPES:
        text = (label.replace("\n", r"\\").replace("–", "--").replace("paired t", r"paired $t$")
                .replace("Corrected t", r"Corrected $t$").replace("interval (t)", r"interval ($t$)")
                .replace("†", r"\textsuperscript{\dag}"))
        scope_labels.append("{" + text + "}")
    cov = data["order_cov"]
    footnote = (r"\textsuperscript{\dag}nominal coverage "
                + ", ".join(f"{cov[f]} at $K = {counts[f]}$" for f in ("Word TF-IDF", "DistilBERT")))
    out.append(fill(r"""\begin{axis}[msPanel, name=msB, at={(\msX,\msY)}, anchor=south west,
  height=\msHtop, xmin=<<xmin>>, xmax=<<xmax>>, ymin=<<ymin>>, ymax=0.62,
  xtick={0,0.02,0.04,0.06,0.08,0.1,0.12}, scaled x ticks=false,
  xticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=2},
  ytick={<<yticks>>}, yticklabels={<<ylabels>>},
  yticklabel style={align=right}, y tick style={draw=none}, ytick align=inside,
  xmajorgrids,
  xlabel={$\Delta$ linked-rewrite F1 (aug.\ $-$ clean)},
  title={(b) 95\% intervals by scope},
]
\draw[msZero, line width=0.8pt] (axis cs:0,<<ymin>>) -- (axis cs:0,0.62);
\draw[msRule, line width=0.7pt, dash pattern=on 2.59pt off 1.12pt] (axis cs:<<xmin>>,-2.5) -- (axis cs:<<xmax>>,-2.5);
""", xmin=num(xmin), xmax=num(xmax), ymin=num(-(len(fig.SCOPES) - 0.02)),
        yticks=",".join(str(-row) for row in range(len(fig.SCOPES))), ylabels=",".join(scope_labels)))
    for family in fig.FAMILIES:  # interval (zorder 3) and mean mark (zorder 4)
        out.append(
            rf"\addplot[only marks, mark=|, mark size=2.75pt, mark options={{msInk, line width=1.1pt}},"
            rf" forget plot, msBars, error bars/.cd, x dir=both, x explicit, error mark=none,"
            rf" error bar style={{line width=2pt, draw={names[family]}}}]"
            rf" table[x=mean, y=y, x error plus=up, x error minus=down] {{\msB{TAGS[family]}}};" + "\n")
    out.append(fill(r"""\node[anchor=south east, inner sep=0pt, text=msNote, font=\scriptsize\itshape]
  at ({rel axis cs:0.985,0} |- {axis cs:0,-2.43}) {$\uparrow$ mean effect};
\node[anchor=north east, inner sep=0pt, text=msNote, font=\scriptsize\itshape]
  at ({rel axis cs:0.985,0} |- {axis cs:0,-2.57}) {$\downarrow$ one further partition};
\node[anchor=south east, inner sep=0pt, text=msNote]
  at ({rel axis cs:0.985,0} |- {axis cs:0,<<foot>>}) {<<footnote>>};
""", foot=num(-(len(fig.SCOPES) - 0.08)), footnote=footnote))
    for family in fig.FAMILIES:
        out.append(
            rf"\addlegendimage{{legend image code/.code={{\draw[{names[family]}, line width=2pt]"
            rf" (0pt,0pt) -- (9.1pt,0pt);}}}}\addlegendentry{{{tex_family(family)}}}" + "\n")
    out.append(r"\addlegendimage{legend image code/.code={\draw[msInk, line width=1.1pt]"
               r" (4.55pt,-2.75pt) -- (4.55pt,2.75pt);}}\addlegendentry{mean effect}" + "\n")
    out.append(r"\end{axis}" + "\n")

    # ---------------------------------------------------------------- (c)
    out.append("% ---- (c) SD of the gains from K partitions / SD over all partitions:\n"
               "%      median and 5th-95th percentile over random orderings (closed at\n"
               "%      K = all partitions, where every ordering gives 1), and the run order\n")
    for c in data["panel_c"]:
        tag = TAGS[c["family"]]
        out.append(f"% {c['family']}: SD over all {c['k_full']} partitions = {c['sd_full']:.6g}\n")
        out.append(table(rf"\msC{tag}", ["k", "p05", "median", "p95"],
                         [(str(int(k)), num(lo), num(me), num(up))
                          for k, lo, me, up in zip(c["ks"], c["lower"], c["median"], c["upper"])]))
        out.append(table(rf"\msR{tag}", ["k", "ratio"],
                         [(str(int(k)), num(r)) for k, r in zip(c["run_k"], c["run_ratio"])]))
    out.append(r"""\begin{axis}[xmode=log, msPanel, msLogK, name=msC, at={(0pt,0pt)}, anchor=south west, set layers,
  height=\msHbottom, ymin=0, ymax=2.05, ytick={0,0.5,1,1.5,2},
  yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1},
  ylabel={SD($K$)\,/\,SD(all partitions)},
  title={(c) Stabilization of the between-partition SD},
]
\begin{pgfonlayer}{axis background}% the +/-10% band lies below the grid, as in the PNG
  \fill[msBand] (axis cs:2.7,0.9) rectangle (axis cs:112,1.1);
\end{pgfonlayer}
\draw[msZero, line width=0.8pt] (axis cs:2.7,1) -- (axis cs:112,1);
""")
    for c in data["panel_c"]:  # band and its edges (zorder 2)
        tag, name = TAGS[c["family"]], names[c["family"]]
        out.append(
            rf"\addplot[name path=msC{tag}lo, draw=none, forget plot] table[x=k, y=p05] {{\msC{tag}}};" + "\n"
            rf"\addplot[name path=msC{tag}hi, draw=none, forget plot] table[x=k, y=p95] {{\msC{tag}}};" + "\n"
            rf"\addplot[fill={name}, fill opacity=0.09, draw=none, forget plot] fill between[of=msC{tag}hi and msC{tag}lo];" + "\n"
            rf"\addplot[{name}, line width=0.6pt, opacity=0.55, forget plot] table[x=k, y=p05] {{\msC{tag}}};" + "\n"
            rf"\addplot[{name}, line width=0.6pt, opacity=0.55, forget plot] table[x=k, y=p95] {{\msC{tag}}};" + "\n")
    for c in data["panel_c"]:  # medians (zorder 3)
        tag, name = TAGS[c["family"]], names[c["family"]]
        out.append(rf"\addplot[{name}, line width=1.4pt, forget plot] table[x=k, y=median] {{\msC{tag}}};" + "\n")
    for c in data["panel_c"]:  # run order (zorder 4): ':' dashes of a 1.1pt line, 2.3pt marks
        tag, name = TAGS[c["family"]], names[c["family"]]
        out.append(
            rf"\addplot[{name}, line width=1.1pt, dash pattern=on 1.1pt off 1.815pt, mark=*, mark size=1.15pt,"
            rf" mark options={{solid, line width=1pt}}, forget plot] table[x=k, y=ratio] {{\msR{tag}}};" + "\n")
    out.append(r"""\node[anchor=north east, inner sep=0pt, text=msNote] at (axis cs:105,0.885) {$\pm$10\%};
\addlegendimage{legend image code/.code={\draw[msNote, line width=1.4pt] (0pt,0pt) -- (11.7pt,0pt);}}
\addlegendentry{median, 2,000 random orders}
\addlegendimage{legend image code/.code={\fill[msNote, fill opacity=0.16, draw=none] (0pt,-2.6pt) rectangle (11.7pt,2.6pt);}}
\addlegendentry{5th--95th percentile}
\addlegendimage{legend image code/.code={\draw[msNote, line width=1.1pt, dash pattern=on 1.1pt off 1.815pt]
  (0pt,0pt) -- (11.7pt,0pt); \fill[msNote] (5.85pt,0pt) circle[radius=1.65pt];}}
\addlegendentry{order as run}
\end{axis}
""")

    # ---------------------------------------------------------------- (d)
    series = data["series"]
    out.append("% ---- (d) corrected resampled-t half-width / its floor (same for every family)\n")
    out.append(table(r"\msD", ["k", "ratio"], [(str(int(k)), num(v)) for k, v in series.items()]))
    out.append(r"""\begin{axis}[xmode=log, msPanel, msLogK, name=msD, at={(\msX,0pt)}, anchor=south west,
  height=\msHbottom, ymin=0.75, ymax=3.75, ytick={1,1.5,2,2.5,3,3.5},
  yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1},
  ylabel={Corrected-$t$ half-width\,/\,floor},
  title={(d) Half-width relative to its floor},
]
\draw[msZero, line width=0.8pt] (axis cs:2.7,1) -- (axis cs:112,1);
\addplot[msCurve, line width=1.4pt, mark=*, mark size=1.5pt, mark options={line width=1pt}, forget plot]
  table[x=k, y=ratio] {\msD};
""")
    for k_mark in (3, 5, 10, 50):  # verbatim positions: upper right of the point
        value = float(series.loc[k_mark])
        out.append(rf"\node[anchor=south west, inner sep=0pt, text=msCurve] at (axis cs:{num(k_mark * 1.07)},{num(value + 0.05)}) {{{value:.2f}}};" + "\n")
    # The PNG hangs this note from y = 0.94; the radical and subscripts of the TeX
    # formula would then touch the axis line, so the note is centered in the strip
    # between the reference line (1.0) and the axis (0.75). It is set after the
    # axis (unclipped), so a font wider than Times cannot cut it off.
    out.append(r"\coordinate (msDfloor) at (axis cs:3,0.875);" + "\n")
    for k_mark, first, color_key, anchor, align in (
        (20, "DistilBERT", "DistilBERT", "south", "center"),
        (100, "lightweight", None, "south east", "right"),
    ):  # verbatim: text 0.95 above the point, leader from the text's lower edge
        value = float(series.loc[k_mark])
        text_color = names[color_key] if color_key else "msCurve"
        out.append(
            rf"\node[anchor={anchor}, align={align}, inner sep=0pt, text={text_color}] (msDnote{k_mark})"
            rf" at (axis cs:{k_mark},{num(value + 0.95)}) {{{first}\\$K = {k_mark}$: {value:.2f}}};" + "\n"
            rf"\draw[msZero, line width=0.6pt, shorten <=1pt, shorten >=4pt] (msDnote{k_mark}.{anchor})"
            rf" -- (axis cs:{k_mark},{num(value)});" + "\n")
    out.append(r"""\node[anchor=north east, inner sep=0pt, text=msNote, font=\scriptsize\itshape]
  at (rel axis cs:0.98,0.97) {same for all three families};
\end{axis}
\node[anchor=west, inner sep=0pt, text=msNote] at (msDfloor)
  {floor: $1.96\times\mathrm{SD}\times\sqrt{n_{\mathrm{test}}/n_{\mathrm{train}}}$, $n_{\mathrm{test}}/n_{\mathrm{train}} = 0.215$};
\end{tikzpicture}%
""")
    out.append(tail)
    return "".join(out)


# --------------------------------------------------------------------------
# Figure 6: data (verbatim from figure_tradeoff)
# --------------------------------------------------------------------------
LABEL_SPEC = {
    # family: (anchor x, anchor y, ha, va) in data units -- verbatim from figure_tradeoff
    "Word TF-IDF": (0.9335, 0.0605, "center", "bottom"),
    "MiniLM": (0.834, 0.0475, "center", "top"),
    "DistilBERT": (0.8355, 0.113, "right", "bottom"),
}
TR_XLIM, TR_YLIM = (0.790, 0.962), (0.0, 0.172)  # verbatim


def tradeoff_data(fig) -> list[dict]:
    light, bert = fig.lightweight_metrics(), fig.distilbert_metrics()
    out = []
    for family in fig.FAMILIES:
        points = {}
        for training in fig.TRAININGS:
            f1 = fig.partition_values(light, bert, family, training, "external_openai", "f1")
            fpr = fig.partition_values(light, bert, family, training, "spam_hard_negative", "fpr_at_0_5")
            if not f1.index.equals(fpr.index):
                raise ValueError(f"L2 and spam partitions differ for {family} {training}")
            points[training] = (fig.summarize(f1), fig.summarize(fpr))
        k = points["clean"][0]["k"]
        if points["augmented"][0]["k"] != k:
            raise ValueError(f"clean and augmented partition counts differ for {family}")
        out.append({"family": family, "k": k, "points": points})
    return out


# --------------------------------------------------------------------------
# Figure 6: TikZ
# --------------------------------------------------------------------------
# Label space (pt, paper fonts): y tick labels and y label left of the axis box,
# the last x tick label right of it; the axis box takes the rest of the line.
# Its height (paper: 189.2pt) gives the PNG's printed height, 218.3pt.
TR_FRAME = 252.0  # the paper's column width
TR_LEFT, TR_RIGHT, TR_HEIGHT = 36.11, 7.03, 189.2
TR_AXIS = TR_FRAME - TR_LEFT - TR_RIGHT

TR_TEMPLATE = r"""% tikz_tradeoff.tex -- Figure 6 (fig:tradeoff), drawn with pgfplots.
% Written by artifact/eswa_study/revision/tikz_inference.py from the per-partition
% metrics files (results/revision_r3_multisplit_metrics.csv,
% revision_r4_multisplit100_metrics.csv and the three DistilBERT metrics files;
% the data and layout of figure_tradeoff() in make_r4_revised_figures.py);
% do not edit by hand. Width \linewidth (one-column figure float); on a line
% narrower than the paper's column the same layout is scaled to the line.
% Needs pgfplots (compat=1.18), the arrows.meta library and the preamble's oi* colors.
<<head>>\begin{tikzpicture}[font=\scriptsize]
% grays of make_r4_revised_figures.py, local to this picture
\definecolor{trGrid}{HTML}{CBD3D8}%    dotted grid
\definecolor{trRegion}{HTML}{7E9AAB}%  "preferable" region
\definecolor{trRegionText}{HTML}{4F6675}
\definecolor{trKey}{HTML}{52616B}%     legend keys
% The PNG shades the region with alpha 0.30 (1 - d)^0.85, d the elliptical
% distance from the lower-right corner (semi-axes 0.62 and 0.50 of the axis box);
% these stops are that profile over white at d = 0, 0.25, 0.5, 0.75, 0.9 and 1.
\pgfdeclareradialshading{trPreferable}{\pgfpoint{0bp}{0bp}}{<<stops>>}
\pgfmathsetlengthmacro{\trW}{\trFrame-<<fixed>>pt}% width of the axis box
\pgfmathsetlengthmacro{\trH}{<<ratio>>*\trW}% height of the axis box
% means and SDs across partitions (row 1 clean-only, row 2 augmented)
<<tables>>\begin{axis}[name=tr, scale only axis, width=\trW, height=\trH, set layers,
  axis x line*=bottom, axis y line*=left, axis line style={line width=0.7pt},
  tick align=outside, major tick length=3pt, every tick/.append style={line width=0.7pt, black},
  tick label style={font=\scriptsize}, label style={font=\footnotesize},
  xmin=<<xmin>>, xmax=<<xmax>>, ymin=<<ymin>>, ymax=<<ymax>>,
  xtick={0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96}, ytick={0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16},
  scaled ticks=false,
  xticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=2},
  yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=2},
  grid=major, major grid style={line width=0.5pt, draw=trGrid, dash pattern=on 0.5pt off 0.825pt},
  xlabel={Generator-disjoint (L2) F1 at threshold 0.5},
  ylabel={Spam hard-negative FPR at threshold 0.5},
  legend style={name=trLegend, draw=none, fill=none, font=\scriptsize, inner sep=0pt,
    nodes={inner sep=1.1pt, text depth=0.15em}, row sep=-0.2pt,
    at={(1,1)}, xshift=-2pt, yshift=-11pt, anchor=north east},
  legend cell align=left,
]
% "preferable" region: a guide, not data; below the grid as in the PNG
\begin{pgfonlayer}{axis background}
  \begin{scope}
    \clip (rel axis cs:0,0) rectangle (rel axis cs:1,1);
    \shade[shading=trPreferable] (rel axis cs:1,0) ellipse [x radius=0.62*\trW, y radius=0.5*\trH];
  \end{scope}
\end{pgfonlayer}
\node[anchor=south east, align=right, inner sep=0pt, text=trRegionText, font=\scriptsize\itshape]
  at (rel axis cs:0.975,0.03) {preferable: higher L2 F1,\\fewer spam false alarms};
<<marks>>\addlegendimage{legend image code/.code={\draw[trKey, line width=1.3pt, fill=white] (5.2pt,0pt) circle[radius=2.6pt];}}
\addlegendentry{clean-only (mean)}
\addlegendimage{legend image code/.code={\draw[trKey, line width=1pt, fill=trKey] (5.2pt,0pt) circle[radius=2.6pt];}}
\addlegendentry{augmented (mean)}
\addlegendimage{legend image code/.code={\draw[trKey, line width=1.4pt, -{Triangle[length=4.6pt, width=4.6pt]}] (0pt,0pt) -- (12.6pt,0pt);}}
\addlegendentry{arrow: clean-only $\rightarrow$ augmented}
\addlegendimage{legend image code/.code={\draw[trKey, line width=0.8pt, opacity=0.75] (0pt,0pt) -- (10.4pt,0pt);}}
\addlegendentry{whiskers: $\pm1$ SD}
\end{axis}
\node[anchor=south west, inner sep=1.1pt] at (trLegend.north west) {$K$ = refitted partitions};
\end{tikzpicture}%
<<tail>>"""


def preferable_stops() -> str:
    """Radial-shading stops reproducing the PNG's alpha profile over white."""
    stops = []
    for d in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        alpha = 0.30 * np.clip(1 - d, 0, 1) ** 0.85  # verbatim formula of preferable_region()
        stops.append(f"color({25 * d:g}bp)=(trRegion!{100 * alpha:.2f}!white)")
    stops.append("color(50bp)=(white)")
    return "; ".join(stops)


def tradeoff_tex(fig, data: list[dict]) -> str:
    names = {f: color_name(fig, f) for f in fig.FAMILIES}
    tables = []
    for entry in data:
        rows = []
        for training in fig.TRAININGS:
            f1s, fprs = entry["points"][training]
            rows.append((training, num(f1s["mean"]), num(f1s["sd"]), num(fprs["mean"]), num(fprs["sd"])))
        tables.append(f"% {entry['family']}: K = {entry['k']} partitions\n")
        tables.append(table(rf"\tr{TAGS[entry['family']]}", ["training", "f1", "f1sd", "fpr", "fprsd"], rows))
    marks = ["% whiskers (zorder 2), arrows (3), means (4), labels (5), each over all families\n"]
    for entry in data:  # +/-1 SD whiskers with 3.6pt caps, alpha 0.75
        c, t = names[entry["family"]], rf"\tr{TAGS[entry['family']]}"
        marks.append(
            rf"\addplot[draw=none, forget plot, error bars/.cd, x dir=both, y dir=both, x explicit, y explicit,"
            rf" error mark=-, error bar style={{{c}, line width=0.8pt, opacity=0.75}},"
            rf" error mark options={{rotate=90, {c}, line width=0.8pt, mark size=1.8pt, opacity=0.75}}]"
            rf" table[x=f1, y=fpr, x error=f1sd, y error=fprsd] {{{t}}};" + "\n")
    for entry in data:  # clean-only -> augmented, ends 3.6pt / 3.8pt short of the centers
        c, t = names[entry["family"]], rf"\tr{TAGS[entry['family']]}"
        marks.append(
            rf"\addplot[{c}, line width=1.5pt, shorten <=3.6pt, shorten >=3.8pt,"
            rf" -{{Triangle[length=5.2pt, width=5pt]}}, forget plot] table[x=f1, y=fpr] {{{t}}};" + "\n")
    for entry in data:  # means: open (clean-only, row 1) and filled (augmented, row 2)
        c, t = names[entry["family"]], rf"\tr{TAGS[entry['family']]}"
        for skip, face in (("{1}{2}", "white"), ("{0}{1}", c)):
            marks.append(
                rf"\addplot[only marks, mark=*, mark size=2.8pt, mark options={{draw={c}, fill={face},"
                rf" line width=1.4pt}}, skip coords between index={skip}, forget plot] table[x=f1, y=fpr] {{{t}}};" + "\n")
    anchors = {("center", "bottom"): "south", ("center", "top"): "north", ("right", "bottom"): "south east"}
    for entry in data:
        x, y, ha, va = LABEL_SPEC[entry["family"]]
        align = "right" if ha == "right" else "center"
        marks.append(
            rf"\node[anchor={anchors[(ha, va)]}, align={align}, inner sep=0pt, text={names[entry['family']]},"
            rf" font=\scriptsize\bfseries\boldmath] at (axis cs:{num(x)},{num(y)})"
            rf" {{{tex_family(entry['family'])}\\$K = {entry['k']}$}};" + "\n")
    head, tail = wrapper(r"\linewidth", f"{TR_FRAME:.2f}", "tr")
    return fill(TR_TEMPLATE, head=head, tail=tail, stops=preferable_stops(),
                fixed=num(TR_LEFT + TR_RIGHT, 6), ratio=f"{TR_HEIGHT / TR_AXIS:.5f}",
                tables="".join(tables), xmin=num(TR_XLIM[0]), xmax=num(TR_XLIM[1]),
                ymin=num(TR_YLIM[0]), ymax=num(TR_YLIM[1]), marks="".join(marks))


# --------------------------------------------------------------------------
def summary(ms: dict, tr: list[dict]) -> list[str]:
    lines = []
    for p in ms["panel_a"]:
        lines.append(f"Fig 3a {p['family']}: K={p['k']}, mean delta={p['mean']:.4f}, "
                     f"{p['reach_zero']} partition intervals reach zero")
    lines.append(f"Fig 3a y range {ms['a_ylim'][0]:.5f}..{ms['a_ylim'][1]:.5f}; "
                 f"3b x range {ms['b_xlim'][0]:.5f}..{ms['b_xlim'][1]:.5f}")
    for c in ms["panel_c"]:
        five = c["run_ratio"][list(c["run_k"]).index(5)]
        lines.append(f"Fig 3c {c['family']}: SD(full)={c['sd_full']:.5f} at K={c['k_full']}; "
                     f"as-run SD(K=5)/SD(full)={five:.2f}")
    series = ms["series"]
    lines.append("Fig 3d ratio: " + ", ".join(f"K={k}: {series.loc[k]:.2f}" for k in (3, 5, 10, 20, 50, 100)))
    for entry in tr:
        (cf, cp), (af, ap) = entry["points"]["clean"], entry["points"]["augmented"]
        lines.append(f"Fig 6 {entry['family']} (K={entry['k']}): clean F1 {cf['mean']:.3f}+/-{cf['sd']:.3f}, "
                     f"FPR {cp['mean']:.3f}+/-{cp['sd']:.3f}; augmented F1 {af['mean']:.3f}+/-{af['sd']:.3f}, "
                     f"FPR {ap['mean']:.3f}+/-{ap['sd']:.3f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--copy-to", type=pathlib.Path, default=default_bundle(),
                        help="also copy the .tex files here (default: the manuscript bundle)")
    args = parser.parse_args()
    fig = load_figure_module()
    ms, tr = multisplit_data(fig), tradeoff_data(fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (("multisplit_inference", multisplit_tex(fig, ms)), ("tradeoff", tradeoff_tex(fig, tr))):
        path = OUT_DIR / f"tikz_{name}.tex"
        path.write_text(text)
        written.append(path)
        if args.copy_to:
            args.copy_to.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, args.copy_to / path.name)
    for line in summary(ms, tr):
        print(line)
    print("wrote " + ", ".join(p.name for p in written) + f" to {OUT_DIR}"
          + (f" and {args.copy_to}" if args.copy_to else ""))


if __name__ == "__main__":
    main()
