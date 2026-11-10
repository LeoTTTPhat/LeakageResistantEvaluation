#!/usr/bin/env python3
"""Figures for the round-4 analyses, in the style of make_figures.py.

* ``pipeline_signature``    -- does the L2 gain survive removal of the pipeline
                               signature? (rev_s_pipeline_signature.py)
* ``l3_explained``          -- proximity to the training data and topic coverage
                               (rev_u3_proximity.py, rev_t_human_error_analysis.py)
* ``era_matched``           -- recall on era-matched real phishing by collection
                               year (rev_u_era_format.py; DistilBERT from
                               rev_u2_era_format_distilbert.py when its output exists)
* ``deployment_robustness`` -- prevalence estimation, held-out rule validation
                               and score-distribution sensitivity
                               (rev_w_deployment_robustness.py)

Every number is read from results/; nothing is refitted. The script prints the
values it plots so that they can be checked against the text.

Usage (from artifact/eswa_study/revision/):
    python make_r4_new_figures.py                    # all four figures
    python make_r4_new_figures.py --only era_matched # selected figures
    python make_r4_new_figures.py --copy-to DIR      # also copy the PNGs to DIR
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap, PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from rev_common import FIGURES, RESULTS, STUDY  # noqa: E402

sys.path.insert(0, str(STUDY))
from make_figures import COLORS, GRID, configure_style, finish  # noqa: E402

WORD = COLORS["Word TF--IDF"]
MINILM = COLORS["MiniLM"]
BERT = COLORS["DistilBERT"]
FAMILY = {
    "word": ("Word TF-IDF", WORD),
    "minilm": ("MiniLM", MINILM),
    "distilbert": ("DistilBERT", BERT),
}
MUTED = "#6B7B85"
INK = "#33414A"
ZERO = "#9AA5AB"

# Stream colours follow the round-3 deployment figure (threshold_cost.png):
# L1 blue, L2 green, L3 vermilion; the era-matched contemporary stream is new.
STREAMS = [
    ("L1 primary", "L1 primary", "#0072B2"),
    ("L2 generator-disjoint", "L2 generator-disjoint", "#009E73"),
    ("L3 human-authored", "L3 human-authored (2000s)", "#D55E00"),
    ("Contemporary (Nazario + real legitimate)",
     "Contemporary (Nazario 2015–2025 + real legitimate)", "#CC79A7"),
]
STREAM_COLOR = {key: colour for key, _, colour in STREAMS}
STREAM_SHORT = {
    "L1 primary": "L1",
    "L2 generator-disjoint": "L2",
    "L3 human-authored": "L3",
    "Contemporary (Nazario + real legitimate)": "Cont.",
}
NAZARIO_EARLY = "#882E72"
NAZARIO_LATE = "#CC79A7"
MINUS = "−"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", MINUS)


def add_axes(fig, left, bottom, width, height, **kwargs):
    """Axes placed in inches from the lower-left corner of the figure."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kwargs)


def fig_text(fig, x, y, text, **kwargs):
    """Figure text placed in inches."""
    w, h = fig.get_size_inches()
    return fig.text(x / w, y / h, text, **kwargs)


def panel_title(fig, x, y, letter, text):
    fig_text(fig, x, y, f"({letter}) {text}", fontsize=9, fontweight="bold",
             ha="left", va="baseline")


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)


# ==========================================================================
# 1. pipeline signature
# ==========================================================================
def figure_pipeline_signature(output_dir: Path) -> None:
    vocab = pd.read_csv(RESULTS / "revision_r4_pipeline_vocab.csv")
    erasure = pd.read_csv(RESULTS / "revision_r4_pipeline_erasure.csv")
    dann = pd.read_csv(RESULTS / "revision_r4_pipeline_dann.csv")

    # Origin decodability of the erased representation: prefer the grouped,
    # cross-validated probe on the erased held-out data (round-4 rerun); fall
    # back to the probe refitted on the eraser's own fitting data.
    cv_column = "origin_auc_cv_on_erased_heldout"
    if cv_column in erasure.columns and erasure[cv_column].notna().all():
        erasure_auc_column = cv_column
    else:
        erasure_auc_column = "origin_auc_after_erasure"
        print(f"  WARNING: {cv_column} missing; falling back to origin_auc_after_erasure")
    print(f"  erasure origin AUC column: {erasure_auc_column}")

    fig = plt.figure(figsize=(7.1, 3.05))

    # ---- (a) vocabulary neutralization ------------------------------------
    ax = add_axes(fig, 0.58, 0.80, 2.50, 1.62)
    ks = np.sort(vocab["removed_ngrams"].unique())
    by_k = vocab.groupby("removed_ngrams")
    n_partitions = int(by_k.size().min())

    def quantile(column, level):
        return by_k[column].quantile(level).reindex(ks).to_numpy()

    def forward(x):
        return np.sign(x) * np.sqrt(np.abs(x))

    def inverse(y):
        return np.sign(y) * np.square(y)

    right = ax.twinx()
    right.fill_between(ks, quantile("origin_auc_after_removal", 0.05),
                       quantile("origin_auc_after_removal", 0.95),
                       color=MUTED, alpha=0.18, lw=0, zorder=1)
    right.plot(ks, quantile("origin_auc_after_removal", 0.5), ls="--", marker="s",
               ms=3.0, lw=1.1, color=MUTED, zorder=2)
    right.set_ylim(0.70, 1.012)
    right.set_ylabel("Held-out origin-probe ROC-AUC", color=MUTED)
    right.tick_params(axis="y", colors=MUTED)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color(MUTED)
    ax.set_zorder(right.get_zorder() + 1)
    ax.patch.set_visible(False)

    for training, style, face, alpha, label in (
        ("clean", ":", "white", 0.11, "Clean-only"),
        ("augmented", "-", WORD, 0.20, "Augmented"),
    ):
        column = f"{training}_f1"
        ax.fill_between(ks, quantile(column, 0.05), quantile(column, 0.95),
                        color=WORD, alpha=alpha, lw=0, zorder=1)
        ax.plot(ks, quantile(column, 0.5), ls=style, marker="o", ms=3.4, lw=1.4,
                color=WORD, mfc=face, mec=WORD, mew=1.0, zorder=3, label=label)
    ax.set_xscale("function", functions=(forward, inverse))
    ax.set_xlim(-1500, 42500)
    ticks = [0, 2500, 10000, 20000, 30000, 40000]
    for axis in (ax, right):
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0", "2.5k", "10k", "20k", "30k", "40k"])
    ax.set_xlabel("Most corpus-specific n-grams removed ($k$)", labelpad=14)
    ax.set_ylabel("L2 F1 (word TF-IDF)")
    ax.set_ylim(0.85, 0.962)
    light_grid(ax)

    # row of paired differences under the tick labels
    delta = by_k["delta_f1"].mean().reindex(ks)
    positive = by_k["delta_f1"].apply(lambda s: int((s > 0).sum())).reindex(ks)
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for k in ticks:
        ax.text(k, -0.155, signed(delta[k]).replace("0.", ".", 1), transform=transform,
                ha="center", va="top", fontsize=6.0, color=WORD)
    ax.text(-0.035, -0.155, "mean ΔF1", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.0, color=WORD)

    vocabulary = float(vocab["union_vocabulary"].mean())
    top = ax.twiny()
    top.set_xscale("function", functions=(forward, inverse))
    top.set_xlim(ax.get_xlim())
    top.xaxis.set_major_locator(FixedLocator(ticks))
    top.xaxis.set_minor_locator(NullLocator())
    shares = [100 * k / vocabulary for k in ticks]
    top.set_xticklabels(["0%"] + [f"{s:.1f}%" if s < 10 else f"{s:.0f}%" for s in shares[1:]])
    top.tick_params(axis="x", labelsize=6.5, colors=MUTED, length=2.5, pad=1.5)
    top.spines["top"].set_visible(True)
    top.spines["top"].set_color(MUTED)
    top.set_xlabel(f"Share of the {vocabulary / 1000:.1f}k-term vocabulary removed",
                   fontsize=7, color=MUTED, labelpad=2.5)

    handles = [
        Line2D([], [], color=WORD, ls="-", marker="o", ms=3.4, lw=1.4, label="Augmented"),
        Line2D([], [], color=WORD, ls=":", marker="o", ms=3.4, mfc="white", lw=1.4,
               label="Clean-only"),
        Line2D([], [], color=MUTED, ls="--", marker="s", ms=3.0, lw=1.1,
               label="Origin probe (right axis)"),
        Patch(color=WORD, alpha=0.2, lw=0,
              label=f"5th–95th percentile ({n_partitions} partitions)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, handlelength=2.0,
              borderaxespad=0.3, labelspacing=0.3)
    panel_title(fig, 0.05, 2.90, "a", "Vocabulary neutralization (word TF-IDF)")

    print("  (a) k, origin AUC median [p5,p95], clean F1 median, aug F1 median, mean delta, #positive")
    for k in ks:
        block = vocab[vocab["removed_ngrams"] == k]
        print(f"      {k:6d} {block['origin_auc_after_removal'].median():.4f} "
              f"[{block['origin_auc_after_removal'].quantile(0.05):.4f},"
              f"{block['origin_auc_after_removal'].quantile(0.95):.4f}] "
              f"{block['clean_f1'].median():.4f} {block['augmented_f1'].median():.4f} "
              f"{delta[k]:+.4f} {positive[k]}/{n_partitions} share {100 * k / vocabulary:.1f}%")

    # ---- (b) linear erasure and domain-adversarial training (MiniLM) -----
    ax = add_axes(fig, 4.30, 0.80, 2.72, 1.62)
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
    on_axes = blended_transform_factory(ax.transData, ax.transAxes)
    for position, (label, values, auc, head, directions) in zip(positions, entries):
        ax.boxplot([values], positions=[position], widths=0.52, whis=(5, 95),
                   showfliers=False, patch_artist=True, zorder=2,
                   boxprops={"facecolor": MINILM + "2E", "edgecolor": MINILM, "lw": 0.9},
                   medianprops={"color": MINILM, "lw": 1.5},
                   whiskerprops={"color": MINILM, "lw": 0.9},
                   capprops={"color": MINILM, "lw": 0.9})
        order = np.argsort(np.argsort(values))
        jitter = np.linspace(-0.17, 0.17, len(values))[order]
        ax.scatter(position + jitter, values, s=7, color=MINILM, alpha=0.75,
                   linewidths=0, zorder=3)
        ax.scatter([position], [values.mean()], marker="D", s=13, facecolor="white",
                   edgecolor=INK, linewidths=0.8, zorder=4)
        ax.text(position, 0.865, f"{auc:.2f}", transform=on_axes, ha="center",
                va="bottom", fontsize=7, color=INK)
        if head is not None:
            ax.text(position, 0.795, f"(own head {head:.2f})", transform=on_axes,
                    ha="center", va="bottom", fontsize=5.8, color=MUTED)
        print(f"  (b) {label.replace(chr(10), ' '):24s} mean delta {values.mean():+.4f} "
              f"median {np.median(values):+.4f} p5 {np.percentile(values, 5):+.4f} "
              f"p95 {np.percentile(values, 95):+.4f} positive {int((values > 0).sum())}/{len(values)} "
              f"origin AUC {auc:.4f}" + (f" head {head:.4f}" if head is not None else "")
              + (f" directions {directions:.2f}" if directions is not None else ""))
    ax.text(0.5, 0.955, "held-out origin-probe ROC-AUC of the representation",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax.axhline(0.0, color=ZERO, lw=0.8, zorder=0)
    ax.axvline(2.725, ymax=0.80, color=GRID, lw=0.8, ls=":", zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([entry[0] for entry in entries], fontsize=7)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, 5.0)
    ax.set_ylim(-0.022, 0.275)
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax.set_ylabel("L2 ΔF1 (augmented − clean)")
    light_grid(ax)
    for (start, stop), text in (((0.0, 2.0), "linear concept erasure\n(MiniLM + LR)"),
                                ((3.45, 4.45), "domain-adversarial\n(MiniLM MLP)")):
        ax.plot([start - 0.25, stop + 0.25], [-0.215, -0.215], transform=on_axes,
                color=MUTED, lw=0.7, clip_on=False)
        ax.text((start + stop) / 2, -0.24, text, transform=on_axes, ha="center",
                va="top", fontsize=6.3, color=MUTED, linespacing=0.95)
    panel_title(fig, 3.78, 2.90, "b", "Linear erasure and DANN (MiniLM)")
    finish(fig, output_dir, "pipeline_signature")


# ==========================================================================
# 2. why L3 fails: proximity and topics
# ==========================================================================
PROXIMITY_CORPORA = [
    ("L2 generator-disjoint phishing", "L2 generator-disjoint", "#009E73", "o"),
    ("Nazario 2015-2022", "Nazario 2015–2022", NAZARIO_EARLY, "s"),
    ("Nazario 2023-2025", "Nazario 2023–2025", NAZARIO_LATE, "D"),
    ("L3 human-authored phishing (2000s)", "L3 human-authored, 2000s", "#D55E00", "^"),
]
TOPIC_SHARES = [
    ("share_PhishFuzzer phishing", "PhishFuzzer\nphishing"),
    ("share_L2 phishing", "L2\nphishing"),
    ("share_L3 human phishing", "L3 human\nphishing"),
    ("share_Nazario 2015-2022", "Nazario\n2015–22"),
    ("share_Nazario 2023-2025", "Nazario\n2023–25"),
]


def _topic_label(terms: str, wrap: int = 28) -> str:
    first = terms.split(", ")[:3]
    label = ", ".join(first)
    if len(label) > wrap:
        label = ", ".join(first[:2]) + ",\n" + first[2]
    return label


def figure_l3_explained(output_dir: Path) -> None:
    proximity = pd.read_csv(RESULTS / "revision_r4_proximity_recall.csv",
                            dtype={"quartile": str})
    topics = pd.read_csv(RESULTS / "revision_r4_l3_topics.csv")

    fig = plt.figure(figsize=(7.1, 3.85))
    top_of_plots = 2.80

    # ---- (a) recall by similarity quartile --------------------------------
    ax = add_axes(fig, 0.50, 0.66, 1.98, top_of_plots - 0.66)
    n_partitions = proximity["seed"].nunique()
    for corpus, label, colour, marker in PROXIMITY_CORPORA:
        block = proximity[proximity["corpus"] == corpus]
        quart = block[block["quartile"] != "all"].groupby("quartile")
        x = quart["similarity_median"].mean().sort_index()
        y = quart["recall_0.5"].mean().sort_index()
        low = y - quart["recall_0.5"].min().sort_index()
        high = quart["recall_0.5"].max().sort_index() - y
        overall = block[block["quartile"] == "all"]
        n_all = int(overall["n"].iloc[0])
        ax.errorbar(x, y, yerr=[low, high], color=colour, marker=marker, ms=4.0, lw=1.4,
                    capsize=1.8, elinewidth=0.8, capthick=0.8, zorder=3,
                    label=f"{label} ($n$ = {n_all:,})")
        print(f"  (a) {label:26s} Q1..Q4 sim {np.round(x.to_numpy(), 3)} recall "
              f"{np.round(y.to_numpy(), 3)} range-low {np.round(low.to_numpy(), 3)} "
              f"range-high {np.round(high.to_numpy(), 3)} n/quartile "
              f"{quart['n'].mean().round(0).to_numpy()}; all: sim "
              f"{overall['similarity_median'].mean():.3f} recall {overall['recall_0.5'].mean():.3f}"
              f"; sim spread across partitions <= "
              f"{(quart['similarity_median'].max() - quart['similarity_median'].min()).max():.3f}")
        if corpus.startswith("L3"):
            for quartile_label, xi, yi in zip(x.index, x, y):
                last = quartile_label == "4"
                ax.annotate(f"Q{quartile_label}", (xi, yi),
                            xytext=(6, -1) if last else (0, 5.5), textcoords="offset points",
                            ha="left" if last else "center", va="center" if last else "bottom",
                            fontsize=5.8, color=MUTED)
    ax.set_xlim(0.31, 0.855)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Max. cosine similarity to training set\n(MiniLM; median within quartile)",
                  fontsize=7.0)
    ax.set_ylabel("Recall at threshold 0.5")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.01), fontsize=6.2,
              handlelength=1.6, borderaxespad=0.0, labelspacing=0.3)
    panel_title(fig, 0.02, 3.68, "a", "Recall vs proximity")
    fig_text(fig, 0.02, 3.52, "augmented word TF-IDF, 5 partitions", fontsize=6.6,
             color=MUTED, ha="left", va="baseline")
    print(f"  (a) partitions: {n_partitions}")

    # ---- (b) topic shares and per-topic recall ------------------------------
    rows = len(topics)
    cell_h = 0.208
    share_w, recall_w = 0.43, 0.48
    bottom = top_of_plots - rows * cell_h
    left_shares = 3.76
    ax_s = add_axes(fig, left_shares, bottom, len(TOPIC_SHARES) * share_w, rows * cell_h)
    left_recall = left_shares + len(TOPIC_SHARES) * share_w + 0.07
    ax_r = add_axes(fig, left_recall, bottom, 2 * recall_w, rows * cell_h)
    ordered = topics.sort_values("share_L3 human phishing", ascending=False).reset_index(drop=True)
    labels = [_topic_label(terms) for terms in ordered["top_terms"]]
    shares = ordered[[column for column, _ in TOPIC_SHARES]].to_numpy()
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)
    ax_s.imshow(shares, cmap=plt.get_cmap("GnBu"), norm=norm, aspect="auto")
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            value = 100 * shares[i, j]
            text = "0" if value == 0 else (f"{value:.0f}" if value >= 9.95 else f"{value:.1f}")
            ax_s.text(j, i, text, ha="center", va="center", fontsize=6.0,
                      color="white" if norm(shares[i, j]) > 0.62 else INK)
    ax_s.set_xticks(range(len(TOPIC_SHARES)))
    ax_s.set_xticklabels([label for _, label in TOPIC_SHARES], fontsize=6.3, linespacing=0.95)
    ax_s.set_yticks(range(rows))
    ax_s.set_yticklabels(labels, fontsize=6.2, linespacing=0.9)

    recall = ordered[["recall_human_at_0.5", "recall_nazario_at_0.5"]].to_numpy()
    counts = ordered[["n_human", "n_nazario"]].to_numpy()
    recall_cmap = ListedColormap(plt.get_cmap("Oranges")(np.linspace(0.0, 0.85, 256)))
    ax_r.imshow(recall, cmap=recall_cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(recall.shape[0]):
        for j in range(recall.shape[1]):
            dark = recall[i, j] > 0.66
            ax_r.text(j, i - 0.15, f"{recall[i, j]:.2f}", ha="center", va="center",
                      fontsize=6.0, color="white" if dark else INK)
            ax_r.text(j, i + 0.27, f"n = {counts[i, j]:,}", ha="center", va="center",
                      fontsize=5.0, color="white" if dark else MUTED)
    ax_r.set_xticks([0, 1])
    ax_r.set_xticklabels(["L3 human\nphishing", "Nazario\n2015–25"], fontsize=6.3,
                         linespacing=0.95)
    ax_r.set_yticks([])
    for axis in (ax_s, ax_r):
        axis.xaxis.tick_top()
        axis.tick_params(axis="both", length=0, pad=2)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xticks(np.arange(-0.5, axis.get_xlim()[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        axis.grid(which="minor", color="white", lw=0.9)
        axis.tick_params(which="minor", length=0)
    header_y = top_of_plots + 0.36
    fig_text(fig, left_shares + len(TOPIC_SHARES) * share_w / 2, header_y,
             "Share of each corpus's phishing (%)", fontsize=6.8, color=INK,
             ha="center", va="bottom")
    fig_text(fig, left_recall + recall_w, header_y, "Recall at 0.5 (word, aug.)",
             fontsize=6.8, color=INK, ha="center", va="bottom")
    for x0, width in ((left_shares, len(TOPIC_SHARES) * share_w), (left_recall, 2 * recall_w)):
        fig.add_artist(Line2D([(x0 + 0.03) / 7.1, (x0 + width - 0.03) / 7.1],
                              [(header_y - 0.035) / 3.85] * 2, color=MUTED, lw=0.6))
    panel_title(fig, 2.72, 3.68, "b", "Topic shares and per-topic recall")
    fig_text(fig, 2.72, 3.52, "12 NMF topics, rows labelled by their first three terms; "
             "sorted by share of L3 human phishing", fontsize=6.6, color=MUTED,
             ha="left", va="baseline")
    for i, row in ordered.iterrows():
        print(f"  (b) topic {int(row['topic']):2d} {labels[i].replace(chr(10), ' '):36s} "
              f"shares {np.round(100 * shares[i], 1)} recall L3 {row['recall_human_at_0.5']:.3f} "
              f"(n={int(row['n_human'])}) Nazario {row['recall_nazario_at_0.5']:.3f} "
              f"(n={int(row['n_nazario'])})")
    print(f"  (b) L3 human phishing assigned to a topic: {int(ordered['n_human'].sum())}; "
          f"Nazario: {int(ordered['n_nazario'].sum())}")
    finish(fig, output_dir, "l3_explained")


# ==========================================================================
# 3. era-matched real phishing by year
# ==========================================================================
def _by_year(frame: pd.DataFrame, family: str, training: str, column: str) -> pd.DataFrame:
    block = frame[(frame["family"] == family) & (frame["training"] == training)]
    grouped = block.groupby("year")[column]
    return pd.DataFrame({"mean": grouped.mean(), "sd": grouped.std(ddof=1),
                         "partitions": grouped.size()}).sort_index()


def figure_era_matched(output_dir: Path) -> None:
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
    year_values = counts.index.to_numpy()

    fig = plt.figure(figsize=(7.1, 3.05))
    axes = [add_axes(fig, 0.55, 0.78, 3.05, 1.80)]
    axes.append(add_axes(fig, 3.88, 0.78, 3.05, 1.80, sharey=axes[0]))
    panels = [("recall_t0.5", "a", "Threshold 0.5"),
              ("recall_tcost", "b", "Cost-minimizing threshold ($\\pi$ = 1%, $r$ = 10)")]
    for ax, (column, letter, title) in zip(axes, panels):
        for family in families:
            name, colour = FAMILY[family]
            for training, style, face, band_alpha in (("clean", ":", "white", 0.09),
                                                      ("augmented", "-", colour, 0.16)):
                curve = _by_year(years, family, training, column)
                if curve.empty:
                    continue
                if curve["partitions"].min() > 1:
                    ax.fill_between(curve.index, curve["mean"] - curve["sd"],
                                    curve["mean"] + curve["sd"], color=colour,
                                    alpha=band_alpha, lw=0, zorder=1)
                ax.plot(curve.index, curve["mean"], ls=style, marker="o", ms=3.0, lw=1.35,
                        color=colour, mfc=face, mec=colour, mew=0.9, zorder=3)
                early = curve.loc[curve.index <= 2022, "mean"]
                late = curve.loc[curve.index >= 2023, "mean"]
                print(f"  ({letter}) {family:10s} {training:9s} unweighted mean of years "
                      f"2015-22 {early.mean():.3f} 2023-25 {late.mean():.3f}; min "
                      f"{curve['mean'].min():.3f} ({int(curve['mean'].idxmin())}) max "
                      f"{curve['mean'].max():.3f} ({int(curve['mean'].idxmax())}); max SD "
                      f"{curve['sd'].max():.3f}; partitions {int(curve['partitions'].min())}")
                if column == "recall_t0.5" and (family, training) in reference:
                    ax.axhline(reference[(family, training)], color=colour, ls=style,
                               lw=0.9, alpha=0.85, zorder=2)
        ax.axvline(2022.5, color=MUTED, ls="--", lw=0.9, zorder=0)
        ax.set_xlim(2014.5, 2025.5)
        ax.set_xticks(year_values)
        ax.set_xticklabels([str(year) for year in year_values], fontsize=6.3)
        ax.tick_params(axis="x", length=2)
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for year, n in counts.items():
            ax.text(year, -0.125, f"{n}", transform=transform, ha="center", va="top",
                    fontsize=5.8, color=MUTED)
        ax.text(2014.5 - 0.2, -0.125, "$n$", transform=transform, ha="right", va="top",
                fontsize=6.0, color=MUTED)
        ax.set_xlabel("Collection year (Nazario phishing corpus)", labelpad=11)
        light_grid(ax)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    axes[0].set_ylabel("Recall on era-matched\nreal phishing")
    axes[0].set_ylim(0.0, 1.21)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for ax in axes:
        ax.text(2022.5, 1.19, "public release of\ninstruction-tuned chat models",
                ha="center", va="top", fontsize=5.9, color=MUTED, linespacing=0.95,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"}, zorder=5)
    high = max(value for key, value in reference.items() if key[0] in families)
    axes[0].annotate("2000s L3 human phishing\n(recall at 0.5, one line per model)",
                     xy=(2015.6, high), xytext=(2014.85, high + 0.16), ha="left",
                     va="bottom", fontsize=6.0, color=INK, linespacing=0.95,
                     arrowprops={"arrowstyle": "-", "lw": 0.6, "color": MUTED})
    for key, value in sorted(reference.items()):
        if key[0] in families:
            print(f"  (a) L3 human 2000s recall@0.5 {key[0]:10s} {key[1]:9s} {value:.3f}")

    def family_label(family):
        name, _ = FAMILY[family]
        count = partitions[family]
        if family == "distilbert" and count < bert_expected:
            return f"{name} ({count} of {bert_expected} partitions)"
        return f"{name} ({count} partitions)"

    handles = [Line2D([], [], color=FAMILY[f][1], lw=1.6, label=family_label(f))
               for f in families]
    styles = [
        Line2D([], [], color=INK, ls="-", marker="o", ms=3.0, lw=1.35, label="Augmented"),
        Line2D([], [], color=INK, ls=":", marker="o", ms=3.0, mfc="white", lw=1.35,
               label="Clean-only"),
        Patch(color=INK, alpha=0.15, lw=0, label="±1 SD over partitions"),
    ]
    ordered = [h for pair in zip(handles + [None] * (3 - len(handles)), styles)
               for h in pair if h is not None]
    fig.legend(handles=ordered, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005),
               fontsize=6.6, handlelength=2.2, columnspacing=1.6, labelspacing=0.3)
    panel_title(fig, 0.05, 2.80, "a", "Threshold 0.5")
    panel_title(fig, 3.88, 2.80, "b", "Cost-minimizing threshold")
    fig_text(fig, 3.88, 2.64, "validation threshold minimizing expected cost at "
             "$\\pi$ = 1%, $r$ = 10", fontsize=6.6, color=MUTED, ha="left", va="baseline")
    print("  n per year: " + ", ".join(f"{year} {n}" for year, n in counts.items()))
    finish(fig, output_dir, "era_matched")


# ==========================================================================
# 4. deployment robustness
# ==========================================================================
ESTIMATORS = [
    ("CC", "CC", "o"),
    ("ACC_t0.5", "ACC, $t$ = 0.5", "s"),
    ("ACC_t95R", "ACC, 95%-recall $t$", "D"),
    ("EM", "EM", "^"),
]
PROCEDURES = [
    ("three-band rule (as stated)", "Three-band rule\n(as stated)"),
    ("always fixed_cost_min", "Always validation\ncost-min. $t$"),
    ("decile map learned on word aug., L1 primary, partitions 17..631",
     "Decile map\nlearned on L1"),
    ("always fixed_95_recall", "Always validation\n95%-recall $t$"),
    ("always fixed_0.5", "Always $t$ = 0.5"),
]
PERTURBATIONS = [
    ("temperature", (0.5, 2.0, 3.0), "temperature"),
    ("shift", (-2.0, -1.0, 1.0, 2.0), "logit shift"),
    ("noise", (0.5, 1.0, 2.0), "logit noise (SD)"),
]
FLOOR = 1e-2


def _value_label(kind: str, value: float) -> str:
    return signed(value, 0) if kind == "shift" else f"{value:g}"


def figure_deployment_robustness(output_dir: Path) -> None:
    estimation = pd.read_csv(RESULTS / "revision_r4_deploy_prevalence_estimation.csv")
    validation = pd.read_csv(RESULTS / "revision_r4_deploy_rule_validation.csv")
    sensitivity = pd.read_csv(RESULTS / "revision_r4_deploy_score_sensitivity.csv")
    stream_keys = [key for key, _, _ in STREAMS]

    width, height = 7.1, 5.85
    fig = plt.figure(figsize=(width, height))

    # ---- (a) prevalence estimation ---------------------------------------
    medians = (estimation.groupby(["stream", "estimator", "true_prevalence"])["ratio_to_truth"]
               .median())
    support = estimation.groupby(["stream", "estimator", "true_prevalence"]).size()
    zero_share = (estimation.assign(zero=estimation["ratio_to_truth"] < FLOOR)
                  .groupby(["stream", "estimator", "true_prevalence"])["zero"].mean())
    offsets = np.linspace(-0.28, 0.28, len(ESTIMATORS))
    a_bottom, a_height, a_left, a_width, a_gap = 3.62, 1.38, 0.66, 1.08, 0.07
    axes_a = []
    for index, prevalence in enumerate(sorted(estimation["true_prevalence"].unique())):
        ax = add_axes(fig, a_left + index * (a_width + a_gap), a_bottom, a_width, a_height,
                      sharey=axes_a[0] if axes_a else None)
        axes_a.append(ax)
        ax.axhspan(0.5, 2.0, color="#E8EDF0", lw=0, zorder=0)
        ax.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
        ax.axhspan(4e-3, 1.5e-2, color="#F3F3F3", lw=0, zorder=0)
        for s_index, stream in enumerate(stream_keys):
            colour = STREAM_COLOR[stream]
            for offset, (estimator, _, marker) in zip(offsets, ESTIMATORS):
                value = float(medians[(stream, estimator, prevalence)])
                clipped = value < FLOOR
                ax.plot(s_index + offset, max(value, FLOOR), marker=marker, ms=4.0,
                        color=colour, mfc="white" if clipped else colour, mec=colour,
                        mew=0.9, ls="none", zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(6e-3, 2.5e3)
        ax.set_xlim(-0.55, len(stream_keys) - 0.45)
        ax.set_xticks(range(len(stream_keys)))
        ax.set_xticklabels([STREAM_SHORT[s] for s in stream_keys], fontsize=6.8)
        for tick, stream in zip(ax.get_xticklabels(), stream_keys):
            tick.set_color(STREAM_COLOR[stream])
            tick.set_fontweight("bold")
        ax.tick_params(axis="x", length=0, pad=2.5)
        ax.set_title(f"true $\\pi$ = {100 * prevalence:g}%", fontsize=7.2,
                     fontweight="normal", pad=3)
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6, which="major")
        ax.set_axisbelow(True)
        if index:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="y", length=0)
    axes_a[0].set_yticks([1e-2, 1e-1, 1, 10, 100, 1000])
    axes_a[0].set_yticklabels(["≤0.01", "0.1", "1", "10", "100", "1000"])
    axes_a[0].yaxis.set_minor_locator(NullLocator())
    axes_a[0].set_ylabel("Estimated / true prevalence\n(median over models × partitions)",
                         fontsize=7.5)
    handles = [Line2D([], [], color=INK, marker=marker, ms=4.0, ls="none", label=label)
               for _, label, marker in ESTIMATORS]
    handles.append(Line2D([], [], color=INK, marker="o", ms=4.0, mfc="white", ls="none",
                          label="median ≤ 0.01 (at floor)"))
    a_right = a_left + 3 * a_width + 2 * a_gap
    fig.legend(handles=handles, loc="center", ncol=5, fontsize=6.1, handletextpad=0.1,
               columnspacing=0.75, bbox_to_anchor=(((a_left + a_right) / 2 - 0.12) / width,
                                                   5.255 / height))
    panel_title(fig, 0.02, 5.42, "a", "Label-free prevalence estimates")
    print("  (a) median ratio estimate/truth at 0.1%, 1%, 5% [share of rows < 0.01] (rows)")
    for stream in stream_keys:
        for estimator, _, _ in ESTIMATORS:
            values = [medians[(stream, estimator, p)] for p in (0.001, 0.01, 0.05)]
            zeros = [zero_share[(stream, estimator, p)] for p in (0.001, 0.01, 0.05)]
            rows_used = int(support[(stream, estimator, 0.001)])
            print(f"      {STREAM_SHORT[stream]:5s} {estimator:9s} "
                  + " ".join(f"{v:9.4g}" for v in values)
                  + "  [" + " ".join(f"{z:.2f}" for z in zeros) + f"]  ({rows_used})")

    # ---- (b) held-out rule validation ------------------------------------
    b_left = 5.02
    ax_excess = add_axes(fig, b_left, a_bottom, 1.05, a_height)
    ax_agree = add_axes(fig, b_left + 1.05 + 0.10, a_bottom, 0.82, a_height, sharey=ax_excess)
    table = validation.groupby(["rule", "stream"])[["agreement", "mean_excess"]].mean()
    models = validation.groupby(["rule", "stream"])["model"].nunique()
    y_offsets = np.linspace(-0.27, 0.27, len(stream_keys))
    for r_index, (rule, _) in enumerate(PROCEDURES):
        for offset, stream in zip(y_offsets, stream_keys):
            if (rule, stream) not in table.index:
                continue
            row = table.loc[(rule, stream)]
            colour = STREAM_COLOR[stream]
            ax_excess.plot(row["mean_excess"], r_index + offset, "o", ms=3.8, color=colour,
                           zorder=3)
            ax_agree.plot(row["agreement"], r_index + offset, "o", ms=3.8, color=colour,
                          zorder=3)
    for axis in (ax_excess, ax_agree):
        for r_index in range(len(PROCEDURES) - 1):
            axis.axhline(r_index + 0.5, color=GRID, lw=0.5)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color=GRID, lw=0.5, alpha=0.6)
        axis.set_axisbelow(True)
    ax_excess.set_xscale("log")
    ax_excess.set_xlim(4e-2, 25)
    ax_excess.set_xticks([0.1, 1, 10])
    ax_excess.set_xticklabels(["10%", "100%", "1000%"])
    ax_excess.xaxis.set_minor_locator(NullLocator())
    ax_excess.set_xlabel("Mean excess cost\nover best rule", fontsize=7.5)
    ax_excess.set_yticks(range(len(PROCEDURES)))
    ax_excess.set_yticklabels([label for _, label in PROCEDURES], fontsize=6.4,
                              linespacing=0.95)
    ax_excess.set_ylim(len(PROCEDURES) - 0.5, -0.5)
    ax_agree.set_xlim(-0.04, 1.04)
    ax_agree.set_xticks([0, 0.5, 1])
    ax_agree.set_xticklabels(["0", "0.5", "1"])
    ax_agree.set_xlabel("Agreement with\nbest rule", fontsize=7.5)
    plt.setp(ax_agree.get_yticklabels(), visible=False)
    panel_title(fig, 3.98, 5.42, "b", "Held-out rule validation (all models)")
    print("  (b) rule, stream: mean_excess, agreement (equal-weight mean over models; #models)")
    for rule, label in PROCEDURES:
        for stream in stream_keys:
            if (rule, stream) in table.index:
                row = table.loc[(rule, stream)]
                print(f"      {label.replace(chr(10), ' ')[:34]:34s} {STREAM_SHORT[stream]:5s} "
                      f"{row['mean_excess']:.3f} {row['agreement']:.3f} "
                      f"models {models[(rule, stream)]}")

    # ---- (c) score-distribution sensitivity -------------------------------
    c_left, c_width = 0.66, 6.36
    ax_rule = add_axes(fig, c_left, 0.62, c_width, 1.08)
    ax_purity = add_axes(fig, c_left, 1.80, c_width, 0.92, sharex=ax_rule)
    means = sensitivity.groupby(["scope", "perturbation", "value", "stream"])[
        ["pi_r_purity", "mean_excess"]].mean()
    model_names = sorted(sensitivity["model"].unique())
    conditions = [("none", "none", 0.0, 0.0)]
    position_x = 1.4
    kind_spans, scope_spans = [], []
    for scope in ("validation and target", "target only"):
        scope_start = position_x
        for kind, values, label in PERTURBATIONS:
            kind_start = position_x
            for value in values:
                conditions.append((scope, kind, value, position_x))
                position_x += 1.0
            kind_spans.append((label, kind_start, position_x - 1.0))
            position_x += 0.55
        scope_spans.append((scope, scope_start, position_x - 1.55))
        position_x += 0.75
    x_offsets = np.linspace(-0.24, 0.24, len(stream_keys))
    for axis, column, connect in ((ax_purity, "pi_r_purity", False),
                                  (ax_rule, "mean_excess", True)):
        for offset, stream in zip(x_offsets, stream_keys):
            colour = STREAM_COLOR[stream]
            for scope in ("none", "validation and target", "target only"):
                for kind in ("none", "temperature", "shift", "noise"):
                    points = [(x, means.loc[(sc, kd, val, stream), column])
                              for sc, kd, val, x in conditions
                              if sc == scope and kd == kind and (sc, kd, val, stream) in means.index]
                    if not points:
                        continue
                    xs, ys = zip(*points)
                    xs = np.array(xs) + offset
                    if connect:
                        axis.plot(xs, ys, color=colour, lw=0.6, alpha=0.4, zorder=2)
                    axis.plot(xs, ys, "o", ms=3.2, color=colour, zorder=3)
        axis.axvspan(-0.6, 0.6, color="#EEF2F4", lw=0, zorder=0)
        axis.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
        axis.set_axisbelow(True)
        for _, start, _ in scope_spans:
            axis.axvline(start - 0.6, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax_purity.set_ylim(0.825, 1.015)
    ax_purity.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax_purity.set_ylabel("$\\pi r$ purity", fontsize=7.5)
    plt.setp(ax_purity.get_xticklabels(), visible=False)
    ax_purity.tick_params(axis="x", length=0)
    ax_rule.set_yscale("log")
    ax_rule.set_ylim(0.05, 6.5)
    ax_rule.set_yticks([0.1, 0.3, 1, 3])
    ax_rule.set_yticklabels(["10%", "30%", "100%", "300%"])
    ax_rule.yaxis.set_minor_locator(NullLocator())
    ax_rule.set_ylabel("Three-band rule:\nmean excess cost", fontsize=7.5)
    ax_rule.set_xticks([x for _, _, _, x in conditions])
    ax_rule.set_xticklabels(["none"] + [_value_label(kind, value)
                                        for _, kind, value, _ in conditions[1:]], fontsize=6.4)
    ax_rule.tick_params(axis="x", length=2, pad=1.5)
    ax_rule.set_xlim(-0.75, position_x - 1.2)
    on_rule = blended_transform_factory(ax_rule.transData, ax_rule.transAxes)
    for label, start, stop in kind_spans:
        ax_rule.plot([start - 0.3, stop + 0.3], [-0.20, -0.20], transform=on_rule,
                     color=MUTED, lw=0.7, clip_on=False)
        ax_rule.text((start + stop) / 2, -0.235, label, transform=on_rule, ha="center",
                     va="top", fontsize=6.2, color=MUTED)
    ax_rule.text(0.0, -0.235, "baseline", transform=on_rule, ha="center", va="top",
                 fontsize=6.2, color=MUTED)
    on_purity = blended_transform_factory(ax_purity.transData, ax_purity.transAxes)
    for scope, start, stop in scope_spans:
        text = ("validation and target perturbed (a detector with another score distribution)"
                if scope == "validation and target"
                else "target only perturbed (score shift at deployment)")
        ax_purity.text((start + stop) / 2, 1.05, text, transform=on_purity, ha="center",
                       va="bottom", fontsize=6.4, color=INK)
    panel_title(fig, 0.02, 3.03, "c", "Sensitivity to the score distribution")
    fig_text(fig, 3.33, 3.03, "word TF-IDF and MiniLM, augmented; all partitions",
             fontsize=6.6, color=MUTED, ha="left", va="baseline")
    print(f"  (c) models: {model_names}")
    print("  (c) purity / three-band mean excess by condition, streams L1 L2 L3 Cont. "
          "(mean over models)")
    for scope, kind, value, _ in conditions:
        cells = [means.loc[(scope, kind, value, stream)] for stream in stream_keys]
        print(f"      {scope:22s} {kind:11s} {value:+.1f} purity "
              + " ".join(f"{c['pi_r_purity']:.3f}" for c in cells) + "  excess "
              + " ".join(f"{c['mean_excess']:.3f}" for c in cells))
    print(f"      purity range over all conditions: {means['pi_r_purity'].min():.3f}-"
          f"{means['pi_r_purity'].max():.3f}")

    handles = [Line2D([], [], color=colour, marker="o", ms=4.2, ls="none", label=label)
               for _, label, colour in STREAMS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               fontsize=7.0, handletextpad=0.25, columnspacing=1.4)
    finish(fig, output_dir, "deployment_robustness")


# ==========================================================================
FIGURE_FUNCTIONS = {
    "pipeline_signature": figure_pipeline_signature,
    "l3_explained": figure_l3_explained,
    "era_matched": figure_era_matched,
    "deployment_robustness": figure_deployment_robustness,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURE_FUNCTIONS),
                        help="draw only these figures")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument("--copy-to", type=Path, default=None,
                        help="also copy the PNG files to this directory")
    args = parser.parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(FIGURE_FUNCTIONS):
        print(f"[{name}]")
        FIGURE_FUNCTIONS[name](args.output_dir)
        if args.copy_to is not None:
            shutil.copy2(args.output_dir / f"{name}.png", args.copy_to / f"{name}.png")
            print(f"  copied {name}.png to {args.copy_to}")
    print(f"figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
