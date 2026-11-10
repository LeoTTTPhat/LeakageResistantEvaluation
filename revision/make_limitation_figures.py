#!/usr/bin/env python3
"""Figures for the two limitation-closing analyses, in the style of make_figures.py.

* ``domain_adaptation.png`` -- the L3 ceiling, the transfer gap, the asymmetry of
  transfer, and the target-data exchange rate (``rev_i_domain_adaptation.py``).
* ``adaptive_evasion.png``  -- the evasion curve and its transfer across
  detectors, from the released aggregate tables.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rev_common import FIGURES, RESULTS, STUDY

sys.path.insert(0, str(STUDY))
from make_figures import COLORS, GRID, configure_style, finish  # noqa: E402

MODEL_LABEL = {"word_lr": "Word TF-IDF", "minilm_lr": "MiniLM"}
MODEL_COLOR = {"word_lr": COLORS["Word TF--IDF"], "minilm_lr": COLORS["MiniLM"]}
ARM_STYLE = {
    "human_only_k": ("-", "o", "target only"),
    "mixed_balanced": ("--", "s", "target + synthetic (upweighted)"),
    "mixed": (":", "^", "target + synthetic (as collected)"),
}


def domain_adaptation_figure(output_dir) -> None:
    summary = pd.read_csv(RESULTS / "revision_domain_adaptation_summary.csv")
    curve = pd.read_csv(RESULTS / "revision_fewshot_curve.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))

    # -- panel A: where the signal is, and which direction transfer works ------
    ax = axes[0]
    conditions = [
        ("synthetic_transfer", "linked_augmented", "synthetic\n\u2192 human"),
        ("human_in_domain", "human_only", "human\n\u2192 human"),
        ("reverse_to_L2", "human_only", "human\n\u2192 L2"),
        ("reverse_to_L1", "human_only", "human\n\u2192 L1"),
    ]
    width = 0.36
    offsets = {"word_lr": -width / 2, "minilm_lr": width / 2}
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
        ax.bar(
            np.arange(len(conditions)) + offset, heights, width,
            yerr=errors, capsize=2, color=MODEL_COLOR[model],
            label=MODEL_LABEL[model], edgecolor="white", linewidth=0.5,
            error_kw={"lw": 0.7},
        )
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([label for _, _, label in conditions])
    ax.set_ylabel("F1 at threshold 0.5")
    ax.set_ylim(0, 1.20)
    ax.set_title("(a) Training source \u2192 evaluation target", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2,
              columnspacing=1.0, handlelength=1.4)

    # -- panel B: the target-data exchange rate --------------------------------
    ax = axes[1]
    word = curve[curve["model"] == "word_lr"].copy()
    word["k"] = word["n_human_train_mean"].astype(int)
    for arm, (style, marker, label) in ARM_STYLE.items():
        block = word[(word["condition"] == arm) & (word["k"] > 0)].sort_values("k")
        if block.empty:
            continue
        ax.plot(
            block["k"], block["f1_mean"], style, marker=marker, ms=3.2,
            lw=1.2, color=MODEL_COLOR["word_lr"], label=label,
            markerfacecolor="white" if arm != "human_only_k" else None,
        )
    ceiling = float(
        summary.query("condition == 'human_in_domain' and model == 'word_lr'")
        ["f1_mean"].iloc[0]
    )
    floor = float(
        word[(word["condition"] == "mixed") & (word["k"] == 0)]["f1_mean"].iloc[0]
    )
    ax.axhline(ceiling, color="#888888", lw=0.8, ls="--")
    ax.text(26, ceiling - 0.06, "full-corpus ceiling", fontsize=6.5, color="#555555")
    ax.axhline(floor, color="#888888", lw=0.8, ls="-.")
    # R4: label below the line; above it, it sat on the first triangle marker
    ax.text(26, floor - 0.025, "synthetic only", fontsize=6.5, color="#555555", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("human-authored training messages ($k$)")
    ax.set_ylabel("F1 at threshold 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Word TF-IDF: target-data exchange rate", loc="left")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=1.0,
              edgecolor="none", borderpad=0.3)

    fig.tight_layout()
    finish(fig, output_dir, "domain_adaptation")


def adaptive_figure(output_dir) -> None:
    by_seed = pd.read_csv(RESULTS / "revision_adaptive_by_seed.csv")
    transfer = pd.read_csv(RESULTS / "revision_adaptive_transfer.csv")

    budgets = [5, 10, 20, 30, 40]
    label = {
        "word_lr_clean_only": ("Word, clean", ":", "o"),
        "word_lr_linked_augmented": ("Word, augmented", "-", "s"),
        "charplusword_lr_linked_augmented": ("Char+word, augmented", "--", "^"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75),
                             gridspec_kw={"width_ratios": [0.94, 1.06]})

    # -- panel A: evasion curve over the perturbation budget -------------------
    ax = axes[0]
    for model, (name, style, marker) in label.items():
        block = by_seed[by_seed["model"] == model]
        if block.empty:
            continue
        means = [block[f"success_at_{b}pct"].mean() for b in budgets]
        errors = [block[f"success_at_{b}pct"].std() for b in budgets]
        ax.errorbar(
            budgets, means, yerr=errors, fmt=marker, ls=style, ms=3.2, lw=1.2,
            capsize=2, label=name, color=COLORS["Word TF--IDF"]
            if model.startswith("word") else COLORS["DistilBERT"],
            markerfacecolor="white" if "clean" in model else None,
            elinewidth=0.7,
        )
    ax.set_xticks(budgets)  # the five budgets that were measured
    ax.set_xlabel("share of content words changed (%)")
    ax.set_ylabel("evasion rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Evasion rate vs. edit budget", loc="left")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(frameon=False, loc="upper left", fontsize=6.5,
                       title="mean \u00b1 SD over 3 splits, 150 messages each",
                       title_fontsize=6.5)
    legend._legend_box.align = "left"

    # -- panel B: does the adversarial message transfer to another detector? ---
    ax = axes[1]
    order = [
        "word_lr_clean_only", "word_lr_linked_augmented",
        "charplusword_lr_linked_augmented", "minilm_lr_clean_only",
        "minilm_lr_linked_augmented",
    ]
    short = {  # R4: full names instead of W-cln / CW-aug / M-aug abbreviations
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
    image = ax.imshow(grid, cmap="viridis_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[m] for m in order], fontsize=7.0)
    ax.set_yticks(range(len(attacked)))
    ax.set_yticklabels([short[m] for m in attacked], fontsize=7.0)
    ax.tick_params(length=0)
    ax.set_xlabel("scored by")
    ax.set_ylabel("messages crafted against")
    ax.set_title("(b) Recall on adversarial messages", loc="left")
    for i, source in enumerate(attacked):  # box the detector each row was crafted against
        j = order.index(source)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#1B1B1B",
                                   lw=1.3, zorder=3))
    for i in range(len(attacked)):
        for j in range(len(order)):
            if np.isnan(grid[i, j]):
                continue
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if grid[i, j] > 0.55 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("recall at threshold 0.5", fontsize=7.5)

    fig.tight_layout()
    fig.canvas.draw()  # place the box legend just below the measured x-axis label
    label_box = ax.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
    axes_box = ax.get_position()
    fig.text((axes_box.x0 + axes_box.x1) / 2, label_box.y0 - 0.012,
             "boxed: scored by the detector the messages were crafted against",
             ha="center", va="top", fontsize=6.5, color="#3D4A52")
    finish(fig, output_dir, "adaptive_evasion")


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    domain_adaptation_figure(FIGURES)
    print("wrote domain_adaptation.{pdf,png}")
    if (RESULTS / "revision_adaptive_by_seed.csv").exists():
        adaptive_figure(FIGURES)
        print("wrote adaptive_evasion.{pdf,png}")
    else:
        print("skipped adaptive_evasion (aggregate tables are missing)")


if __name__ == "__main__":
    main()
