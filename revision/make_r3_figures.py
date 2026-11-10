#!/usr/bin/env python3
"""Figure for the round-3 multi-partition inference (comment 2).

Round 4: Figure 3 (``multisplit_inference``) is now drawn by
``make_r4_revised_figures.py`` from 100 lightweight and 20 DistilBERT partitions.
This script reproduces the round-3 version only, under the name
``multisplit_inference_r3``, so that re-running it can no longer overwrite the
round-4 figure.
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rev_common import FIGURES, RESULTS, STUDY

sys.path.insert(0, str(STUDY))
from make_figures import COLORS, GRID, configure_style, finish  # noqa: E402

FAMILIES = ["Word TF-IDF", "MiniLM", "DistilBERT"]
LABEL = {"Word TF-IDF": "Word TF--IDF", "MiniLM": "MiniLM", "DistilBERT": "DistilBERT"}


def colour(family: str) -> str:
    return COLORS[LABEL[family]]


def figure_multisplit(output_dir) -> None:
    inference = pd.read_csv(RESULTS / "revision_r3_inference.csv")
    decomposition = json.loads(
        (RESULTS / "revision_r3_variance_decomposition.json").read_text()
    )
    convergence = pd.read_csv(RESULTS / "revision_r3_split_convergence.csv")
    per_split = {
        entry["family"]: entry["per_split_delta"] for entry in decomposition["families"]
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))

    # (a) per-partition effects
    ax = axes[0]
    for index, family in enumerate(FAMILIES):
        values = np.array(sorted(per_split[family].values()))
        jitter = np.linspace(-0.17, 0.17, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=11,
            color=colour(family),
            alpha=0.75,
            linewidths=0,
        )
        mean = values.mean()
        ax.plot([index - 0.28, index + 0.28], [mean, mean], color=colour(family), lw=1.6)
    ax.axhline(0.0, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(
        [f"{f}\n({len(per_split[f])})" for f in ["Word TF-IDF", "MiniLM", "DistilBERT"]]
    )
    ax.set_ylabel(r"$\Delta$ linked F1")
    ax.set_title("(a) Effect on each partition")

    # (b) interval comparison
    ax = axes[1]
    scopes = [
        ("all splits (hierarchical bootstrap)", "Hierarchical"),
        ("all splits (paired split-level t)", "Split-level $t$"),
        ("one further split (t prediction interval)", "One new partition"),
    ]
    offsets = np.linspace(-0.24, 0.24, len(scopes))
    for row, (scope, name) in enumerate(scopes):
        for index, family in enumerate(FAMILIES):
            entry = inference[
                (inference["family"] == family) & (inference["scope"] == scope)
            ]
            if entry.empty:
                continue
            entry = entry.iloc[0]
            ax.plot(
                [entry["ci_low"], entry["ci_high"]],
                [index + offsets[row]] * 2,
                color=colour(family),
                lw=1.5,
                alpha=1.0 - 0.25 * row,
                solid_capstyle="butt",
            )
            ax.plot(
                entry["delta"],
                index + offsets[row],
                marker="|",
                color=colour(family),
                markersize=5,
                mew=1.2,
            )
    ax.axvline(0.0, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels([LABEL[f].replace("--", "-") for f in FAMILIES])
    ax.set_ylim(-0.45, len(FAMILIES) + 0.62)
    ax.set_xlabel(r"$\Delta$ linked F1")
    ax.set_title("(b) 95% intervals by scope")
    handles = [
        plt.Line2D([], [], color="#5A6B75", lw=1.5, alpha=1.0 - 0.25 * row)
        for row in range(len(scopes))
    ]
    ax.legend(
        handles,
        [name for _, name in scopes],
        loc="upper left",
        handlelength=1.5,
        borderaxespad=0.2,
        labelspacing=0.25,
    )

    # (c) convergence in the number of partitions
    ax = axes[2]
    for family in FAMILIES:
        block = convergence[convergence["family"] == family].sort_values("splits_used")
        if block.empty:
            continue
        ax.plot(
            block["splits_used"],
            block["hierarchical_half_width"],
            marker="o",
            ms=3.2,
            lw=1.3,
            color=colour(family),
            label=LABEL[family].replace("--", "-"),
        )
        ax.plot(
            block["splits_used"],
            block["t_half_width"],
            marker="s",
            ms=3.0,
            lw=1.0,
            ls=":",
            color=colour(family),
        )
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20])
    ax.set_xticklabels(["3", "5", "10", "20"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(2.6, 24)
    ax.set_xlabel("Refitted partitions")
    ax.set_ylabel("95% interval half-width")
    ax.set_title("(c) Precision vs partitions")
    ax.legend(loc="upper right", handlelength=1.4)

    for ax in axes:
        ax.grid(True, color=GRID, lw=0.5, alpha=0.55)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.4)
    finish(fig, output_dir, "multisplit_inference_r3")  # R4 figure keeps the plain name


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_multisplit(FIGURES)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
