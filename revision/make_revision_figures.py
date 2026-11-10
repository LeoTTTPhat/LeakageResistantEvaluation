#!/usr/bin/env python3
"""Figures for the revision analyses, in the style of make_figures.py."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rev_common import FIGURES, RESULTS, STUDY

sys.path.insert(0, str(STUDY))
from make_figures import COLORS, configure_style, finish  # noqa: E402

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


def figure_calibration(output_dir):
    frame = pd.read_csv(RESULTS / "revision_distilbert_probabilities.csv")
    frame = frame[(frame["max_length"] == 160) & (frame["epoch"] == 1)]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--", zorder=1)
    for training, (style, marker) in TRAIN_STYLE.items():
        block = frame[
            (frame["training"] == training)
            & (frame["condition"] == "external_openai")
        ]
        centres, observed, _ = reliability(
            block["probability"].to_numpy(), block["label"].to_numpy()
        )
        ax.plot(
            centres,
            observed,
            style,
            marker=marker,
            ms=3,
            lw=1.3,
            color=COLORS["DistilBERT"],
            alpha=1.0 if training == "linked-augmented" else 0.65,
            label="Augmented" if training == "linked-augmented" else "Clean",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("(a) Reliability, gen.-disjoint", loc="left")
    ax.legend(loc="upper left")

    for ax, (condition, title) in zip(
        axes[1:],
        [
            ("linked_rewrites", "(b) Confidence, linked"),
            ("external_openai", "(c) Confidence, gen.-disjoint"),
        ],
    ):
        for training, (style, _) in TRAIN_STYLE.items():
            block = frame[
                (frame["training"] == training) & (frame["condition"] == condition)
            ]
            ax.hist(
                block["probability"],
                bins=25,
                histtype="step",
                lw=1.3,
                ls=style,
                density=True,
                color=COLORS["DistilBERT"],
                alpha=1.0 if training == "linked-augmented" else 0.65,
                label="Augmented" if training == "linked-augmented" else "Clean",
            )
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.set_title(title, loc="left")
    axes[1].legend(loc="upper center")
    fig.tight_layout()
    finish(fig, output_dir, "distilbert_calibration")


def figure_learning_curve(output_dir):
    frame = pd.read_csv(RESULTS / "revision_distilbert_learning_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharex=True)
    panels = [
        (160, "f1", "(a) F1, max length 160"),
        (320, "f1", "(b) F1, max length 320"),
        (160, "ece", "(c) ECE, max length 160"),
    ]
    condition_colour = {
        "linked_rewrites": COLORS["Word TF--IDF"],
        "external_openai": COLORS["DistilBERT"],
    }
    for ax, (max_length, metric, title) in zip(axes, panels):
        block = frame[frame["max_length"] == max_length]
        if block.empty:
            continue
        for condition, colour in condition_colour.items():
            for training, (style, _) in TRAIN_STYLE.items():
                part = block[
                    (block["eval_condition"] == condition)
                    & (block["training"] == training)
                ]
                if part.empty:
                    continue
                curve = part.groupby("epoch_progress")[metric].mean()
                ax.plot(
                    curve.index,
                    curve.to_numpy(),
                    style,
                    lw=1.3,
                    color=colour,
                    alpha=1.0 if training == "linked-augmented" else 0.6,
                )
        ax.axvline(1.0, color="#888888", lw=0.8, ls="--")
        ax.set_xlabel("Epochs of training")
        ax.set_ylabel("F1" if metric == "f1" else "ECE")
        ax.set_title(title, loc="left")
    handles = [
        plt.Line2D([], [], color=COLORS["Word TF--IDF"], lw=1.3, label="Linked"),
        plt.Line2D([], [], color=COLORS["DistilBERT"], lw=1.3, label="Gen.-disjoint"),
        plt.Line2D([], [], color="#444444", lw=1.3, ls="-", label="Augmented"),
        plt.Line2D([], [], color="#444444", lw=1.3, ls=":", label="Clean"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    finish(fig, output_dir, "distilbert_learning_curve")


def figure_threshold_cost(output_dir):
    frame = pd.read_csv(RESULTS / "revision_threshold_cost_curves.csv")
    dedup = RESULTS / "revision_r4_threshold_cost_curves_dedup.csv"
    if dedup.exists():  # round 4: L3 stream without the SpamAssassin duplicates
        frame = pd.concat([frame[frame["stream"] != "human_authored"], pd.read_csv(dedup)],
                          ignore_index=True)
    frame = frame[frame["model"] == "word_lr_linked_augmented"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    stream_colour = {
        "primary_pipeline": COLORS["Word TF--IDF"],
        "pipeline_disjoint": COLORS["MiniLM"],
        "human_authored": COLORS["DistilBERT"],
    }
    stream_label = {
        "primary_pipeline": "L1 primary",
        "pipeline_disjoint": "L2 gen.-disjoint",
        "human_authored": "L3 human",
    }
    for index, (ax, prevalence) in enumerate(zip(axes, sorted(frame["prevalence"].unique()))):
        block = frame[frame["prevalence"] == prevalence]
        for stream, colour in stream_colour.items():
            part = block[block["stream"] == stream].sort_values("threshold")
            if part.empty:
                continue
            ax.plot(
                part["threshold"],
                part["expected_cost"],
                lw=1.4,
                color=colour,
                label=stream_label[stream],
            )
            best = part.loc[part["expected_cost"].idxmin()]
            ax.plot([best["threshold"]], [best["expected_cost"]], "v", ms=4, color=colour)
        ax.axhline(
            prevalence * 10.0, color="#888888", lw=0.8, ls="--"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Expected cost per message")
        ax.set_title(f"({chr(97 + index)}) $\\pi$ = {prevalence:.1%}, $r$ = 10", loc="left")
    # R4: the legend sat on the curves inside panel (a); it now sits below the
    # panels, as in the learning-curve figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    finish(fig, output_dir, "threshold_cost")


def figure_transfer_ladder_r2(output_dir):
    """Round-2 single-panel ladder (five-seed means, no intervals). RETIRED in round 4.

    Figure 7 is now drawn by ``make_r4_revised_figures.py`` (``transfer_ladder``)
    from the full partition sets, with intervals and partition counts. This
    function is kept only as a record of the round-2 figure, is no longer called
    from ``main()``, and writes ``transfer_ladder_r2`` so that it can never
    overwrite the round-4 ``transfer_ladder`` file.
    """
    lightweight = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    ceiling = pd.read_csv(RESULTS / "revision_stream_ceiling_summary.csv")
    audit = pd.read_csv(RESULTS / "revision_cluster_aware_intervals.csv")

    models = [
        ("word_lr_clean_only", "Word clean", COLORS["Word TF--IDF"], 0.55),
        ("word_lr_linked_augmented", "Word aug.", COLORS["Word TF--IDF"], 1.0),
        ("minilm_lr_clean_only", "MiniLM clean", COLORS["MiniLM"], 0.55),
        ("minilm_lr_linked_augmented", "MiniLM aug.", COLORS["MiniLM"], 1.0),
    ]
    levels = ["primary_pipeline", "pipeline_disjoint", "human_authored"]
    labels = ["L1\nlinked", "L2\ngen.-disjoint", "L3\nhuman-authored"]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    positions = np.arange(len(levels))
    for key, label, colour, alpha in models:
        values = [
            ceiling[(ceiling["model"] == key) & (ceiling["stream"] == level)][
                "f1_at_0_5"
            ].mean()
            for level in levels
        ]
        ax.plot(
            positions,
            values,
            "-o",
            ms=3.5,
            lw=1.4,
            color=colour,
            alpha=alpha,
            label=label,
        )
    dedup = audit[
        (audit["corpus"] == "external_openai")
        & (audit["model"] == "word_lr_linked_augmented")
    ]
    if not dedup.empty:
        ax.plot(
            [1],
            [dedup["f1_cluster_deduplicated"].iloc[0]],
            marker="x",
            ms=6,
            color=COLORS["Word TF--IDF"],
        )
        ax.annotate(
            "cluster-deduplicated",
            xy=(0.98, dedup["f1_cluster_deduplicated"].iloc[0]),
            xytext=(0.04, 0.50),
            fontsize=6.5,
            color="#555555",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#999999"),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 at threshold 0.5")
    ax.set_ylim(0, 1.02)
    ax.set_title("Transfer ladder")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    finish(fig, output_dir, "transfer_ladder_r2")  # never "transfer_ladder" (R4 file)


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Round 4: the old ladder is no longer drawn here; transfer_ladder.{pdf,png}
    # (Figure 7) comes from make_r4_revised_figures.py. Call
    # figure_transfer_ladder_r2(FIGURES) explicitly to reproduce the round-2 figure.
    figure_threshold_cost(FIGURES)
    if (RESULTS / "revision_distilbert_probabilities.csv").exists():
        figure_calibration(FIGURES)
        figure_learning_curve(FIGURES)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
