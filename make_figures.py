#!/usr/bin/env python3
"""Generate publication figures directly from archived result summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COLORS = {
    "Word TF--IDF": "#0072B2",
    "MiniLM": "#009E73",
    "DistilBERT": "#D55E00",
}
LIGHT = "#D9E2E8"
GRID = "#CBD3D8"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def dose_figure(output_dir: Path) -> None:
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
        # R4: limits widened slightly so no error-bar cap is clipped
        ("linked_rewrites", "Linked rewrites", (0.925, 0.988)),
        ("external_openai", "Generator-disjoint corpus", (0.776, 0.96)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    x = np.arange(len(conditions))
    for panel_index, (evaluation, title, ylim) in enumerate(panels):
        ax = axes[panel_index]
        for family, display in families:
            subset = (
                frame[
                    frame["family"].eq(family)
                    & frame["evaluation_condition"].eq(evaluation)
                ]
                .set_index("training_condition")
                .loc[conditions]
            )
            ax.errorbar(
                x,
                subset["f1_mean"],
                yerr=subset["f1_std"],
                color=COLORS[display],
                marker="o",
                markersize=4.2,
                linewidth=1.7,
                capsize=2.2,
                capthick=0.8,
                label=display.replace("--", "-"),
                zorder=3,
            )
        # R4: panel letters as "(a)", "(b)", matching every other figure
        ax.set_title(f"({chr(97 + panel_index)}) {title}", loc="left")
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.set_ylabel("F1")
        ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle=":", zorder=0)
        ax.tick_params(axis="x", length=0)
    axes[0].legend(loc="lower right")
    fig.text(
        0.5,
        -0.02,
        "Training exposure (variants per source group)",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(w_pad=1.7)
    finish(fig, output_dir, "augmentation_dose")


def paired_summary(
    frame: pd.DataFrame, model: str
) -> tuple[pd.Series, pd.Series]:
    external = frame[
        frame["model"].eq(model) & frame["condition"].eq("external_openai")
    ]
    spam = frame[
        frame["model"].eq(model) & frame["condition"].eq("spam_hard_negative")
    ]
    if len(external) != 1 or len(spam) != 1:
        raise ValueError(f"Missing paired summary rows for {model}")
    return external.iloc[0], spam.iloc[0]


def tradeoff_figure(output_dir: Path) -> None:
    lightweight = pd.read_csv(RESULTS / "summary.csv")
    transformer = pd.read_csv(RESULTS / "transformer_summary.csv")
    model_specs = {
        "Word TF--IDF": (
            lightweight,
            "Word LR, clean-only",
            "Word LR, linked-augmented",
        ),
        "MiniLM": (
            lightweight,
            "MiniLM LR, clean-only",
            "MiniLM LR, linked-augmented",
        ),
        "DistilBERT": (
            transformer,
            "DistilBERT, clean-only",
            "DistilBERT, linked-augmented",
        ),
    }

    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    label_offsets = {
        "Word TF--IDF": (-66, 6),
        "MiniLM": (7, -13),
        "DistilBERT": (-72, 7),
    }
    for display, (frame, clean_name, augmented_name) in model_specs.items():
        clean_external, clean_spam = paired_summary(frame, clean_name)
        aug_external, aug_spam = paired_summary(frame, augmented_name)
        points = [
            (clean_external, clean_spam, "clean"),
            (aug_external, aug_spam, "augmented"),
        ]
        x_values = [float(point[0]["f1_mean"]) for point in points]
        y_values = [float(point[1]["fpr_at_0_5_mean"]) for point in points]
        ax.annotate(
            "",
            xy=(x_values[1], y_values[1]),
            xytext=(x_values[0], y_values[0]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS[display],
                "lw": 1.4,
                "mutation_scale": 8,
            },
            zorder=2,
        )
        for index, (external, spam, _) in enumerate(points):
            ax.errorbar(
                external["f1_mean"],
                spam["fpr_at_0_5_mean"],
                xerr=external["f1_std"],
                yerr=spam["fpr_at_0_5_std"],
                fmt="o",
                markersize=5.0,
                markerfacecolor="white" if index == 0 else COLORS[display],
                markeredgecolor=COLORS[display],
                markeredgewidth=1.2,
                ecolor=COLORS[display],
                elinewidth=0.8,
                capsize=2,
                zorder=3,
            )
        ax.annotate(
            display,
            (x_values[1], y_values[1]),
            xytext=label_offsets[display],
            textcoords="offset points",
            color=COLORS[display],
            fontsize=7.5,
            fontweight="bold",
        )

    ax.set_xlim(0.79, 0.965)
    ax.set_ylim(0.02, 0.225)
    ax.set_xlabel("Generator-disjoint F1  (higher is better)")
    ax.set_ylabel("Spam FPR@0.5  (lower is better)")
    ax.grid(color=GRID, linewidth=0.6, linestyle=":", zorder=0)
    state_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#52616B",
            markerfacecolor="white",
            linewidth=0,
            label="Clean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#52616B",
            markerfacecolor="#52616B",
            linewidth=0,
            label="Augmented",
        ),
    ]
    ax.legend(handles=state_legend, loc="upper left", ncol=2, handletextpad=0.3)
    fig.tight_layout()
    finish(fig, output_dir, "external_spam_tradeoff")


def stress_figure(output_dir: Path) -> None:
    leakage = pd.read_csv(RESULTS / "rigor_leakage_summary.csv")
    cue = pd.read_csv(RESULTS / "rigor_cue_mask_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))

    ax = axes[0]
    families = [("Word LR", "Word TF--IDF"), ("MiniLM LR", "MiniLM")]
    split_specs = [("source_group", "Source-group", "#8CBFD9"), ("row_stratified", "Row-wise", "#E69F00")]
    x = np.arange(len(families))
    width = 0.34
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
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            capsize=2,
            label=display,
            zorder=3,
        )
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.0016,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_title("(a) Split leakage", loc="left")
    ax.set_xticks(x, [display.replace("--", "-") for _, display in families])
    ax.set_ylim(0.94, 0.995)
    ax.set_ylabel("F1")
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle=":", zorder=0)
    # --- Test-group overlap callout (key finding of this panel) ---
    src_ink, row_ink = "#2F7DB0", "#B5760A"   # readable, hue-matched to bars
    _cx = 0.758
    _texts = [
        ax.text(_cx, 0.858, "TEST-GROUP OVERLAP", transform=ax.transAxes,
                ha="center", va="center", fontsize=6.6, fontweight="bold",
                color="#5A6B75", zorder=5),
        ax.text(_cx - 0.132, 0.788, "0.0%", transform=ax.transAxes,
                ha="center", va="center", fontsize=12.0, fontweight="bold",
                color=src_ink, zorder=5),
        ax.text(_cx + 0.132, 0.788, "99.7%", transform=ax.transAxes,
                ha="center", va="center", fontsize=12.0, fontweight="bold",
                color=row_ink, zorder=5),
        ax.text(_cx, 0.726, "source-group  vs  row-wise", transform=ax.transAxes,
                ha="center", va="center", fontsize=6.0, color="#7C8A93",
                zorder=5),
    ]
    ax.annotate("", xy=(_cx + 0.040, 0.788), xytext=(_cx - 0.020, 0.788),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "color": "#8A99A2",
                            "lw": 1.3, "mutation_scale": 8}, zorder=5)
    for _t in _texts:
        _t.set_clip_on(False)
    # size the background box to actually wrap the text (measure, don't guess)
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    _inv = ax.transAxes.inverted()
    _xs, _ys = [], []
    for _t in _texts:
        _bb = _t.get_window_extent(_rend)
        (_x0, _y0) = _inv.transform((_bb.x0, _bb.y0))
        (_x1, _y1) = _inv.transform((_bb.x1, _bb.y1))
        _xs += [_x0, _x1]; _ys += [_y0, _y1]
    _padx, _pady = 0.026, 0.026
    _x0, _x1 = min(_xs) - _padx, max(_xs) + _padx
    _y0, _y1 = min(_ys) - _pady, max(_ys) + _pady
    card = FancyBboxPatch(
        (_x0, _y0), _x1 - _x0, _y1 - _y0,
        boxstyle="round,pad=0,rounding_size=0.030",
        transform=ax.transAxes,
        facecolor="#F5F8FA", edgecolor="#D3DCE2", linewidth=0.7, zorder=4,
    )
    card.set_clip_on(False)
    ax.add_patch(card)

    ax = axes[1]
    evaluations = [
        ("linked_rewrites", "Linked"),
        ("external_openai", "Gen.-disjoint"),
    ]
    input_specs = [
        ("original", "Original", "#009E73"),
        ("top20_positive_unigrams_masked", "Top-20 masked", "#D55E00"),
    ]
    x = np.arange(len(evaluations))
    for offset, (input_condition, display, color) in zip(
        (-width / 2, width / 2), input_specs
    ):
        means = []
        stds = []
        for evaluation, _ in evaluations:
            row = cue[
                cue["evaluation_condition"].eq(evaluation)
                & cue["input_condition"].eq(input_condition)
            ].iloc[0]
            means.append(row["f1_mean"])
            stds.append(row["f1_std"])
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            capsize=2,
            label=display,
            zorder=3,
        )
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.004,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_title("(b) Lexical-cue masking", loc="left")
    ax.set_xticks(x, [display for _, display in evaluations])
    ax.set_ylim(0.84, 1.005)
    ax.set_ylabel("F1")
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle=":", zorder=0)
    ax.legend(loc="upper right")
    fig.tight_layout(w_pad=1.8)
    finish(fig, output_dir, "leakage_and_masking")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "figures",
        help="Directory for generated PDF and PNG figures.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    dose_figure(args.output_dir)
    tradeoff_figure(args.output_dir)
    stress_figure(args.output_dir)
    print(f"Wrote six figure files to {args.output_dir}")


if __name__ == "__main__":
    main()
