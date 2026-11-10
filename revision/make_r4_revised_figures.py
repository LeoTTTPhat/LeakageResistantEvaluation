#!/usr/bin/env python3
"""Round-4 redrawn figures (Reviewer 1, comment 7), in the style of make_figures.py.

Three manuscript figures are redrawn from the full sets of independently refitted
source-grouped partitions (100 for the lightweight models, 20 for DistilBERT):

* ``multisplit_inference`` (Figure 3; replaces the round-3 file of the same name)
    (a) paired effect (augmented - clean) on linked-rewrite F1 for every partition,
        with that partition's own 95% grouped-bootstrap interval;
    (b) 95% intervals for the mean effect and for one further partition, by scope;
    (c) stabilization of the between-partition SD with the number of partitions K;
    (d) ratio of the corrected resampled-t half-width to its large-K floor.
  Data: results/revision_r4_inference.csv, results/revision_r4_stabilization.csv
  (written by rev_r_r4_inference.py).

* ``tradeoff`` (Figure 6; replaces the hand-coded pgfplots figure)
    generator-disjoint (L2) F1 against spam hard-negative FPR, both at threshold
    0.5; one arrow per family from the clean-only to the augmented mean;
    whiskers are +/-1 SD across partitions.

* ``evaluation_ladder`` (Figure 7, formerly ``transfer_ladder``; replaces the round-2 single-panel file, which
  make_revision_figures.py no longer writes)
    (a) F1 at threshold 0.5 at L1, L2, L2 cluster-deduplicated and L3;
    (b) recall at threshold 0.5 at L1, L2, L3 and on the era-matched Nazario
        corpus (2015-2022 and 2023-2025).
    Means across partitions with 2.5th-97.5th percentile intervals across
    partitions.  DistilBERT enters panel (b) only once
    results/revision_r4_era_format_distilbert_metrics.csv (rev_u2_era_format_
    distilbert.py) holds its complete set of partitions; until then panel (b)
    shows the lightweight models only and the script says so.  Re-running this
    script after that job has finished adds DistilBERT; nothing else changes.

Every count (K) printed in a figure is computed from the files, and the script
prints a summary of what each mark is based on.

Run from artifact/eswa_study/revision/:
    python make_r4_revised_figures.py [--allow-partial-distilbert-era]
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from rev_common import FIGURES, RESULTS, STUDY

sys.path.insert(0, str(STUDY))
from make_figures import COLORS, GRID, configure_style, finish  # noqa: E402

FAMILIES = ["Word TF-IDF", "MiniLM", "DistilBERT"]  # family names used in results/
COLOR_KEY = {"Word TF-IDF": "Word TF--IDF", "MiniLM": "MiniLM", "DistilBERT": "DistilBERT"}
TRAININGS = ["clean", "augmented"]
TRAINING_LABEL = {"clean": "clean-only", "augmented": "augmented"}

LIGHT_METRIC_FILES = [
    "revision_r3_multisplit_metrics.csv",  # 20 partitions (round 3)
    "revision_r4_multisplit100_metrics.csv",  # 80 further partitions (round 4)
]
BERT_METRIC_FILES = [
    "revision_distilbert_stream_metrics.csv",  # 3 published partitions
    "revision_r3_distilbert_multisplit_metrics.csv",  # 5 partitions (round 3)
    "revision_r4_distilbert_multisplit_metrics.csv",  # 12 partitions (round 4)
]
LIGHT_MODEL = {
    ("Word TF-IDF", "clean"): "Word LR, clean-only",
    ("Word TF-IDF", "augmented"): "Word LR, linked-augmented",
    ("MiniLM", "clean"): "MiniLM LR, clean-only",
    ("MiniLM", "augmented"): "MiniLM LR, linked-augmented",
}
BERT_TRAINING = {"clean": "clean-only", "augmented": "linked-augmented"}
ERA_FAMILY = {"Word TF-IDF": "word", "MiniLM": "minilm", "DistilBERT": "distilbert"}

TEST_GROUPS = 324  # source groups in every partition's test split (rev_r_r4_inference.py)
DEDUP_THRESHOLD = 0.90
BERT_ERA_FILE = "revision_r4_era_format_distilbert_metrics.csv"
BERT_ERA_INFO = "revision_r4_era_format_distilbert_run_info.json"
# Partitions fine-tuned by rev_u2_era_format_distilbert.py (its SEEDS constant; not
# imported because that module imports torch).  Its run_info file, written when the
# job finishes, takes precedence.
BERT_ERA_SEEDS = [17, 29, 42, 71, 113]

SUMMARY: list[str] = []


def colour(family: str) -> str:
    return COLORS[COLOR_KEY[family]]


def note(text: str) -> None:
    SUMMARY.append(text)
    print(text)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def lightweight_metrics() -> pd.DataFrame:
    frame = pd.concat(
        [pd.read_csv(RESULTS / name) for name in LIGHT_METRIC_FILES], ignore_index=True
    )
    if frame.duplicated(["seed", "model", "condition"]).any():
        raise ValueError("a lightweight partition appears in more than one metrics file")
    return frame


def distilbert_metrics() -> pd.DataFrame:
    columns = ["seed", "training", "condition", "n", "precision", "recall", "f1", "fpr_at_0_5"]
    frame = pd.concat(
        [pd.read_csv(RESULTS / name)[columns] for name in BERT_METRIC_FILES],
        ignore_index=True,
    )
    if frame.duplicated(["seed", "training", "condition"]).any():
        raise ValueError("a DistilBERT partition appears in more than one metrics file")
    return frame


def partition_values(light, bert, family, training, condition, column) -> pd.Series:
    """One value per partition (indexed by partition seed)."""
    if family == "DistilBERT":
        block = bert[
            (bert["training"] == BERT_TRAINING[training]) & (bert["condition"] == condition)
        ]
    else:
        block = light[
            (light["model"] == LIGHT_MODEL[(family, training)])
            & (light["condition"] == condition)
        ]
    values = block.set_index("seed")[column].sort_index().astype(float)
    if values.isna().any() or values.empty:
        raise ValueError(f"missing {column} for {family} {training} {condition}")
    return values


def summarize(values) -> dict:
    values = np.asarray(values, dtype=float)
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "low": float(low),
        "high": float(high),
        "k": int(len(values)),
    }


def era_lightweight() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "revision_r4_era_format_metrics.csv")
    return frame[frame["format"] == "published"]


def era_distilbert(allow_partial: bool) -> tuple[pd.DataFrame | None, str]:
    """DistilBERT era-format rows (published format) if the run is complete enough."""
    path = RESULTS / BERT_ERA_FILE
    if not path.exists():
        return None, f"{BERT_ERA_FILE} not found"
    frame = pd.read_csv(path)
    frame = frame[frame["format"] == "published"]
    expected = list(BERT_ERA_SEEDS)
    info = RESULTS / BERT_ERA_INFO
    finished = info.exists()
    if finished:
        expected = list(json.loads(info.read_text()).get("seeds", expected))
    needed = {"L1_linked", "L2", "L3_human", "nazario"}
    complete = []
    for seed, block in frame.groupby("seed"):
        ok = all(
            set(block.loc[block["training"] == training, "stream"]) >= needed
            for training in TRAININGS
        )
        if ok:
            complete.append(int(seed))
    status = (
        f"{BERT_ERA_FILE}: {len(complete)} of {len(expected)} partitions complete "
        f"(seeds {sorted(complete)}; job {'finished' if finished else 'still running or interrupted'})"
    )
    if set(expected) <= set(complete):
        return frame[frame["seed"].isin(expected)], status
    if allow_partial and len(complete) >= 2:
        return frame[frame["seed"].isin(complete)], status + " -- PARTIAL set used"
    return None, status


# --------------------------------------------------------------------------
# Figure 3: multi-partition inference
# --------------------------------------------------------------------------
SCOPES = [
    ("all splits (hierarchical bootstrap)", "Hierarchical\nbootstrap"),
    ("all splits (paired split-level t)", "Split-level\npaired t"),
    ("all splits (corrected resampled t, Nadeau-Bengio)", "Corrected t\n(Nadeau–Bengio)"),
    ("one further split (t prediction interval)", "Prediction\ninterval (t)"),
    ("one further split (order-statistic interval", "Prediction\n(order stat.)†"),
]


def scope_row(inference: pd.DataFrame, family: str, scope: str) -> pd.Series:
    block = inference[(inference["family"] == family) & inference["scope"].str.startswith(scope)]
    if len(block) != 1:
        raise ValueError(f"expected one row for {family} / {scope}, found {len(block)}")
    return block.iloc[0]


def figure_multisplit(output_dir) -> None:
    inference = pd.read_csv(RESULTS / "revision_r4_inference.csv")
    stabilization = pd.read_csv(RESULTS / "revision_r4_stabilization.csv")
    single = inference[inference["scope"] == "single split (grouped bootstrap)"]

    fig, axes = plt.subplots(
        2, 2, figsize=(7.1, 4.9), gridspec_kw={"height_ratios": [1.12, 1.0]}
    )
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) every partition with its own within-partition interval ----------------
    ax = ax_a
    counts = {}
    for index, family in enumerate(FAMILIES):
        block = single[single["family"] == family].sort_values("delta")
        k = len(block)
        counts[family] = k
        x = index + np.linspace(-0.38, 0.38, k)
        dense = k > 50
        ax.vlines(
            x, block["ci_low"], block["ci_high"], color=colour(family),
            lw=0.45 if dense else 0.85, alpha=0.38 if dense else 0.55, zorder=2,
        )
        ax.scatter(
            x, block["delta"], s=3.0 if dense else 9.0, color=colour(family),
            linewidths=0, zorder=3,
        )
        mean = float(block["delta"].mean())
        ax.hlines(
            mean, index - 0.43, index + 0.43, color="#1B1B1B", lw=1.1, zorder=4,
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")],
        )
        note(
            f"Fig 3a {family}: K={k}, mean delta={mean:.4f}, "
            f"{int((block['ci_low'] <= 0).sum())} partition intervals reach zero"
        )
    ax.axhline(0.0, color="#8A99A2", lw=0.8, zorder=1)
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([f"{family}\nK = {counts[family]}" for family in FAMILIES])
    for label, family in zip(ax.get_xticklabels(), FAMILIES):
        label.set_color(colour(family))
        label.set_fontweight("bold")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(FAMILIES) - 0.45)
    ax.set_ylabel(r"$\Delta$ linked-rewrite F1 (aug. $-$ clean)")
    ax.set_title("(a) Effect on each partition", loc="left")
    ax.text(
        0.02, 0.98,
        "points: partitions, sorted by effect\n"
        "bars: 95% grouped-bootstrap interval within\n"
        f"the partition ({TEST_GROUPS} test groups per partition)\n"
        "black line: mean over partitions",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#3D4A52",
        linespacing=1.25,
    )

    # (b) intervals by scope -----------------------------------------------------
    ax = ax_b
    offsets = {"Word TF-IDF": -0.24, "MiniLM": 0.0, "DistilBERT": 0.24}
    for row, (scope, _) in enumerate(SCOPES):
        for family in FAMILIES:
            entry = scope_row(inference, family, scope)
            y = row + offsets[family]
            ax.plot(
                [entry["ci_low"], entry["ci_high"]], [y, y], color=colour(family),
                lw=2.0, solid_capstyle="butt", zorder=3,
            )
            ax.plot(
                entry["delta"], y, marker="|", color="#1B1B1B", markersize=5.5,
                mew=1.1, zorder=4,
            )
    ax.axvline(0.0, color="#8A99A2", lw=0.8, zorder=1)
    ax.axhline(2.5, color="#B8C2C8", lw=0.7, ls="--", zorder=1)
    ax.set_yticks(range(len(SCOPES)))
    ax.set_yticklabels([label for _, label in SCOPES], fontsize=7.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(SCOPES) - 0.02, -0.62)  # inverted; free strip below the last row
    ax.set_xlabel(r"$\Delta$ linked-rewrite F1 (aug. $-$ clean)")
    ax.set_title("(b) 95% intervals by scope", loc="left")
    rows = ax.get_yaxis_transform()  # x in axes fraction, y in data (row) units
    ax.text(0.985, 2.43, "↑ mean effect", transform=rows, ha="right", va="bottom",
            fontsize=6.5, color="#56636B", style="italic")
    ax.text(0.985, 2.57, "↓ one further partition", transform=rows, ha="right",
            va="top", fontsize=6.5, color="#56636B", style="italic")
    order_cov = {
        family: scope_row(inference, family, SCOPES[-1][0])["scope"].split("coverage ")[-1].rstrip(")")
        for family in FAMILIES
    }
    ax.text(
        0.985, len(SCOPES) - 0.08,
        "† nominal coverage "
        + ", ".join(f"{order_cov[f]} at K = {counts[f]}" for f in ("Word TF-IDF", "DistilBERT")),
        transform=rows, ha="right", va="bottom", fontsize=6.5, color="#56636B",
    )
    handles = [Line2D([], [], color=colour(f), lw=2.0, label=f) for f in FAMILIES]
    handles.append(Line2D([], [], color="#1B1B1B", marker="|", ls="none", ms=5.5, mew=1.1,
                          label="mean effect"))
    ax.legend(handles=handles, loc="upper right", fontsize=6.5, handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.3)

    # (c) stabilization of the between-partition SD ------------------------------
    ax = ax_c
    ax.axhspan(0.9, 1.1, color="#E4E9EC", lw=0, zorder=0)
    ax.axhline(1.0, color="#8A99A2", lw=0.8, zorder=1)
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
        # At K = K_full every random subset is the full set, so its SD equals SD(full).
        ks = np.r_[random.index.to_numpy(), k_full]
        lower = np.r_[random["p05"].to_numpy() / sd_full, 1.0]
        median = np.r_[random["median"].to_numpy() / sd_full, 1.0]
        upper = np.r_[random["p95"].to_numpy() / sd_full, 1.0]
        ax.fill_between(ks, lower, upper, color=colour(family), alpha=0.09, lw=0, zorder=2)
        for edge in (lower, upper):
            ax.plot(ks, edge, color=colour(family), lw=0.6, alpha=0.55, zorder=2)
        ax.plot(ks, median, color=colour(family), lw=1.4, zorder=3)
        ax.plot(
            as_run["k"], as_run["sd"] / sd_full, color=colour(family), lw=1.1, ls=":",
            marker="o", ms=2.3, zorder=4,
        )
        note(
            f"Fig 3c {family}: SD(full)={sd_full:.5f} at K={k_full}; "
            f"as-run SD(K=5)/SD(full)={float(as_run.loc[as_run['k'] == 5, 'sd'].iloc[0]) / sd_full:.2f}"
        )
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["3", "5", "10", "20", "50", "100"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(2.7, 112)
    ax.set_ylim(0, 2.05)
    ax.set_xlabel("Partitions used, K")
    ax.set_ylabel("SD(K) / SD(all partitions)")
    ax.set_title("(c) Stabilization of the between-partition SD", loc="left")
    ax.text(105, 0.885, "±10%", ha="right", va="top", fontsize=6.5, color="#56636B")
    handles = [
        Line2D([], [], color="#56636B", lw=1.4, label="median, 2,000 random orders"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#56636B", alpha=0.16, edgecolor="none",
                      label="5th–95th percentile"),
        Line2D([], [], color="#56636B", lw=1.1, ls=":", marker="o", ms=2.3,
               label="order as run"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.5, handlelength=1.8,
              borderaxespad=0.2, labelspacing=0.3)

    # (d) corrected-t half-width relative to its floor ---------------------------
    ax = ax_d
    ratio = stabilization[stabilization["ordering"] == "as run"].pivot(
        index="k", columns="family", values="corrected_to_floor_ratio"
    )
    shared = ratio.dropna()
    if not np.allclose(shared.to_numpy(), shared.to_numpy()[:, :1]):
        raise ValueError("corrected-to-floor ratio differs between families")
    series = ratio["Word TF-IDF"].dropna().sort_index()
    ax.axhline(1.0, color="#8A99A2", lw=0.8, zorder=1)
    ax.plot(series.index, series.to_numpy(), color="#2F3B42", lw=1.4, marker="o", ms=3.0, zorder=3)
    # values to the upper right of the points: the curve falls to the right, so the
    # labels never sit on it
    for k_mark in (3, 5, 10, 50):
        value = float(series.loc[k_mark])
        ax.text(k_mark * 1.07, value + 0.05, f"{value:.2f}", ha="left", va="bottom",
                fontsize=6.5, color="#2F3B42")
    ax.text(3.0, 1.0 - 0.06, "floor: 1.96 × SD × √(n$_{test}$/n$_{train}$),"
            " n$_{test}$/n$_{train}$ = 0.215", ha="left", va="top", fontsize=6.5,
            color="#56636B")
    for k_mark, text, colour_key, ha, relpos in (
        (20, "DistilBERT\nK = 20: {:.2f}", "DistilBERT", "center", (0.5, 0.0)),
        (100, "lightweight\nK = 100: {:.2f}", None, "right", (1.0, 0.0)),
    ):
        value = float(series.loc[k_mark])
        ax.annotate(
            text.format(value), xy=(k_mark, value), xytext=(k_mark, value + 0.95),
            ha=ha, va="bottom", fontsize=6.5, multialignment=ha,
            color=colour(colour_key) if colour_key else "#2F3B42",
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#8A99A2", "shrinkA": 1,
                        "shrinkB": 4, "relpos": relpos},
        )
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["3", "5", "10", "20", "50", "100"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(2.7, 112)
    ax.set_ylim(0.75, 3.75)
    ax.set_xlabel("Partitions used, K")
    ax.set_ylabel("Corrected-t half-width / floor")
    ax.set_title("(d) Half-width relative to its floor", loc="left")
    ax.text(0.98, 0.97, "same for all three families", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color="#56636B", style="italic")
    note(
        "Fig 3d ratio: "
        + ", ".join(f"K={k}: {series.loc[k]:.2f}" for k in (3, 5, 10, 20, 50, 100))
    )

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.grid(True, color=GRID, lw=0.5, alpha=0.55)
        ax.set_axisbelow(True)
    ax_a.grid(False, axis="x")
    ax_b.grid(False, axis="y")
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    finish(fig, output_dir, "multisplit_inference")


# --------------------------------------------------------------------------
# Figure 6: L2 performance against spam false alarms
# --------------------------------------------------------------------------
def preferable_region(ax) -> None:
    """Soft, label-carrying guide in the lower-right corner (not data)."""
    n = 300
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    distance = np.hypot((1 - xx) / 0.62, yy / 0.50)  # elliptical, centred on the corner
    alpha = 0.30 * np.clip(1 - distance, 0, 1) ** 0.85
    rgba = np.zeros((n, n, 4))
    rgba[..., :3] = to_rgb("#7E9AAB")
    rgba[..., 3] = alpha
    ax.imshow(rgba, extent=(0, 1, 0, 1), transform=ax.transAxes, origin="lower",
              aspect="auto", zorder=0, interpolation="bilinear")
    ax.text(0.975, 0.03, "preferable: higher L2 F1,\nfewer spam false alarms",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color="#4F6675", style="italic", zorder=1, linespacing=1.2)


def figure_tradeoff(output_dir) -> None:
    light, bert = lightweight_metrics(), distilbert_metrics()
    fig, ax = plt.subplots(figsize=(3.58, 3.2))  # tight bbox ~ one column (3.48 in)
    preferable_region(ax)
    label_spec = {
        # family: (anchor x, anchor y, ha, va) in data units
        "Word TF-IDF": (0.9335, 0.0605, "center", "bottom"),
        "MiniLM": (0.834, 0.0475, "center", "top"),
        "DistilBERT": (0.8355, 0.113, "right", "bottom"),
    }
    for family in FAMILIES:
        points = {}
        for training in TRAININGS:
            f1 = partition_values(light, bert, family, training, "external_openai", "f1")
            fpr = partition_values(light, bert, family, training, "spam_hard_negative", "fpr_at_0_5")
            if not f1.index.equals(fpr.index):
                raise ValueError(f"L2 and spam partitions differ for {family} {training}")
            points[training] = (summarize(f1), summarize(fpr))
        k = points["clean"][0]["k"]
        if points["augmented"][0]["k"] != k:
            raise ValueError(f"clean and augmented partition counts differ for {family}")
        c = colour(family)
        clean_xy = (points["clean"][0]["mean"], points["clean"][1]["mean"])
        aug_xy = (points["augmented"][0]["mean"], points["augmented"][1]["mean"])
        for training in TRAININGS:
            f1s, fprs = points[training]
            ax.errorbar(
                f1s["mean"], fprs["mean"], xerr=f1s["sd"], yerr=fprs["sd"], fmt="none",
                ecolor=c, elinewidth=0.8, capsize=1.8, capthick=0.8, alpha=0.75, zorder=2,
            )
        ax.annotate(
            "", xy=aug_xy, xytext=clean_xy,
            arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.5, "mutation_scale": 9,
                        "shrinkA": 3.6, "shrinkB": 3.8},
            zorder=3,
        )
        ax.plot(*clean_xy, "o", ms=5.6, mfc="white", mec=c, mew=1.4, zorder=4)
        ax.plot(*aug_xy, "o", ms=5.6, mfc=c, mec=c, mew=1.4, zorder=4)
        x, y, ha, va = label_spec[family]
        ax.text(x, y, f"{family}\nK = {k}", color=c, fontsize=7.0, fontweight="bold",
                ha=ha, va=va, zorder=5, linespacing=1.15)
        note(
            f"Fig 6 {family} (K={k}): clean F1 {clean_xy[0]:.3f}+/-{points['clean'][0]['sd']:.3f}, "
            f"FPR {clean_xy[1]:.3f}+/-{points['clean'][1]['sd']:.3f}; augmented F1 "
            f"{aug_xy[0]:.3f}+/-{points['augmented'][0]['sd']:.3f}, FPR "
            f"{aug_xy[1]:.3f}+/-{points['augmented'][1]['sd']:.3f}"
        )

    ax.set_xlim(0.790, 0.962)
    ax.set_ylim(0.0, 0.172)
    ax.set_xlabel("Generator-disjoint (L2) F1 at threshold 0.5")
    ax.set_ylabel("Spam hard-negative FPR at threshold 0.5")
    ax.grid(color=GRID, linewidth=0.5, linestyle=":", zorder=0)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=5.2, mfc="white", mec="#52616B", mew=1.3,
               label="clean-only (mean)"),
        Line2D([], [], marker="o", ls="none", ms=5.2, mfc="#52616B", mec="#52616B",
               label="augmented (mean)"),
        Line2D([], [], color="#52616B", lw=1.4, marker=">", ms=4, markevery=[1],
               label="arrow: clean-only → augmented"),
        Line2D([], [], color="#52616B", lw=0.8, alpha=0.75, label="whiskers: ±1 SD"),
    ]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=6.5, handlelength=1.6,
                       borderaxespad=0.3, labelspacing=0.35,
                       title="K = refitted partitions", title_fontsize=6.5)
    legend._legend_box.align = "left"
    fig.tight_layout()
    finish(fig, output_dir, "tradeoff")


# --------------------------------------------------------------------------
# Figure 7: F1 and recall by evaluation level
# --------------------------------------------------------------------------
MODELS = [(family, training) for family in FAMILIES for training in TRAININGS]


def model_offsets(models, step=0.105):
    centre = (len(models) - 1) / 2
    return {model: (i - centre) * step for i, model in enumerate(models)}


def draw_interval(ax, x, stats, family, training, marker="o", ms=4.0):
    c = colour(family)
    ax.errorbar(
        x, stats["mean"],
        yerr=[[stats["mean"] - stats["low"]], [stats["high"] - stats["mean"]]],
        fmt="none", ecolor=c, elinewidth=0.9, capsize=1.6, capthick=0.8, zorder=3,
    )
    ax.plot(x, stats["mean"], marker=marker, ls="none", ms=ms, mec=c, mew=1.1,
            mfc="white" if training == "clean" else c, zorder=4)


def f1_levels(light, bert) -> dict:
    """Per (family, training): summaries at L1, L2, L2-dedup (lightweight only), L3."""
    era = era_lightweight()
    redundancy = pd.read_csv(RESULTS / "revision_r3_redundancy_sensitivity.csv")
    redundancy = redundancy[np.isclose(redundancy["linkage_threshold"], DEDUP_THRESHOLD)]
    out = {}
    for family, training in MODELS:
        entry = {
            "L1": summarize(partition_values(light, bert, family, training, "linked_rewrites", "f1")),
            "L2": summarize(partition_values(light, bert, family, training, "external_openai", "f1")),
        }
        if family == "DistilBERT":
            entry["L3"] = summarize(
                partition_values(light, bert, family, training, "human_phishing", "f1")
            )
        else:
            block = era[
                (era["family"] == ERA_FAMILY[family]) & (era["training"] == training)
                & (era["stream"] == "L3_human")
            ]
            entry["L3"] = summarize(block["f1"])
            row = redundancy[redundancy["family"] == family]
            if len(row) != 1:
                raise ValueError(f"no redundancy row for {family}")
            column = "dedup_f1_clean" if training == "clean" else "dedup_f1_augmented"
            # Mean over the five published partitions (rev_m_provenance.py); the
            # per-partition values were not stored, so no interval can be drawn.
            entry["L2dedup"] = {"mean": float(row[column].iloc[0]), "k": 5}
        out[(family, training)] = entry
    return out


RECALL_LEVELS = [
    ("L1_linked", "recall_t0.5", "L1\nlinked\nrewrites"),
    ("L2", "recall_t0.5", "L2\ngen.-\ndisjoint"),
    ("L3_human", "recall_t0.5", "L3\nhuman\n(2000s)"),
    ("nazario", "recall_t0.5_2015-2022", "Nazario\n2015–2022"),
    ("nazario", "recall_t0.5_2023-2025", "Nazario\n2023–2025"),
]


def recall_levels(era_light, era_bert) -> dict:
    out = {}
    for family, training in MODELS:
        if family == "DistilBERT":
            if era_bert is None:
                continue
            frame = era_bert
        else:
            frame = era_light
        entry = []
        for stream, column, _ in RECALL_LEVELS:
            block = frame[
                (frame["family"] == ERA_FAMILY[family]) & (frame["training"] == training)
                & (frame["stream"] == stream)
            ]
            entry.append(summarize(block[column]))
        out[(family, training)] = entry
    return out


def figure_evaluation_ladder(output_dir, allow_partial: bool) -> None:
    light, bert = lightweight_metrics(), distilbert_metrics()
    f1 = f1_levels(light, bert)
    era_bert, status = era_distilbert(allow_partial)
    note(status)
    recall = recall_levels(era_lightweight(), era_bert)
    if era_bert is None:
        note("Fig 7b: DistilBERT OMITTED from panel (b) -- re-run this script when "
             "rev_u2_era_format_distilbert.py has finished.")

    fig = plt.figure(figsize=(7.1, 3.35))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.06], wspace=0.2,
                             left=0.072, right=0.995, top=0.927, bottom=0.275)
    left = outer[0].subgridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.07)
    ax_top = fig.add_subplot(left[0])
    ax_bottom = fig.add_subplot(left[1], sharex=ax_top)
    ax_b = fig.add_subplot(outer[1])

    # (a) F1 at threshold 0.5, broken vertical axis ------------------------------
    levels = ["L1", "L2", "L2dedup", "L3"]
    offsets = model_offsets(MODELS)
    for ax in (ax_top, ax_bottom):
        for position, level in enumerate(levels):
            for family, training in MODELS:
                stats = f1[(family, training)].get(level)
                if stats is None:
                    continue
                x = position + offsets[(family, training)]
                if level == "L2dedup":
                    c = colour(family)
                    ax.plot(x, stats["mean"], marker="X", ls="none", ms=5.2, mec=c, mew=1.0,
                            mfc="white" if training == "clean" else c, zorder=4)
                else:
                    draw_interval(ax, x, stats, family, training)
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for boundary in (0.5, 1.5, 2.5):
            ax.axvline(boundary, color="#E1E6E9", lw=0.7, zorder=0)
    ax_top.set_ylim(0.74, 1.0)
    ax_bottom.set_ylim(0.0, 0.48)
    ax_top.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax_bottom.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    slash = {"marker": [(-1, -0.55), (1, 0.55)], "markersize": 7, "linestyle": "none",
             "color": "#1B1B1B", "mec": "#1B1B1B", "mew": 0.8, "clip_on": False}
    ax_top.plot([0], [0], transform=ax_top.transAxes, **slash)
    ax_bottom.plot([0], [1], transform=ax_bottom.transAxes, **slash)
    k_light = {lvl: f1[("Word TF-IDF", "clean")][lvl]["k"] for lvl in levels}
    k_bert = {lvl: f1[("DistilBERT", "clean")].get(lvl, {}).get("k") for lvl in levels}
    for model in MODELS:  # every model of a family must have the same K at a level
        reference = k_bert if model[0] == "DistilBERT" else k_light
        for lvl in levels:
            if model in f1 and lvl in f1[model] and f1[model][lvl]["k"] != reference[lvl]:
                raise ValueError(f"K differs within a level: {model} {lvl}")
    names = {"L1": "L1\nlinked\nrewrites", "L2": "L2\ngen.-\ndisjoint",
             "L2dedup": "L2\ndedup.\n(cos. 0.90)", "L3": "L3\nhuman\n(2000s)"}
    dash = "–"
    ax_bottom.set_xticks(range(len(levels)))
    ax_bottom.set_xticklabels(
        [f"{names[lvl]}\nK = {k_light[lvl]} / {k_bert[lvl] if k_bert[lvl] else dash}"
         for lvl in levels],
        fontsize=7.0,
    )
    ax_bottom.tick_params(axis="x", length=0)
    ax_bottom.set_xlim(-0.5, len(levels) - 0.5)
    ax_top.set_title("(a) F1 at threshold 0.5", loc="left")
    ax_bottom.text(0.015, 0.95,
                   "K = partitions: lightweight / DistilBERT\n"
                   "vertical axis broken between 0.48 and 0.74",
                   transform=ax_bottom.transAxes, ha="left", va="top", fontsize=6.5,
                   color="#56636B", linespacing=1.3)
    top_box, bottom_box = ax_top.get_position(), ax_bottom.get_position()
    fig.text(0.008, (top_box.y1 + bottom_box.y0) / 2, "F1 at threshold 0.5", rotation=90,
             ha="left", va="center", fontsize=8.5)
    for (family, training), entry in f1.items():
        note(
            f"Fig 7a {family} {training}: "
            + "; ".join(
                f"{lvl} {entry[lvl]['mean']:.3f}"
                + (f" [{entry[lvl]['low']:.3f}, {entry[lvl]['high']:.3f}]" if "low" in entry[lvl] else "")
                + f" K={entry[lvl]['k']}"
                for lvl in levels if lvl in entry
            )
        )

    # (b) recall at threshold 0.5 -------------------------------------------------
    ax = ax_b
    present = [model for model in MODELS if model in recall]
    offsets_b = model_offsets(present, step=0.12 if len(present) == 6 else 0.14)
    for position in range(len(RECALL_LEVELS)):
        for model in present:
            draw_interval(ax, position + offsets_b[model], recall[model][position], *model)
    for boundary in (0.5, 1.5, 2.5, 3.5):
        ax.axvline(boundary, color="#E1E6E9", lw=0.7, zorder=0)
    ax.axvspan(2.5, 4.5, color="#F1F4F6", lw=0, zorder=0)
    ax.text(3.5, 0.035, "era-matched real phishing\n(all messages are phishing)",
            ha="center", va="bottom", fontsize=6.5, color="#56636B", style="italic")
    k_b_light = {recall[("Word TF-IDF", "clean")][i]["k"] for i in range(len(RECALL_LEVELS))}
    if len(k_b_light) != 1:
        raise ValueError("lightweight K differs between recall levels")
    k_b_light = k_b_light.pop()
    k_b_bert = recall[("DistilBERT", "clean")][0]["k"] if ("DistilBERT", "clean") in recall else None
    ax.set_xticks(range(len(RECALL_LEVELS)))
    ax.set_xticklabels([label for _, _, label in RECALL_LEVELS], fontsize=7.0)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(RECALL_LEVELS) - 0.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Recall at threshold 0.5")
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(b) Recall at threshold 0.5", loc="left")
    k_text = (f"K = {k_b_light} partitions (lightweight)\nK = {k_b_bert} (DistilBERT)"
              if k_b_bert else f"K = {k_b_light} partitions\n(lightweight models)")
    ax.text(0.015, 0.36, k_text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color="#56636B", linespacing=1.3)
    for model, entry in recall.items():
        note(
            f"Fig 7b {model[0]} {model[1]}: "
            + "; ".join(
                f"{RECALL_LEVELS[i][0]}:{RECALL_LEVELS[i][1]} {s['mean']:.3f} "
                f"[{s['low']:.3f}, {s['high']:.3f}] K={s['k']}"
                for i, s in enumerate(entry)
            )
        )

    # shared legend ------------------------------------------------------------
    handles = []
    for family in FAMILIES:
        for training in TRAININGS:
            c = colour(family)
            handles.append(Line2D([], [], marker="o", ls="none", ms=4.6, mec=c, mew=1.1,
                                  mfc="white" if training == "clean" else c,
                                  label=f"{family}, {TRAINING_LABEL[training]}"))
    handles.append(Line2D([], [], marker="X", ls="none", ms=5.2, mec="#52616B", mew=1.0,
                          mfc="#52616B",
                          label="L2 dedup.: mean of 5 partitions, no interval"))
    handles.append(Line2D([], [], color="#52616B", lw=0.9, marker="_", ms=5, mew=0.8,
                          label="2.5th–97.5th percentile across partitions"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               fontsize=6.6, handletextpad=0.4, columnspacing=1.0, labelspacing=0.45)
    finish(fig, output_dir, "evaluation_ladder")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--allow-partial-distilbert-era", action="store_true",
        help="draw DistilBERT in Figure 7(b) from an incomplete era-format run (>= 2 partitions)",
    )
    args = parser.parse_args()
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_multisplit(FIGURES)
    figure_tradeoff(FIGURES)
    figure_evaluation_ladder(FIGURES, args.allow_partial_distilbert_era)
    print(f"wrote multisplit_inference, tradeoff, evaluation_ladder (.pdf, .png) to {FIGURES}")


if __name__ == "__main__":
    main()
