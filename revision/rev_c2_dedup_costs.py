#!/usr/bin/env python3
"""Human-authored cost analyses on a stream without contradictory duplicates (round 4).

``rev_t3_l3_sa_overlap.py`` shows that the L3 aggregate contains the SpamAssassin
corpus, so the round-3 human-authored cost stream -- the aggregate plus every
SpamAssassin message as a negative -- held about 640 messages twice with
contradictory labels and about 3,650 legitimate messages twice. This script
removes the aggregate's SpamAssassin duplicates (``revision_r4_l3_sa_duplicates.csv``)
and recomputes, for the human-authored stream only, the quantities behind the
round-3 tables that use it: expected cost by rule (Table "cost"), the
operating-region map and pi*r purity, prevalence misspecification, and the
operational value and break-even ratios. L1 and L2 streams are unaffected and
are not recomputed. Outputs never overwrite the round-3 files.

Outputs (results/):
  revision_r4_threshold_cost_dedup_summary.csv
  revision_r4_regions_dedup.csv, revision_r4_miss_cost_boundaries_dedup.csv
  revision_r4_misspec_dedup.csv
  revision_r4_operational_dedup.csv, revision_r4_break_even_dedup.csv
  revision_r4_stream_ceiling_dedup.csv
  revision_r4_threshold_cost_curves_dedup.csv  seed-42 curves for the threshold figure
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rev_c_thresholds import (
    COST_RATIOS,
    GRID,
    PREVALENCES,
    build_streams,
    cost,
    flagged_fraction_threshold,
    rates,
    threshold_for_recall,
)
from rev_common import RESULTS
import rev_o_deployment as deployment

STREAM = "human_authored"


def load_dedup():
    duplicates = pd.read_csv(RESULTS / "revision_r4_l3_sa_duplicates.csv")
    drop = set(duplicates.loc[duplicates["duplicate"], "row_index"].astype(int))
    lightweight = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    realworld = pd.read_csv(RESULTS / "revision_realworld_probabilities.csv")
    transformer = pd.read_csv(RESULTS / "revision_distilbert_stream_probabilities.csv")
    for name, frame in (("realworld", realworld), ("transformer", transformer)):
        mask = (frame["condition"] == "human_phishing") & frame["row_index"].isin(drop)
        if name == "realworld":
            realworld = frame[~mask]
        else:
            transformer = frame[~mask]
    store = build_streams(lightweight, realworld, transformer)
    return {key: {"validation": value["validation"], STREAM: value[STREAM]}
            for key, value in store.items() if STREAM in value}, len(drop)


def cost_table(store):
    rows = []
    for (model, seed), streams in store.items():
        vy, vp = streams["validation"]
        vtpr, vfpr = rates(vy, vp, GRID)
        recall_threshold = threshold_for_recall(vy, vp)
        y, p = streams[STREAM]
        tpr, fpr = rates(y, p, GRID)
        for prevalence in PREVALENCES:
            for ratio in COST_RATIOS:
                t_cost = float(GRID[int(np.argmin(cost(vtpr, vfpr, prevalence, ratio)))])
                rate = float((vp >= t_cost).mean())
                target = cost(tpr, fpr, prevalence, ratio)
                oracle = int(np.argmin(target))
                for rule, threshold in (("fixed_0.5", 0.5), ("fixed_95_recall", recall_threshold),
                                        ("fixed_cost_min", t_cost),
                                        ("dynamic_rate_match", flagged_fraction_threshold(p, rate)),
                                        ("oracle_cost_min", float(GRID[oracle]))):
                    flagged = p >= threshold
                    realized = (1 - prevalence) * flagged[y == 0].mean() + prevalence * ratio * (1 - flagged[y == 1].mean())
                    rows.append({"model": model, "seed": seed, "stream": STREAM, "prevalence": prevalence,
                                 "cost_ratio": ratio, "rule": rule, "threshold": threshold,
                                 "recall": float(flagged[y == 1].mean()), "fpr": float(flagged[y == 0].mean()),
                                 "expected_cost": float(realized), "regret": float(realized - target[oracle])})
    detail = pd.DataFrame(rows)
    return detail.groupby(["model", "stream", "prevalence", "cost_ratio", "rule"], as_index=False)[
        ["threshold", "recall", "fpr", "expected_cost", "regret"]].mean()


def curve_table(store, seed=42, model="word_lr_linked_augmented", ratio=10.0):
    """Seed-42 expected-cost curves (the threshold figure) on the deduplicated stream."""
    y, p = store[(model, seed)][STREAM]
    tpr, fpr = rates(y, p, GRID)
    keep = np.linspace(0, len(GRID) - 1, 201).astype(int)
    frames = []
    for prevalence in PREVALENCES:
        frames.append(pd.DataFrame({"model": model, "stream": STREAM, "prevalence": prevalence,
                                    "cost_ratio": ratio, "threshold": GRID[keep],
                                    "expected_cost": cost(tpr, fpr, prevalence, ratio)[keep]}))
    return pd.concat(frames, ignore_index=True)


def ceiling_table(store):
    rows = []
    for (model, seed), streams in store.items():
        y, p = streams[STREAM]
        order = np.argsort(-p)
        tp = np.cumsum(y[order])
        f1 = 2 * tp / (np.arange(1, len(y) + 1) + y.sum())
        rows.append({"model": model, "seed": seed, "prevalence_in_stream": float(y.mean()),
                     "f1_at_0_5": float(2 * ((p >= 0.5) & (y == 1)).sum() / ((p >= 0.5).sum() + y.sum())),
                     "f1_at_oracle_threshold": float(f1.max())})
    return pd.DataFrame(rows).groupby("model", as_index=False).mean(numeric_only=True)


def main() -> None:
    store, dropped = load_dedup()
    print(f"dropped {dropped} SpamAssassin duplicates from the L3 aggregate; {len(store)} model-partition entries")
    summary = cost_table(store)
    summary.to_csv(RESULTS / "revision_r4_threshold_cost_dedup_summary.csv", index=False)
    regions = deployment.operating_regions(store)
    regions.to_csv(RESULTS / "revision_r4_regions_dedup.csv", index=False)
    deployment.miss_cost_boundaries(regions).to_csv(RESULTS / "revision_r4_miss_cost_boundaries_dedup.csv", index=False)
    deployment.prevalence_misspecification(store).to_csv(RESULTS / "revision_r4_misspec_dedup.csv", index=False)
    value = deployment.operational_value(store)
    value.to_csv(RESULTS / "revision_r4_operational_dedup.csv", index=False)
    deployment.break_even_summary(value).to_csv(RESULTS / "revision_r4_break_even_dedup.csv", index=False)
    ceiling_table(store).to_csv(RESULTS / "revision_r4_stream_ceiling_dedup.csv", index=False)
    curve_table(store).to_csv(RESULTS / "revision_r4_threshold_cost_curves_dedup.csv", index=False)

    # side-by-side with the round-3 values for the human-authored stream
    r3 = pd.read_csv(RESULTS / "revision_threshold_cost_summary.csv")
    key = ["model", "stream", "prevalence", "cost_ratio", "rule"]
    both = summary.merge(r3[key + ["expected_cost"]], on=key, suffixes=("_dedup", "_r3"))
    view = both[(both["model"] == "word_lr_linked_augmented") & (both["cost_ratio"] == 10.0)]
    pd.set_option("display.width", 200)
    print(view[["prevalence", "rule", "expected_cost_r3", "expected_cost_dedup"]].round(4).to_string(index=False))
    boundaries = pd.read_csv(RESULTS / "revision_r4_miss_cost_boundaries_dedup.csv")
    print(boundaries[boundaries["model"] == "word_lr_linked_augmented"].round(3).to_string(index=False))
    r3_be = pd.read_csv(RESULTS / "revision_r3_break_even.csv")
    be = pd.read_csv(RESULTS / "revision_r4_break_even_dedup.csv")
    cols = ["family", "prevalence", "delta_missed_per_million", "delta_false_alarms_per_million",
            "delta_expected_cost", "break_even_cost_ratio", "break_even_direction"]
    print(r3_be[(r3_be["stream"] == STREAM) & (r3_be["prevalence"] == 0.01)][cols].round(4).to_string(index=False))
    print(be[be["prevalence"] == 0.01][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
