#!/usr/bin/env python3
"""Deployment-oriented operating points and the operational value of augmentation.

Round-3 comments 3 and 5. Comment 5 asks that threshold selection be treated as
a deployment decision across phishing prevalences and asymmetric error costs
rather than as a single fixed recall target. Comment 3 asks whether the observed
augmentation gains justify their computational cost and, for DistilBERT, the
deterioration in generator-disjoint calibration and precision.

Three analyses:

1. Operating-region map.  For a 7 x 7 grid of prevalence and cost ratio, which
   *implementable* rule (threshold 0.5, 95%-recall, validation cost-minimizing,
   label-free rate matching) minimizes expected cost on each stream, and what
   does it cost relative to the unattainable target oracle?

2. Prevalence-misspecification surface.  A deployment must estimate prevalence
   before it can select a cost-minimizing threshold. The threshold chosen under
   a believed prevalence is deployed at a different true prevalence, and the
   excess cost over the correctly specified choice is recorded. This says how
   accurate the prevalence estimate has to be.

3. Operational value of augmentation.  Expected cost is converted into counts
   per million delivered messages -- phishing missed and legitimate messages
   flagged -- at each rule's operating point, for the clean and augmented member
   of each pair, together with the fitting time each pair costs. This states the
   augmentation decision in units a mail-security operator can act on, including
   the cases where the augmented model is the worse deployment.

Outputs (results/):
  revision_r3_operating_regions.csv
  revision_r3_prevalence_misspecification.csv
  revision_r3_operational_value.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rev_common import RESULTS
from rev_c_thresholds import (
    GRID,
    build_streams,
    cost,
    flagged_fraction_threshold,
    rates,
    threshold_for_recall,
)

PREVALENCES = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
COST_RATIOS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
IMPLEMENTABLE = ["fixed_0.5", "fixed_95_recall", "fixed_cost_min", "dynamic_rate_match"]
BELIEVED = [0.001, 0.005, 0.01, 0.05]
PAIRS = [
    ("Word TF-IDF", "word_lr_clean_only", "word_lr_linked_augmented", 0.48, 3.18),
    ("MiniLM", "minilm_lr_clean_only", "minilm_lr_linked_augmented", 0.04, 0.33),
    ("DistilBERT", "distilbert_clean_only", "distilbert_linked_augmented", 113.0, 733.0),
]
STREAM_LABEL = {
    "primary_pipeline": "Primary (L1)",
    "pipeline_disjoint": "Generator-disjoint (L2)",
    "human_authored": "Human-authored (L3)",
}


def rule_thresholds(streams, prevalence, ratio):
    validation_y, validation_p = streams["validation"]
    validation_tpr, validation_fpr = rates(validation_y, validation_p, GRID)
    validation_cost = cost(validation_tpr, validation_fpr, prevalence, ratio)
    fixed_cost_threshold = float(GRID[int(np.argmin(validation_cost))])
    validation_rate = float((validation_p >= fixed_cost_threshold).mean())
    return {
        "fixed_0.5": lambda probability: 0.5,
        "fixed_95_recall": lambda probability: threshold_for_recall(
            validation_y, validation_p
        ),
        "fixed_cost_min": lambda probability: fixed_cost_threshold,
        "dynamic_rate_match": lambda probability: flagged_fraction_threshold(
            probability, validation_rate
        ),
    }


def realized_cost(y, probability, threshold, prevalence, ratio):
    predicted = probability >= threshold
    tpr = float(predicted[y == 1].mean()) if (y == 1).any() else np.nan
    fpr = float(predicted[y == 0].mean()) if (y == 0).any() else np.nan
    return (1 - prevalence) * fpr + prevalence * ratio * (1 - tpr), tpr, fpr


def operating_regions(store):
    rows = []
    for (model, seed), streams in store.items():
        for prevalence in PREVALENCES:
            for ratio in COST_RATIOS:
                thresholds = rule_thresholds(streams, prevalence, ratio)
                for stream, (y, probability) in streams.items():
                    if stream == "validation":
                        continue
                    tpr, fpr = rates(y, probability, GRID)
                    target_cost = cost(tpr, fpr, prevalence, ratio)
                    oracle = float(target_cost.min())
                    best_rule, best_cost, best_threshold = None, np.inf, np.nan
                    record = {
                        "model": model,
                        "seed": seed,
                        "stream": stream,
                        "prevalence": prevalence,
                        "cost_ratio": ratio,
                        "oracle_cost": oracle,
                        "cost_flag_nothing": prevalence * ratio,
                    }
                    for rule in IMPLEMENTABLE:
                        threshold = thresholds[rule](probability)
                        value, _, _ = realized_cost(
                            y, probability, threshold, prevalence, ratio
                        )
                        record[f"cost_{rule}"] = value
                        if value < best_cost:
                            best_rule, best_cost, best_threshold = rule, value, threshold
                    record["best_rule"] = best_rule
                    record["best_cost"] = best_cost
                    record["best_threshold"] = best_threshold
                    record["best_regret"] = best_cost - oracle
                    record["beats_flag_nothing"] = bool(best_cost < prevalence * ratio)
                    rows.append(record)
    return pd.DataFrame(rows)


def prevalence_misspecification(store):
    """Cost of selecting a threshold for the wrong prevalence."""
    rows = []
    ratio = 10.0
    for (model, seed), streams in store.items():
        validation_y, validation_p = streams["validation"]
        validation_tpr, validation_fpr = rates(validation_y, validation_p, GRID)
        chosen = {}
        for believed in BELIEVED:
            values = cost(validation_tpr, validation_fpr, believed, ratio)
            chosen[believed] = float(GRID[int(np.argmin(values))])
        for stream, (y, probability) in streams.items():
            if stream == "validation":
                continue
            for true_prevalence in BELIEVED:
                reference, _, _ = realized_cost(
                    y, probability, chosen[true_prevalence], true_prevalence, ratio
                )
                for believed in BELIEVED:
                    value, tpr, fpr = realized_cost(
                        y, probability, chosen[believed], true_prevalence, ratio
                    )
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "stream": stream,
                            "cost_ratio": ratio,
                            "believed_prevalence": believed,
                            "true_prevalence": true_prevalence,
                            "threshold": chosen[believed],
                            "expected_cost": value,
                            "matched_cost": reference,
                            "excess_cost": value - reference,
                            "excess_ratio": value / reference if reference > 0 else np.nan,
                            "recall": tpr,
                            "fpr": fpr,
                        }
                    )
    return pd.DataFrame(rows)


def operational_value(store):
    """Missed phishing and false alarms per million delivered messages."""
    rows = []
    million = 1_000_000
    for family, clean_model, augmented_model, clean_fit, augmented_fit in PAIRS:
        seeds = sorted(
            {seed for (model, seed) in store if model in (clean_model, augmented_model)}
        )
        for seed in seeds:
            if (clean_model, seed) not in store or (augmented_model, seed) not in store:
                continue
            for prevalence in [0.001, 0.01, 0.05]:
                for ratio in [10.0]:
                    for stream in ("primary_pipeline", "pipeline_disjoint", "human_authored"):
                        record = {
                            "family": family,
                            "seed": seed,
                            "stream": stream,
                            "prevalence": prevalence,
                            "cost_ratio": ratio,
                            "added_fit_seconds": augmented_fit - clean_fit,
                        }
                        usable = True
                        for tag, model in (("clean", clean_model), ("augmented", augmented_model)):
                            streams = store[(model, seed)]
                            if stream not in streams:
                                usable = False
                                break
                            y, probability = streams[stream]
                            thresholds = rule_thresholds(streams, prevalence, ratio)
                            best = min(
                                (
                                    (
                                        realized_cost(
                                            y,
                                            probability,
                                            thresholds[rule](probability),
                                            prevalence,
                                            ratio,
                                        )[0],
                                        rule,
                                    )
                                    for rule in IMPLEMENTABLE
                                ),
                            )
                            value, rule = best
                            threshold = thresholds[rule](probability)
                            _, tpr, fpr = realized_cost(
                                y, probability, threshold, prevalence, ratio
                            )
                            record[f"{tag}_rule"] = rule
                            record[f"{tag}_threshold"] = threshold
                            record[f"{tag}_recall"] = tpr
                            record[f"{tag}_fpr"] = fpr
                            record[f"{tag}_expected_cost"] = value
                            record[f"{tag}_missed_per_million"] = (
                                million * prevalence * (1 - tpr)
                            )
                            record[f"{tag}_false_alarms_per_million"] = (
                                million * (1 - prevalence) * fpr
                            )
                        if not usable:
                            continue
                        record["delta_expected_cost"] = (
                            record["augmented_expected_cost"] - record["clean_expected_cost"]
                        )
                        record["delta_missed_per_million"] = (
                            record["augmented_missed_per_million"]
                            - record["clean_missed_per_million"]
                        )
                        record["delta_false_alarms_per_million"] = (
                            record["augmented_false_alarms_per_million"]
                            - record["clean_false_alarms_per_million"]
                        )
                        # Break-even cost ratio. Per million delivered
                        # messages the cost difference is
                        #     delta_false_alarms + r * delta_missed,
                        # so the sign pattern of the two deltas determines both
                        # the break-even ratio and which side of it favours
                        # augmentation.
                        missed = record["delta_missed_per_million"]
                        alarms = record["delta_false_alarms_per_million"]
                        if missed < 0 and alarms <= 0:
                            break_even = 0.0
                            direction = "augmented cheaper at every r"
                        elif missed >= 0 and alarms > 0:
                            break_even = np.inf
                            direction = "augmented never cheaper"
                        elif missed < 0:
                            break_even = -alarms / missed
                            direction = "augmented cheaper above"
                        else:
                            break_even = -alarms / missed
                            direction = "augmented cheaper below"
                        record["break_even_cost_ratio"] = float(break_even)
                        record["break_even_direction"] = direction
                        rows.append(record)
    return pd.DataFrame(rows)


def miss_cost_boundaries(regions: pd.DataFrame) -> pd.DataFrame:
    """Is the winning rule governed by prevalence and cost ratio separately, or
    only by their product?

    ``pi * r`` is the expected cost of flagging nothing, i.e. the price of a
    miss per delivered message. If the winning rule is a function of that
    product alone, a deployment needs only one number rather than two.
    """
    regions = regions.copy()
    regions["miss_cost"] = regions["prevalence"] * regions["cost_ratio"]
    rows = []
    for (model, stream), block in regions.groupby(["model", "stream"]):
        winner = (
            block.groupby(["prevalence", "cost_ratio", "miss_cost"])["best_rule"]
            .agg(lambda s: s.value_counts().idxmax())
            .reset_index()
            .sort_values("miss_cost")
        )
        # Purity of a product-only rule: the share of (pi, r) cells whose winner
        # equals the majority winner within its miss-cost decile.
        winner["decile"] = pd.qcut(
            winner["miss_cost"].rank(method="first"), 10, labels=False
        )
        majority = winner.groupby("decile")["best_rule"].transform(
            lambda s: s.value_counts().idxmax()
        )
        purity = float((winner["best_rule"] == majority).mean())
        for rule, part in winner.groupby("best_rule"):
            rows.append(
                {
                    "model": model,
                    "stream": stream,
                    "rule": rule,
                    "cells": int(len(part)),
                    "miss_cost_min": float(part["miss_cost"].min()),
                    "miss_cost_median": float(part["miss_cost"].median()),
                    "miss_cost_max": float(part["miss_cost"].max()),
                    "product_only_purity": purity,
                }
            )
    return pd.DataFrame(rows)


def break_even_summary(value: pd.DataFrame) -> pd.DataFrame:
    """Aggregate break-even ratio, computed from mean deltas.

    Averaging per-partition break-even ratios is not meaningful when some
    partitions improve both error types (ratio zero) and others improve neither
    (ratio infinite), so the summary recomputes the ratio from the mean error
    deltas instead.
    """
    grouped = value.groupby(["family", "stream", "prevalence", "cost_ratio"]).mean(
        numeric_only=True
    )
    rows = []
    for (family, stream, prevalence, ratio), row in grouped.iterrows():
        missed = row["delta_missed_per_million"]
        alarms = row["delta_false_alarms_per_million"]
        if missed < 0 and alarms <= 0:
            break_even, direction = 0.0, "augmented cheaper at every r"
        elif missed >= 0 and alarms > 0:
            break_even, direction = np.inf, "augmented never cheaper"
        elif missed < 0:
            break_even, direction = -alarms / missed, "augmented cheaper above"
        else:
            break_even, direction = -alarms / missed, "augmented cheaper below"
        rows.append(
            {
                "family": family,
                "stream": stream,
                "prevalence": prevalence,
                "cost_ratio": ratio,
                "clean_expected_cost": row["clean_expected_cost"],
                "augmented_expected_cost": row["augmented_expected_cost"],
                "delta_expected_cost": row["delta_expected_cost"],
                "clean_recall": row["clean_recall"],
                "augmented_recall": row["augmented_recall"],
                "clean_fpr": row["clean_fpr"],
                "augmented_fpr": row["augmented_fpr"],
                "delta_missed_per_million": missed,
                "delta_false_alarms_per_million": alarms,
                "break_even_cost_ratio": break_even,
                "break_even_direction": direction,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    lightweight = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    realworld_path = RESULTS / "revision_realworld_probabilities.csv"
    realworld = pd.read_csv(realworld_path) if realworld_path.exists() else None
    transformer_path = RESULTS / "revision_distilbert_stream_probabilities.csv"
    transformer = pd.read_csv(transformer_path) if transformer_path.exists() else None
    store = build_streams(lightweight, realworld, transformer)

    regions = operating_regions(store)
    regions.to_csv(RESULTS / "revision_r3_operating_regions.csv", index=False)
    boundaries = miss_cost_boundaries(regions)
    boundaries.to_csv(RESULTS / "revision_r3_miss_cost_boundaries.csv", index=False)

    misspecification = prevalence_misspecification(store)
    misspecification.to_csv(
        RESULTS / "revision_r3_prevalence_misspecification.csv", index=False
    )

    value = operational_value(store)
    value.to_csv(RESULTS / "revision_r3_operational_value.csv", index=False)
    summary = break_even_summary(value)
    summary.to_csv(RESULTS / "revision_r3_break_even.csv", index=False)

    word = regions[regions["model"] == "word_lr_linked_augmented"]
    print("Best implementable rule by region (augmented word model):")
    table = (
        word.groupby(["stream", "prevalence", "cost_ratio"])["best_rule"]
        .agg(lambda s: s.value_counts().idxmax())
        .unstack("cost_ratio")
    )
    print(table.to_string())
    print()
    print("Miss-cost ranges by winning rule (augmented word model):")
    print(
        boundaries[boundaries["model"] == "word_lr_linked_augmented"][
            ["stream", "rule", "cells", "miss_cost_min", "miss_cost_max",
             "product_only_purity"]
        ].to_string(index=False)
    )
    print()
    print("Prevalence misspecification, augmented word model, r=10, excess cost ratio:")
    block = misspecification[misspecification["model"] == "word_lr_linked_augmented"]
    print(
        block.groupby(["stream", "true_prevalence", "believed_prevalence"])["excess_ratio"]
        .mean()
        .unstack("believed_prevalence")
        .to_string()
    )
    print()
    print("Operational value and break-even (mean over partitions, r=10):")
    print(
        summary[summary["prevalence"] == 0.01][
            [
                "family",
                "stream",
                "clean_expected_cost",
                "augmented_expected_cost",
                "delta_expected_cost",
                "clean_recall",
                "augmented_recall",
                "clean_fpr",
                "augmented_fpr",
                "delta_missed_per_million",
                "delta_false_alarms_per_million",
                "break_even_cost_ratio",
                "break_even_direction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
