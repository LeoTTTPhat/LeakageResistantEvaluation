#!/usr/bin/env python3
"""Augmentation in deployment units over the larger partition sets (round 4).

Round 3 stated the augmentation decision as missed phishing messages and false
alarms per million delivered (``rev_o_deployment.py``, Table "operational"), but
for DistilBERT that table rested on the three published partitions -- the only
ones with stored validation probabilities at the time. The twelve partitions
added in this round store validation and every stream, so the same quantities
can be computed over fifteen DistilBERT partitions, and over the twenty
lightweight partitions of ``rev_k_multisplit.py`` (L1 and L2; L3 probabilities
exist for the five published partitions only). The human-authored stream is the
SpamAssassin-deduplicated stream of ``rev_w_deployment_robustness.build_store``.

Each model is placed at its own best implementable rule (threshold 0.5,
validation 95%-recall, validation cost-minimizing, label-free rate matching), as
in round 3. The break-even cost ratio is computed from the mean error deltas.

Outputs (results/):
  revision_r4_operational_partitions.csv    per family x stream x partition
  revision_r4_break_even_partitions.csv     means and break-even ratios
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rev_o_deployment as deployment
import rev_w_deployment_robustness as robustness
from rev_common import RESULTS

RENAME = {"L1 primary": "primary_pipeline", "L2 generator-disjoint": "pipeline_disjoint",
          "L3 human-authored": "human_authored"}
PAIRS = [("Word TF-IDF", "word_lr_clean_only", "word_lr_linked_augmented"),
         ("MiniLM", "minilm_lr_clean_only", "minilm_lr_linked_augmented"),
         ("DistilBERT", "distilbert_clean_only", "distilbert_linked_augmented")]


def break_even(missed: float, alarms: float) -> tuple[float, str]:
    if missed < 0 and alarms <= 0:
        return 0.0, "augmented cheaper at every r"
    if missed >= 0 and alarms > 0:
        return float("inf"), "augmented never cheaper"
    if missed == 0:
        return 0.0, "augmented cheaper at every r"
    if missed < 0:
        return -alarms / missed, "augmented cheaper above"
    return -alarms / missed, "augmented cheaper below"


def main() -> None:
    store = robustness.build_store()
    store = {key: {RENAME.get(s, s): v for s, v in entry.items() if s in RENAME or s == "validation"}
             for key, entry in store.items()}
    rows = []
    million = 1_000_000
    for family, clean_model, augmented_model in PAIRS:
        seeds = sorted({seed for (model, seed) in store if model == clean_model}
                       & {seed for (model, seed) in store if model == augmented_model})
        for seed in seeds:
            for prevalence in (0.001, 0.01, 0.05):
                ratio = 10.0
                for stream in ("primary_pipeline", "pipeline_disjoint", "human_authored"):
                    record = {"family": family, "seed": seed, "stream": stream,
                              "prevalence": prevalence, "cost_ratio": ratio}
                    usable = True
                    for tag, model in (("clean", clean_model), ("augmented", augmented_model)):
                        streams = store[(model, seed)]
                        if stream not in streams:
                            usable = False
                            break
                        y, p = streams[stream]
                        thresholds = deployment.rule_thresholds(streams, prevalence, ratio)
                        value, rule = min((deployment.realized_cost(y, p, thresholds[r](p), prevalence, ratio)[0], r)
                                          for r in deployment.IMPLEMENTABLE)
                        _, tpr, fpr = deployment.realized_cost(y, p, thresholds[rule](p), prevalence, ratio)
                        record.update({f"{tag}_rule": rule, f"{tag}_recall": tpr, f"{tag}_fpr": fpr,
                                       f"{tag}_expected_cost": value,
                                       f"{tag}_missed_per_million": million * prevalence * (1 - tpr),
                                       f"{tag}_false_alarms_per_million": million * (1 - prevalence) * fpr})
                    if not usable:
                        continue
                    record["delta_expected_cost"] = record["augmented_expected_cost"] - record["clean_expected_cost"]
                    record["delta_missed_per_million"] = (record["augmented_missed_per_million"]
                                                          - record["clean_missed_per_million"])
                    record["delta_false_alarms_per_million"] = (record["augmented_false_alarms_per_million"]
                                                                - record["clean_false_alarms_per_million"])
                    rows.append(record)
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "revision_r4_operational_partitions.csv", index=False)
    summary = []
    for (family, stream, prevalence), block in detail.groupby(["family", "stream", "prevalence"]):
        missed = block["delta_missed_per_million"].mean()
        alarms = block["delta_false_alarms_per_million"].mean()
        ratio_value, direction = break_even(missed, alarms)
        summary.append({"family": family, "stream": stream, "prevalence": prevalence, "partitions": len(block),
                        "clean_expected_cost": block["clean_expected_cost"].mean(),
                        "augmented_expected_cost": block["augmented_expected_cost"].mean(),
                        "delta_expected_cost": block["delta_expected_cost"].mean(),
                        "share_partitions_augmented_costlier": float((block["delta_expected_cost"] > 0).mean()),
                        "clean_fpr": block["clean_fpr"].mean(), "augmented_fpr": block["augmented_fpr"].mean(),
                        "delta_missed_per_million": missed, "delta_false_alarms_per_million": alarms,
                        "break_even_cost_ratio": ratio_value, "break_even_direction": direction})
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "revision_r4_break_even_partitions.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary[summary["prevalence"] == 0.01].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
