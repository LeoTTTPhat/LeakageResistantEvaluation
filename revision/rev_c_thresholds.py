#!/usr/bin/env python3
"""Cost-sensitive operating-point selection under asymmetric error costs.

The submitted manuscript selected a single validation threshold at 95% recall.
Reviewer 1 asks for a threshold *framework* that reflects the asymmetry between
false positives and false negatives and the prevalence of phishing in a real
mail stream, and asks that fixed and dynamic strategies be distinguished.

With ``C_FP`` normalized to one and ``r = C_FN / C_FP``, the expected cost per
delivered message at prevalence ``pi`` is

    C(t) = (1 - pi) * FPR(t) + pi * r * (1 - TPR(t)),

which depends on the stream only through the two rates, so no reweighting of the
evaluation corpora is required. Four rules are compared:

``fixed_0.5``          the default decision threshold;
``fixed_95_recall``    the manuscript's validation rule;
``fixed_cost_min``     cost-minimizing on validation, then frozen (a fixed rule);
``dynamic_rate_match`` re-fitted per stream without labels, by holding the
                       flagged fraction at its validation value;
``oracle_cost_min``    cost-minimizing on the target stream itself, an
                       unattainable bound used to measure regret.

Two trivial policies bound the useful range: flagging nothing costs ``pi * r``
and flagging everything costs ``1 - pi``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rev_common import RESULTS, SEEDS, TRANSFORMER_SEEDS

PREVALENCES = [0.001, 0.01, 0.05]
COST_RATIOS = [1.0, 5.0, 10.0, 20.0]
GRID = np.unique(np.concatenate([np.linspace(0.0, 1.0, 2001), [0.5]]))


def rates(y: np.ndarray, probability: np.ndarray, thresholds: np.ndarray):
    positives = probability[y == 1]
    negatives = probability[y == 0]
    tpr = (positives[None, :] >= thresholds[:, None]).mean(axis=1) if len(positives) else np.zeros(len(thresholds))
    fpr = (negatives[None, :] >= thresholds[:, None]).mean(axis=1) if len(negatives) else np.zeros(len(thresholds))
    return tpr, fpr


def cost(tpr: np.ndarray, fpr: np.ndarray, prevalence: float, ratio: float):
    return (1 - prevalence) * fpr + prevalence * ratio * (1 - tpr)


def threshold_for_recall(y, probability, target=0.95):
    positives = np.sort(probability[y == 1])
    if len(positives) == 0:
        return 0.5
    index = max(0, int(np.floor((1.0 - target) * len(positives))))
    return float(positives[index])


def flagged_fraction_threshold(probability: np.ndarray, fraction: float) -> float:
    if fraction <= 0:
        return 1.0
    if fraction >= 1:
        return 0.0
    return float(np.quantile(probability, 1.0 - fraction))


def build_streams(lightweight, realworld, transformer):
    """Return {(model, seed): {stream: (y, probability)}} plus validation."""
    store: dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    columns = [c for c in lightweight.columns if c.startswith("probability_")]
    for column in columns:
        model = column.replace("probability_", "")
        for seed in SEEDS:
            block = lightweight[lightweight["seed"] == seed]
            entry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            validation = block[block["condition"] == "validation"]
            entry["validation"] = (
                validation["label"].to_numpy(),
                validation[column].to_numpy(),
            )
            linked = block[block["condition"] == "linked_rewrites"]
            spam = block[block["condition"] == "spam_hard_negative"]
            entry["primary_pipeline"] = (
                np.concatenate([linked["label"].to_numpy(), spam["label"].to_numpy()]),
                np.concatenate([linked[column].to_numpy(), spam[column].to_numpy()]),
            )
            external = block[block["condition"] == "external_openai"]
            entry["pipeline_disjoint"] = (
                external["label"].to_numpy(),
                external[column].to_numpy(),
            )
            if realworld is not None:
                rw = realworld[realworld["seed"] == seed]
                human = rw[rw["condition"] == "human_phishing"]
                ham = rw[rw["condition"] == "spamassassin_ham"]
                sa_spam = rw[rw["condition"] == "spamassassin_spam"]
                entry["human_authored"] = (
                    np.concatenate(
                        [
                            human["label"].to_numpy(),
                            ham["label"].to_numpy(),
                            sa_spam["label"].to_numpy(),
                        ]
                    ),
                    np.concatenate(
                        [
                            human[column].to_numpy(),
                            ham[column].to_numpy(),
                            sa_spam[column].to_numpy(),
                        ]
                    ),
                )
            store[(model, seed)] = entry

    if transformer is not None:
        for training in transformer["training"].unique():
            model = f"distilbert_{training.replace('-', '_')}"
            for seed in TRANSFORMER_SEEDS:
                block = transformer[
                    (transformer["seed"] == seed)
                    & (transformer["training"] == training)
                ]
                if block.empty:
                    continue

                def get(name):
                    part = block[block["condition"] == name]
                    return part["label"].to_numpy(), part["probability"].to_numpy()

                entry = {"validation": get("validation")}
                linked_y, linked_p = get("linked_rewrites")
                spam_y, spam_p = get("spam_hard_negative")
                entry["primary_pipeline"] = (
                    np.concatenate([linked_y, spam_y]),
                    np.concatenate([linked_p, spam_p]),
                )
                entry["pipeline_disjoint"] = get("external_openai")
                human_y, human_p = get("human_phishing")
                ham_y, ham_p = get("spamassassin_ham")
                sa_y, sa_p = get("spamassassin_spam")
                entry["human_authored"] = (
                    np.concatenate([human_y, ham_y, sa_y]),
                    np.concatenate([human_p, ham_p, sa_p]),
                )
                store[(model, seed)] = entry
    return store


def main() -> None:
    lightweight = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    realworld_path = RESULTS / "revision_realworld_probabilities.csv"
    realworld = pd.read_csv(realworld_path) if realworld_path.exists() else None
    transformer_path = RESULTS / "revision_distilbert_stream_probabilities.csv"
    transformer = pd.read_csv(transformer_path) if transformer_path.exists() else None

    store = build_streams(lightweight, realworld, transformer)

    rows = []
    curve_rows = []
    for (model, seed), streams in store.items():
        validation_y, validation_p = streams["validation"]
        validation_tpr, validation_fpr = rates(validation_y, validation_p, GRID)
        recall_threshold = threshold_for_recall(validation_y, validation_p)

        for prevalence in PREVALENCES:
            for ratio in COST_RATIOS:
                validation_cost = cost(validation_tpr, validation_fpr, prevalence, ratio)
                fixed_cost_threshold = float(GRID[int(np.argmin(validation_cost))])
                validation_rate = float(
                    (validation_p >= fixed_cost_threshold).mean()
                )
                for stream, (y, probability) in streams.items():
                    if stream == "validation":
                        continue
                    tpr, fpr = rates(y, probability, GRID)
                    target_cost = cost(tpr, fpr, prevalence, ratio)
                    oracle_index = int(np.argmin(target_cost))
                    rules = {
                        "fixed_0.5": 0.5,
                        "fixed_95_recall": recall_threshold,
                        "fixed_cost_min": fixed_cost_threshold,
                        "dynamic_rate_match": flagged_fraction_threshold(
                            probability, validation_rate
                        ),
                        "oracle_cost_min": float(GRID[oracle_index]),
                    }
                    oracle_cost = float(target_cost[oracle_index])
                    for rule, threshold in rules.items():
                        predicted = probability >= threshold
                        realized_tpr = (
                            float(predicted[y == 1].mean()) if (y == 1).any() else np.nan
                        )
                        realized_fpr = (
                            float(predicted[y == 0].mean()) if (y == 0).any() else np.nan
                        )
                        realized = (1 - prevalence) * realized_fpr + prevalence * ratio * (
                            1 - realized_tpr
                        )
                        rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "stream": stream,
                                "prevalence": prevalence,
                                "cost_ratio": ratio,
                                "rule": rule,
                                "threshold": threshold,
                                "recall": realized_tpr,
                                "fpr": realized_fpr,
                                "flagged_fraction": float(predicted.mean()),
                                "expected_cost": realized,
                                "oracle_cost": oracle_cost,
                                "regret": realized - oracle_cost,
                                "cost_flag_nothing": prevalence * ratio,
                                "cost_flag_everything": 1 - prevalence,
                            }
                        )
        if seed == 42:
            for stream, (y, probability) in streams.items():
                if stream == "validation":
                    continue
                tpr, fpr = rates(y, probability, GRID)
                for prevalence in PREVALENCES:
                    for ratio in [10.0]:
                        values = cost(tpr, fpr, prevalence, ratio)
                        keep = np.linspace(0, len(GRID) - 1, 201).astype(int)
                        curve_rows.append(
                            pd.DataFrame(
                                {
                                    "model": model,
                                    "stream": stream,
                                    "prevalence": prevalence,
                                    "cost_ratio": ratio,
                                    "threshold": GRID[keep],
                                    "expected_cost": values[keep],
                                }
                            )
                        )

    # Best attainable F1 per stream, separating a badly placed operating point
    # from an inability to discriminate at all.
    ceiling_rows = []
    for (model, seed), streams in store.items():
        for stream, (y, probability) in streams.items():
            if stream == "validation" or y.max() == 0:
                continue
            tpr, fpr = rates(y, probability, GRID)
            positives = float((y == 1).sum())
            negatives = float((y == 0).sum())
            true_positive = tpr * positives
            false_positive = fpr * negatives
            denominator = 2 * true_positive + false_positive + (positives - true_positive)
            f1_curve = np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            )
            best = int(np.argmax(f1_curve))
            ceiling_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "stream": stream,
                    "prevalence_in_stream": float((y == 1).mean()),
                    "f1_at_0_5": float(
                        f1_curve[int(np.argmin(np.abs(GRID - 0.5)))]
                    ),
                    "f1_at_oracle_threshold": float(f1_curve[best]),
                    "oracle_threshold": float(GRID[best]),
                    "recall_at_oracle": float(tpr[best]),
                    "precision_at_oracle": float(
                        true_positive[best]
                        / max(true_positive[best] + false_positive[best], 1e-12)
                    ),
                }
            )
    ceiling = pd.DataFrame(ceiling_rows)
    ceiling.to_csv(RESULTS / "revision_stream_ceiling_by_seed.csv", index=False)
    ceiling.groupby(["model", "stream"], as_index=False).mean(
        numeric_only=True
    ).drop(columns=["seed"]).to_csv(
        RESULTS / "revision_stream_ceiling_summary.csv", index=False
    )

    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "revision_threshold_cost_by_seed.csv", index=False)
    summary = (
        detail.groupby(
            ["model", "stream", "prevalence", "cost_ratio", "rule"], as_index=False
        )[["threshold", "recall", "fpr", "flagged_fraction", "expected_cost", "regret"]]
        .mean()
    )
    summary.to_csv(RESULTS / "revision_threshold_cost_summary.csv", index=False)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(
            RESULTS / "revision_threshold_cost_curves.csv", index=False
        )

    view = summary[
        (summary["model"] == "word_lr_linked_augmented")
        & (summary["cost_ratio"] == 10.0)
    ]
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
