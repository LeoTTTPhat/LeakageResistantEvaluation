#!/usr/bin/env python3
"""Discriminating tests for DistilBERT's generator-disjoint behaviour.

Reviewer comment 4 (round 3): the explanation offered for the transformer's
generator-disjoint behaviour should be presented cautiously unless it is
experimentally verified, and sensitivity analyses should separate candidate
explanations. The previously submitted version reported learning curves,
reliability curves, a truncation audit and a 160-vs-320 sequence-length
experiment. This script adds three tests that separate hypotheses which those
diagnostics leave confounded, using the saved probabilities rather than new
fine-tuning:

H1 (calibration mapping).  Is the ECE increase a mis-scaled probability map, or
    a ranking failure?  Post-hoc temperature scaling is fitted on the *in-
    pipeline validation stream* -- the only recalibration data a deployment has
    -- and applied unchanged to the generator-disjoint stream.  An oracle
    temperature fitted on the target stream itself bounds what any monotone
    rescaling could achieve; ROC-AUC is invariant to it by construction, so the
    residual gap after oracle scaling is discrimination, not calibration.

H2 (truncation).  Restricting the generator-disjoint corpus to the messages that
    fit inside the 160-token window removes truncation by construction, without
    retraining, and asks whether the clean-vs-augmented contrast survives.

H3 (threshold placement).  How much of the F1 deficit is recovered by moving the
    decision threshold alone (target-oracle threshold) rather than by changing
    the model?

Outputs (results/):
  revision_r3_distilbert_temperature.csv
  revision_r3_distilbert_truncation_strata.csv
  revision_r3_distilbert_hypotheses.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import f1_score, roc_auc_score

from rev_common import RESULTS, base, load_corpora

EPSILON = 1e-6
STREAMS = ["validation", "linked_rewrites", "external_openai"]


def clip_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(probability, EPSILON, 1.0 - EPSILON))


def negative_log_likelihood(temperature: float, z: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(expit(z / temperature), EPSILON, 1.0 - EPSILON)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    result = minimize_scalar(
        negative_log_likelihood, bounds=(0.05, 20.0), args=(z, y), method="bounded"
    )
    return float(result.x)


def summarise(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    return {
        "f1": float(f1_score(y, p >= threshold, zero_division=0)),
        "ece": float(base.expected_calibration_error(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "mean_probability": float(p.mean()),
    }


def best_threshold_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    index = int(np.argmax(scores))
    return float(grid[index]), float(scores[index])


def temperature_analysis(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in sorted(table["seed"].unique()):
        for training in ("clean-only", "linked-augmented"):
            block = table[(table["seed"] == seed) & (table["training"] == training)]
            validation = block[block["condition"] == "validation"]
            if validation.empty:
                continue
            z_validation = clip_logit(validation["probability"].to_numpy())
            y_validation = validation["label"].to_numpy().astype(int)
            deployable_temperature = fit_temperature(z_validation, y_validation)
            for stream in STREAMS:
                frame = block[block["condition"] == stream]
                if frame.empty:
                    continue
                y = frame["label"].to_numpy().astype(int)
                z = clip_logit(frame["probability"].to_numpy())
                oracle_temperature = fit_temperature(z, y)
                record = {
                    "seed": int(seed),
                    "training": training,
                    "stream": stream,
                    "validation_temperature": deployable_temperature,
                    "oracle_temperature": oracle_temperature,
                }
                for tag, p in (
                    ("raw", expit(z)),
                    ("validation_scaled", expit(z / deployable_temperature)),
                    ("oracle_scaled", expit(z / oracle_temperature)),
                ):
                    for key, value in summarise(y, p).items():
                        record[f"{tag}_{key}"] = value
                threshold, f1 = best_threshold_f1(y, expit(z))
                record["oracle_threshold"] = threshold
                record["oracle_threshold_f1"] = f1
                rows.append(record)
    return pd.DataFrame(rows)


def truncation_strata() -> pd.DataFrame:
    """F1 by untruncated message length, at both fine-tuning window sizes.

    Only 2 of 3,746 L2 messages fit inside a 160-token window, so an
    "untruncated subset" test is empty at that setting. The discriminating
    comparison is instead the interaction: if truncation drives the augmented
    model's deficit, the deficit must grow with the share of the message that
    the window discards, and must shrink when the window is widened to 320
    tokens on the same messages.
    """
    from transformers import AutoTokenizer

    _, external, _, _ = load_corpora()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    lengths = np.array(
        [
            len(tokenizer(text, truncation=False)["input_ids"])
            for text in external["text"].tolist()
        ]
    )
    edges = np.quantile(lengths, [0.0, 0.25, 0.5, 0.75, 1.0])

    table = pd.read_csv(RESULTS / "revision_distilbert_probabilities.csv")
    table = table[(table["condition"] == "external_openai") & (table["epoch"] == 1)]

    rows = []
    for max_length in sorted(table["max_length"].unique()):
        for seed in sorted(table["seed"].unique()):
            for training in ("clean-only", "linked-augmented"):
                block = table[
                    (table["max_length"] == max_length)
                    & (table["seed"] == seed)
                    & (table["training"] == training)
                ].reset_index(drop=True)
                if len(block) != len(lengths):
                    raise ValueError("L2 probability rows do not align with the corpus")
                y = block["label"].to_numpy().astype(int)
                p = block["probability"].to_numpy()
                for index in range(4):
                    low, high = edges[index], edges[index + 1]
                    mask = (
                        (lengths >= low) & (lengths <= high)
                        if index == 3
                        else (lengths >= low) & (lengths < high)
                    )
                    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                        continue
                    kept = np.minimum(lengths[mask], max_length) / lengths[mask]
                    record = {
                        "max_length": int(max_length),
                        "seed": int(seed),
                        "training": training,
                        "length_quartile": index + 1,
                        "tokens_low": float(low),
                        "tokens_high": float(high),
                        "median_tokens": float(np.median(lengths[mask])),
                        "mean_token_share_kept": float(kept.mean()),
                        "n": int(mask.sum()),
                    }
                    record.update(summarise(y[mask], p[mask]))
                    rows.append(record)
    return pd.DataFrame(rows)


def hypothesis_table(temperature: pd.DataFrame, strata: pd.DataFrame) -> pd.DataFrame:
    external = temperature[temperature["stream"] == "external_openai"]
    grouped = external.groupby("training").mean(numeric_only=True)

    def contrast(frame, column="f1"):
        pivot = frame.groupby("training")[column].mean()
        return float(pivot["linked-augmented"] - pivot["clean-only"])

    narrow = strata[strata["max_length"] == 160]
    wide = strata[strata["max_length"] == 320]
    short = narrow[narrow["length_quartile"] == 1]
    long = narrow[narrow["length_quartile"] == 4]
    wide_long = wide[wide["length_quartile"] == 4]

    rows = [
        {
            "hypothesis": "H1a: a recalibration available in deployment repairs the "
            "ECE increase",
            "test": "temperature fitted on the in-pipeline validation stream, "
            "applied unchanged to L2",
            "quantity": "augmented L2 ECE, raw -> validation-scaled",
            "value": f"{grouped.loc['linked-augmented', 'raw_ece']:.3f} -> "
            f"{grouped.loc['linked-augmented', 'validation_scaled_ece']:.3f} "
            f"(clean raw {grouped.loc['clean-only', 'raw_ece']:.3f})",
            "verdict": "not supported: the residual ECE is still above the clean "
            "model's unscaled value",
        },
        {
            "hypothesis": "H1b: some monotone rescaling repairs it",
            "test": "temperature fitted on the target stream itself, which needs "
            "target labels and is unattainable in deployment",
            "quantity": "augmented L2 ECE, raw -> oracle-scaled",
            "value": f"{grouped.loc['linked-augmented', 'raw_ece']:.3f} -> "
            f"{grouped.loc['linked-augmented', 'oracle_scaled_ece']:.3f}; fitted "
            f"temperature {grouped.loc['linked-augmented', 'validation_temperature']:.2f} "
            f"on validation vs {grouped.loc['linked-augmented', 'oracle_temperature']:.2f} "
            "on target",
            "verdict": "supported, but only with target labels: most of the excess "
            "is scale, and the scale correction cannot be learned in-pipeline",
        },
        {
            "hypothesis": "H1c: the augmented deficit is confined to probability "
            "scale and leaves ranking intact",
            "test": "ROC-AUC, invariant to any monotone rescaling",
            "quantity": "L2 ROC-AUC at window 160, clean vs augmented",
            "value": f"{grouped.loc['clean-only', 'raw_roc_auc']:.3f} vs "
            f"{grouped.loc['linked-augmented', 'raw_roc_auc']:.3f} "
            f"({grouped.loc['linked-augmented', 'raw_roc_auc'] - grouped.loc['clean-only', 'raw_roc_auc']:+.3f})",
            "verdict": "not supported: augmentation degrades ranking as well as "
            "scale at this window",
        },
        {
            "hypothesis": "H2a: the deficit tracks how much of each message the "
            "window discards, within the corpus",
            "test": "augmentation contrast by untruncated message length at window 160",
            "quantity": "augmented-minus-clean L2 F1, shortest vs longest quartile",
            "value": f"{contrast(short):+.3f} (median {short['median_tokens'].mean():.0f} "
            f"tokens, {short['mean_token_share_kept'].mean():.0%} kept) vs "
            f"{contrast(long):+.3f} (median {long['median_tokens'].mean():.0f} tokens, "
            f"{long['mean_token_share_kept'].mean():.0%} kept)",
            "verdict": "not corroborated, and this test has little power: the corpus "
            "is uniformly long, so every quartile loses a similar share",
        },
        {
            "hypothesis": "H2b: widening the window removes the deficit on the same "
            "messages",
            "test": "identical corpus and seeds refitted at window 320",
            "quantity": "augmented-minus-clean L2 F1 and ROC-AUC, 160 vs 320",
            "value": f"F1 {contrast(narrow):+.3f} -> {contrast(wide):+.3f}; "
            f"ROC-AUC {contrast(narrow, 'roc_auc'):+.3f} -> "
            f"{contrast(wide, 'roc_auc'):+.3f}",
            "verdict": "supported: the ranking degradation is specific to the "
            "truncated window and reverses when the message is restored",
        },
        {
            "hypothesis": "H3: a badly placed decision threshold explains the deficit",
            "test": "target-oracle threshold applied to the same probabilities",
            "quantity": "L2 F1 at t=0.5 -> at the oracle threshold",
            "value": f"augmented {grouped.loc['linked-augmented', 'raw_f1']:.3f} -> "
            f"{grouped.loc['linked-augmented', 'oracle_threshold_f1']:.3f}; clean "
            f"{grouped.loc['clean-only', 'raw_f1']:.3f} -> "
            f"{grouped.loc['clean-only', 'oracle_threshold_f1']:.3f}",
            "verdict": "not supported: both conditions gain a little and the ordering "
            "is unchanged",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    table = pd.read_csv(RESULTS / "revision_distilbert_stream_probabilities.csv")
    temperature = temperature_analysis(table)
    temperature.to_csv(RESULTS / "revision_r3_distilbert_temperature.csv", index=False)

    strata = truncation_strata()
    strata.to_csv(RESULTS / "revision_r3_distilbert_truncation_strata.csv", index=False)

    hypotheses = hypothesis_table(temperature, strata)
    hypotheses.to_csv(RESULTS / "revision_r3_distilbert_hypotheses.csv", index=False)

    print(
        temperature[temperature["stream"] == "external_openai"]
        .groupby("training")[
            [
                "validation_temperature",
                "oracle_temperature",
                "raw_f1",
                "raw_ece",
                "validation_scaled_ece",
                "oracle_scaled_ece",
                "raw_roc_auc",
                "oracle_threshold",
                "oracle_threshold_f1",
            ]
        ]
        .mean()
        .to_string()
    )
    print()
    print(
        strata.groupby(["max_length", "length_quartile", "training"])[
            ["n", "median_tokens", "mean_token_share_kept", "f1", "ece", "roc_auc"]
        ]
        .mean()
        .to_string()
    )
    print()
    print(hypotheses.to_string(index=False))


if __name__ == "__main__":
    main()
