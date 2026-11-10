#!/usr/bin/env python3
"""Additional controls for augmentation diversity, leakage, and lexical reliance."""

from __future__ import annotations

import json
import platform
import re
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from run_experiment import (
    RESULTS,
    SEEDS,
    EmbeddingModel,
    WordModel,
    canonical,
    embedding_cache,
    load_external,
    load_primary,
    metrics,
    split_groups,
    threshold_for_recall,
)


FAMILIES = ("Word LR", "MiniLM LR")
CANONICAL_VARIANT = ("well_known", "short")
REWRITE_VARIANTS = (
    ("fabricated", "short"),
    ("fabricated", "medium"),
    ("fabricated", "long"),
    ("well_known", "medium"),
    ("well_known", "long"),
)
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "fpr_at_0_5",
    "fpr_at_95_recall",
]


def select_variants(
    frame: pd.DataFrame, variants: tuple[tuple[str, str], ...]
) -> pd.DataFrame:
    allowed = {f"{entity}/{length}" for entity, length in variants}
    keys = frame["Entity_Type"].astype(str) + "/" + frame["Length_Type"].astype(str)
    return frame[keys.isin(allowed)].copy()


def embeddings_for(
    frame: pd.DataFrame, embedding_by_index: dict[int, np.ndarray]
) -> np.ndarray:
    return np.stack([embedding_by_index[index] for index in frame.index])


def fit_family(
    family: str,
    train: pd.DataFrame,
    embedding_by_index: dict[int, np.ndarray],
) -> WordModel | EmbeddingModel:
    labels = train["label"].to_numpy().astype(int)
    if family == "Word LR":
        return WordModel.fit(train["text"].tolist(), labels)
    return EmbeddingModel.fit(embeddings_for(train, embedding_by_index), labels)


def predict_family(
    family: str,
    model: WordModel | EmbeddingModel,
    frame: pd.DataFrame,
    embedding_by_index: dict[int, np.ndarray],
    external_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    if family == "Word LR":
        if not isinstance(model, WordModel):
            raise TypeError("Word LR requires WordModel")
        return model.predict_proba(frame["text"].tolist())
    if not isinstance(model, EmbeddingModel):
        raise TypeError("MiniLM LR requires EmbeddingModel")
    matrix = (
        external_embeddings
        if external_embeddings is not None
        else embeddings_for(frame, embedding_by_index)
    )
    return model.predict_proba(matrix)


def evaluate(
    frame: pd.DataFrame,
    probability: np.ndarray,
    recall_threshold: float,
    negative_only: bool = False,
) -> dict[str, float]:
    if negative_only:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "fpr_at_0_5": float((probability >= 0.5).mean()),
            "fpr_at_95_recall": float(
                (probability >= recall_threshold).mean()
            ),
        }
    labels = frame["label"].to_numpy().astype(int)
    result = metrics(labels, probability)
    negatives = labels == 0
    result["fpr_at_0_5"] = float((probability[negatives] >= 0.5).mean())
    result["fpr_at_95_recall"] = float(
        (probability[negatives] >= recall_threshold).mean()
    )
    return result


def summarize(
    frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]
) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, as_index=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary.drop(columns=["index"], errors="ignore")


def top_positive_unigrams(model: WordModel, limit: int = 20) -> list[tuple[str, float]]:
    excluded = {"url", "email", "ip", "subject", "sender"}
    names = model.vectorizer.get_feature_names_out()
    coefficients = model.classifier.coef_[0]
    ranked = np.argsort(coefficients)[::-1]
    selected: list[tuple[str, float]] = []
    for index in ranked:
        feature = str(names[index])
        if (
            " " in feature
            or feature in excluded
            or len(feature) < 3
            or not feature.isalpha()
        ):
            continue
        selected.append((feature, float(coefficients[index])))
        if len(selected) == limit:
            return selected
    raise RuntimeError(f"Only found {len(selected)} eligible unigram features")


def mask_features(texts: list[str], features: list[str]) -> list[str]:
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(feature) for feature in features) + r")\b",
        flags=re.IGNORECASE,
    )
    return [pattern.sub(" MASK ", text) for text in texts]


def main() -> None:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    primary = load_primary().reset_index(drop=True)
    external = load_external().reset_index(drop=True)
    binary = primary[primary["label"].notna()].copy()
    binary["label"] = binary["label"].astype(int)
    spam = canonical(primary[primary["Type"].eq("Spam")])

    primary_embeddings, external_embeddings, embedding_info = embedding_cache(
        primary, external
    )
    embedding_by_index = {
        index: primary_embeddings[position]
        for position, index in enumerate(primary.index)
    }

    dose_rows: list[dict[str, object]] = []
    leave_one_out_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    cue_rows: list[dict[str, object]] = []
    cue_feature_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        train_ids, val_ids, test_ids = split_groups(binary, seed)
        train_all = binary[binary["group_id"].isin(train_ids)].copy()
        train_clean = canonical(train_all)
        validation_clean = canonical(binary[binary["group_id"].isin(val_ids)])
        test_all = binary[binary["group_id"].isin(test_ids)].copy()
        test_clean = canonical(test_all)
        test_linked = select_variants(test_all, REWRITE_VARIANTS)

        dose_conditions = {
            "canonical_only_1x": train_clean,
            "canonical_repeated_6x": pd.concat(
                [train_clean] * 6, axis=0, ignore_index=False
            ),
            "entity_short_2x": select_variants(
                train_all,
                (("well_known", "short"), ("fabricated", "short")),
            ),
            "length_well_known_3x": select_variants(
                train_all,
                (
                    ("well_known", "short"),
                    ("well_known", "medium"),
                    ("well_known", "long"),
                ),
            ),
            "full_factorial_6x": train_all,
        }
        fitted: dict[tuple[str, str], WordModel | EmbeddingModel] = {}
        thresholds: dict[tuple[str, str], float] = {}

        for family in FAMILIES:
            for training_condition, training_frame in dose_conditions.items():
                fit_started = time.perf_counter()
                model = fit_family(family, training_frame, embedding_by_index)
                fitted[(family, training_condition)] = model
                fit_rows.append(
                    {
                        "seed": seed,
                        "experiment": "augmentation_dose",
                        "family": family,
                        "training_condition": training_condition,
                        "n_rows": len(training_frame),
                        "n_groups": training_frame["group_id"].nunique(),
                        "seconds": time.perf_counter() - fit_started,
                    }
                )
                validation_probability = predict_family(
                    family,
                    model,
                    validation_clean,
                    embedding_by_index,
                )
                threshold = threshold_for_recall(
                    validation_clean["label"].to_numpy().astype(int),
                    validation_probability,
                )
                thresholds[(family, training_condition)] = threshold

                for evaluation_condition, frame, external_matrix, negative_only in [
                    ("clean", test_clean, None, False),
                    ("linked_rewrites", test_linked, None, False),
                    (
                        "external_openai",
                        external,
                        external_embeddings if family == "MiniLM LR" else None,
                        False,
                    ),
                    ("spam_hard_negative", spam, None, True),
                ]:
                    probability = predict_family(
                        family,
                        model,
                        frame,
                        embedding_by_index,
                        external_matrix,
                    )
                    dose_rows.append(
                        {
                            "seed": seed,
                            "family": family,
                            "training_condition": training_condition,
                            "training_rows": len(training_frame),
                            "training_variants_per_group": (
                                len(training_frame)
                                / training_frame["group_id"].nunique()
                            ),
                            "evaluation_condition": evaluation_condition,
                            "n": len(frame),
                            "n_groups": frame["group_id"].nunique(),
                            "threshold_95_recall": threshold,
                            **evaluate(
                                frame, probability, threshold, negative_only
                            ),
                        }
                    )

        for target_entity, target_length in REWRITE_VARIANTS:
            target = ((target_entity, target_length),)
            training_variants = (CANONICAL_VARIANT,) + tuple(
                variant for variant in REWRITE_VARIANTS if variant != target[0]
            )
            leave_target_out = select_variants(train_all, training_variants)
            target_test = select_variants(test_all, target)
            target_slug = f"{target_entity}/{target_length}"

            for family in FAMILIES:
                fit_started = time.perf_counter()
                leave_out_model = fit_family(
                    family, leave_target_out, embedding_by_index
                )
                fit_rows.append(
                    {
                        "seed": seed,
                        "experiment": "leave_one_rewrite_out",
                        "family": family,
                        "training_condition": f"leave_out_{target_slug}",
                        "n_rows": len(leave_target_out),
                        "n_groups": leave_target_out["group_id"].nunique(),
                        "seconds": time.perf_counter() - fit_started,
                    }
                )
                for training_condition, model in [
                    (
                        "canonical_only",
                        fitted[(family, "canonical_only_1x")],
                    ),
                    ("leave_target_out", leave_out_model),
                    (
                        "full_factorial",
                        fitted[(family, "full_factorial_6x")],
                    ),
                ]:
                    probability = predict_family(
                        family, model, target_test, embedding_by_index
                    )
                    leave_one_out_rows.append(
                        {
                            "seed": seed,
                            "family": family,
                            "target_entity": target_entity,
                            "target_length": target_length,
                            "target_variant": target_slug,
                            "training_condition": training_condition,
                            "n": len(target_test),
                            "n_groups": target_test["group_id"].nunique(),
                            **metrics(
                                target_test["label"].to_numpy().astype(int),
                                probability,
                            ),
                        }
                    )

        for family in FAMILIES:
            group_model = fitted[(family, "full_factorial_6x")]
            group_probability = predict_family(
                family, group_model, test_all, embedding_by_index
            )
            group_result = metrics(
                test_all["label"].to_numpy().astype(int), group_probability
            )
            leakage_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "split_strategy": "source_group",
                    "train_rows": len(train_all),
                    "test_rows": len(test_all),
                    "test_group_overlap_rate": 0.0,
                    **group_result,
                }
            )

            row_train_index, row_holdout_index = train_test_split(
                binary.index,
                test_size=0.30,
                random_state=seed,
                stratify=binary["label"],
            )
            _, row_test_index = train_test_split(
                row_holdout_index,
                test_size=0.50,
                random_state=seed + 1000,
                stratify=binary.loc[row_holdout_index, "label"],
            )
            row_train = binary.loc[row_train_index].copy()
            row_test = binary.loc[row_test_index].copy()
            fit_started = time.perf_counter()
            row_model = fit_family(family, row_train, embedding_by_index)
            fit_rows.append(
                {
                    "seed": seed,
                    "experiment": "leakage_audit",
                    "family": family,
                    "training_condition": "row_stratified",
                    "n_rows": len(row_train),
                    "n_groups": row_train["group_id"].nunique(),
                    "seconds": time.perf_counter() - fit_started,
                }
            )
            row_probability = predict_family(
                family, row_model, row_test, embedding_by_index
            )
            row_train_groups = set(row_train["group_id"])
            overlap = row_test["group_id"].isin(row_train_groups).mean()
            leakage_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "split_strategy": "row_stratified",
                    "train_rows": len(row_train),
                    "test_rows": len(row_test),
                    "test_group_overlap_rate": float(overlap),
                    **metrics(
                        row_test["label"].to_numpy().astype(int),
                        row_probability,
                    ),
                }
            )

        word_model = fitted[("Word LR", "full_factorial_6x")]
        if not isinstance(word_model, WordModel):
            raise TypeError("Cue masking requires WordModel")
        selected = top_positive_unigrams(word_model)
        cue_feature_rows.extend(
            {
                "seed": seed,
                "rank": rank,
                "feature": feature,
                "coefficient": coefficient,
            }
            for rank, (feature, coefficient) in enumerate(selected, start=1)
        )
        selected_names = [feature for feature, _ in selected]
        threshold = thresholds[("Word LR", "full_factorial_6x")]
        for evaluation_condition, frame in [
            ("linked_rewrites", test_linked),
            ("external_openai", external),
        ]:
            original_probability = word_model.predict_proba(frame["text"].tolist())
            masked_probability = word_model.predict_proba(
                mask_features(frame["text"].tolist(), selected_names)
            )
            for input_condition, probability in [
                ("original", original_probability),
                ("top20_positive_unigrams_masked", masked_probability),
            ]:
                cue_rows.append(
                    {
                        "seed": seed,
                        "evaluation_condition": evaluation_condition,
                        "input_condition": input_condition,
                        "n": len(frame),
                        "threshold_95_recall": threshold,
                        **evaluate(frame, probability, threshold),
                    }
                )

    dose = pd.DataFrame(dose_rows)
    dose.to_csv(RESULTS / "rigor_dose_metrics_by_seed.csv", index=False)
    summarize(
        dose,
        ["family", "training_condition", "evaluation_condition"],
        METRIC_COLUMNS,
    ).to_csv(RESULTS / "rigor_dose_summary.csv", index=False)

    leave_one_out = pd.DataFrame(leave_one_out_rows)
    leave_one_out.to_csv(
        RESULTS / "rigor_leave_one_out_metrics_by_seed.csv", index=False
    )
    summarize(
        leave_one_out,
        ["family", "target_variant", "training_condition"],
        ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "ece"],
    ).to_csv(RESULTS / "rigor_leave_one_out_summary.csv", index=False)

    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(RESULTS / "rigor_leakage_metrics_by_seed.csv", index=False)
    summarize(
        leakage,
        ["family", "split_strategy"],
        [
            "test_group_overlap_rate",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier",
            "ece",
        ],
    ).to_csv(RESULTS / "rigor_leakage_summary.csv", index=False)

    cue = pd.DataFrame(cue_rows)
    cue.to_csv(RESULTS / "rigor_cue_mask_metrics_by_seed.csv", index=False)
    cue_summary = summarize(
        cue,
        ["evaluation_condition", "input_condition"],
        METRIC_COLUMNS,
    )
    cue_summary.to_csv(RESULTS / "rigor_cue_mask_summary.csv", index=False)
    pd.DataFrame(cue_feature_rows).to_csv(
        RESULTS / "rigor_cue_features.csv", index=False
    )
    pd.DataFrame(fit_rows).to_csv(
        RESULTS / "rigor_runtime_costs.csv", index=False
    )

    run_info = {
        "wall_clock_seconds": time.time() - started,
        "python": platform.python_version(),
        "seeds": SEEDS,
        "families": list(FAMILIES),
        "rewrite_variants": [
            {"entity_type": entity, "length_type": length}
            for entity, length in REWRITE_VARIANTS
        ],
        "cue_mask_features_per_seed": 20,
        "embedding_cache": embedding_info,
    }
    (RESULTS / "rigor_run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )

    print("Augmentation-dose summary")
    print(
        summarize(
            dose,
            ["family", "training_condition", "evaluation_condition"],
            ["f1", "fpr_at_0_5"],
        ).to_string(index=False)
    )
    print("\nLeakage-audit summary")
    print(
        summarize(
            leakage,
            ["family", "split_strategy"],
            ["test_group_overlap_rate", "f1", "brier", "ece"],
        ).to_string(index=False)
    )
    print("\nLexical-cue masking summary")
    print(
        cue_summary[
            [
                "evaluation_condition",
                "input_condition",
                "f1_mean",
                "f1_std",
                "pr_auc_mean",
                "brier_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
