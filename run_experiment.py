#!/usr/bin/env python3
"""Leakage-resistant evaluation on real LLM-generated email corpora.

The primary PhishFuzzer corpus contains six linked rewrites for every source item.
All splits are made by Original_ID before any variant is selected. The external
corpus is independently generated with OpenAI models and is never used for tuning.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CACHE = HERE / "cache"
SEEDS = [17, 29, 42, 71, 113]
PRIMARY_SEED = 42
N_BOOTSTRAP = 5000

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SPACE_RE = re.compile(r"\s+")


def neutralize(value: object) -> str:
    text = "" if value is None else str(value)
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = IP_RE.sub(" <IP> ", text)
    return SPACE_RE.sub(" ", text).strip()


def join_email(row: pd.Series) -> str:
    return neutralize(
        f"Subject: {row.get('Subject', '')}\n"
        f"Sender: {row.get('Sender', '')}\n"
        f"{row.get('Body', '')}"
    )


def load_primary() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "phishfuzzer.parquet")
    df = df[df["Language"].eq("en")].copy()
    df["text"] = df.apply(join_email, axis=1)
    df["label"] = df["Type"].map({"Valid": 0, "Phishing": 1})
    df["group_id"] = df["Original_ID"].astype(str)
    return df


def load_external() -> pd.DataFrame:
    df = pd.read_csv(DATA / "gpt_phishing_email.csv")
    df = df.dropna(subset=["content", "label"]).copy()
    df["text"] = df["content"].map(neutralize)
    df["label"] = df["label"].astype(int)
    df["group_id"] = [f"external-{i}" for i in range(len(df))]
    return df[["text", "label", "group_id"]]


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["Entity_Type"].eq("well_known") & df["Length_Type"].eq("short")
    ].copy()


def split_groups(binary: pd.DataFrame, seed: int) -> tuple[set[str], set[str], set[str]]:
    groups = (
        binary[["group_id", "label"]]
        .drop_duplicates()
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    train_ids, holdout_ids = train_test_split(
        groups["group_id"],
        test_size=0.30,
        random_state=seed,
        stratify=groups["label"],
    )
    holdout = groups[groups["group_id"].isin(holdout_ids)]
    val_ids, test_ids = train_test_split(
        holdout["group_id"],
        test_size=0.50,
        random_state=seed + 1000,
        stratify=holdout["label"],
    )
    return set(train_ids), set(val_ids), set(test_ids)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(ece)


def threshold_for_recall(
    y_true: np.ndarray, probability: np.ndarray, target: float = 0.95
) -> float:
    positives = np.sort(probability[y_true == 1])
    if len(positives) == 0:
        return 0.5
    index = max(0, int(math.floor((1.0 - target) * len(positives))))
    return float(positives[index])


def metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    result = {
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "brier": brier_score_loss(y_true, probability),
        "ece": expected_calibration_error(y_true, probability),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, probability)
        result["pr_auc"] = average_precision_score(y_true, probability)
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return {key: float(value) for key, value in result.items()}


@dataclass
class WordModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    @classmethod
    def fit(cls, texts: list[str], labels: np.ndarray) -> "WordModel":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(
            C=4.0,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
        ).fit(matrix, labels)
        return cls(vectorizer, classifier)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.vectorizer.transform(texts))[:, 1]


@dataclass
class CharWordModel:
    word: TfidfVectorizer
    char: TfidfVectorizer
    classifier: LogisticRegression

    @classmethod
    def fit(cls, texts: list[str], labels: np.ndarray) -> "CharWordModel":
        word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=40000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=40000,
            sublinear_tf=True,
        )
        matrix = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
        classifier = LogisticRegression(
            C=4.0,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
        ).fit(matrix, labels)
        return cls(word, char, classifier)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        matrix = hstack(
            [self.word.transform(texts), self.char.transform(texts)]
        ).tocsr()
        return self.classifier.predict_proba(matrix)[:, 1]


@dataclass
class EmbeddingModel:
    classifier: LogisticRegression

    @classmethod
    def fit(cls, matrix: np.ndarray, labels: np.ndarray) -> "EmbeddingModel":
        classifier = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
        ).fit(matrix, labels)
        return cls(classifier)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(matrix)[:, 1]


def embedding_cache(
    primary: pd.DataFrame, external: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    all_text = primary["text"].tolist() + external["text"].tolist()
    fingerprint = hashlib.sha256("\n".join(all_text).encode("utf-8")).hexdigest()[:16]
    cache_path = CACHE / f"minilm-{fingerprint}.npy"
    cache_hit = cache_path.exists()
    if cache_hit:
        matrix = np.load(cache_path)
    else:
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        matrix = encoder.encode(
            all_text,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(cache_path, matrix)
    info = {
        "cache_hit": cache_hit,
        "cache_path": str(cache_path.relative_to(HERE)),
        "rows": len(all_text),
        "seconds": time.perf_counter() - started,
    }
    return matrix[: len(primary)], matrix[len(primary) :], info


def model_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("+", "plus")
        .replace(",", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def length_bin(text: str) -> str:
    words = len(text.split())
    if words < 75:
        return "<75"
    if words < 150:
        return "75-149"
    return "150+"


def grouped_bootstrap_delta(
    frame: pd.DataFrame,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, float]:
    groups = frame["group_id"].drop_duplicates().to_numpy()
    group_rows = {
        group: np.flatnonzero(frame["group_id"].to_numpy() == group) for group in groups
    }
    y = frame["label"].to_numpy()
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        f1_a = f1_score(y[rows], probability_a[rows] >= 0.5, zero_division=0)
        f1_b = f1_score(y[rows], probability_b[rows] >= 0.5, zero_division=0)
        deltas[index] = f1_b - f1_a
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": float(
            f1_score(y, probability_b >= 0.5, zero_division=0)
            - f1_score(y, probability_a >= 0.5, zero_division=0)
        ),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": float(
            min(1.0, 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
        ),
    }


def main() -> None:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    primary = load_primary().reset_index(drop=True)
    external = load_external().reset_index(drop=True)
    binary = primary[primary["label"].notna()].copy()
    binary["label"] = binary["label"].astype(int)
    spam = primary[primary["Type"].eq("Spam")].copy()

    primary_embeddings, external_embeddings, embedding_info = embedding_cache(
        primary, external
    )
    primary_embedding_by_index = {
        index: primary_embeddings[position]
        for position, index in enumerate(primary.index)
    }

    dataset_stats = {
        "primary_rows_english": int(len(primary)),
        "primary_groups_english": int(primary["group_id"].nunique()),
        "binary_rows": int(len(binary)),
        "binary_groups": int(binary["group_id"].nunique()),
        "phishing_rows": int((binary["label"] == 1).sum()),
        "valid_rows": int((binary["label"] == 0).sum()),
        "spam_rows_hard_negative": int(len(spam)),
        "external_rows": int(len(external)),
        "external_phishing": int((external["label"] == 1).sum()),
        "external_valid": int((external["label"] == 0).sum()),
        "external_duplicate_texts": int(external["text"].duplicated().sum()),
        "external_conflicting_text_labels": int(
            (external.groupby("text")["label"].nunique() > 1).sum()
        ),
        "seeds": SEEDS,
        "bootstrap_replicates": N_BOOTSTRAP,
    }
    (RESULTS / "dataset_stats.json").write_text(
        json.dumps(dataset_stats, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    primary_predictions: dict[str, np.ndarray] = {}
    primary_linked: pd.DataFrame | None = None

    for seed in SEEDS:
        train_ids, val_ids, test_ids = split_groups(binary, seed)
        train_all = binary[binary["group_id"].isin(train_ids)]
        train_clean = canonical(train_all)
        val = canonical(binary[binary["group_id"].isin(val_ids)])
        test_groups = binary[binary["group_id"].isin(test_ids)]
        test_clean = canonical(test_groups)
        test_linked = test_groups[
            ~(
                test_groups["Entity_Type"].eq("well_known")
                & test_groups["Length_Type"].eq("short")
            )
        ].copy()
        spam_clean = canonical(spam)

        for partition, group_ids in [
            ("train", train_ids),
            ("validation", val_ids),
            ("test", test_ids),
        ]:
            group_frame = (
                binary[binary["group_id"].isin(group_ids)][["group_id", "label"]]
                .drop_duplicates()
                .sort_values("group_id")
            )
            split_rows.extend(
                {
                    "seed": seed,
                    "group_id": row.group_id,
                    "label": int(row.label),
                    "partition": partition,
                }
                for row in group_frame.itertuples(index=False)
            )

        models: dict[str, object] = {}
        fit_specs = [
            (
                "Word LR, clean-only",
                WordModel.fit,
                train_clean["text"].tolist(),
                train_clean["label"].to_numpy(),
            ),
            (
                "Word LR, linked-augmented",
                WordModel.fit,
                train_all["text"].tolist(),
                train_all["label"].to_numpy(),
            ),
            (
                "Char+word LR, linked-augmented",
                CharWordModel.fit,
                train_all["text"].tolist(),
                train_all["label"].to_numpy(),
            ),
        ]
        for model_name, fit_method, texts, labels in fit_specs:
            fit_started = time.perf_counter()
            models[model_name] = fit_method(texts, labels)
            cost_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "stage": "classifier_fit",
                    "condition": "training",
                    "n": len(labels),
                    "seconds": time.perf_counter() - fit_started,
                }
            )
        if seed == PRIMARY_SEED:
            audited_model = models["Word LR, linked-augmented"]
            if not isinstance(audited_model, WordModel):
                raise TypeError("Expected a word model for feature audit")
            names = audited_model.vectorizer.get_feature_names_out()
            coefficients = audited_model.classifier.coef_[0]
            for direction, indices in [
                ("legitimate", np.argsort(coefficients)[:30]),
                ("phishing", np.argsort(coefficients)[-30:][::-1]),
            ]:
                feature_rows.extend(
                    {
                        "seed": seed,
                        "direction": direction,
                        "rank": rank,
                        "feature": names[index],
                        "coefficient": float(coefficients[index]),
                    }
                    for rank, index in enumerate(indices, start=1)
                )

        train_clean_embeddings = np.stack(
            [primary_embedding_by_index[index] for index in train_clean.index]
        )
        train_all_embeddings = np.stack(
            [primary_embedding_by_index[index] for index in train_all.index]
        )
        for model_name, matrix, labels in [
            (
                "MiniLM LR, clean-only",
                train_clean_embeddings,
                train_clean["label"].to_numpy(),
            ),
            (
                "MiniLM LR, linked-augmented",
                train_all_embeddings,
                train_all["label"].to_numpy(),
            ),
        ]:
            fit_started = time.perf_counter()
            models[model_name] = EmbeddingModel.fit(matrix, labels)
            cost_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "stage": "classifier_fit_from_cached_embeddings",
                    "condition": "training",
                    "n": len(labels),
                    "seconds": time.perf_counter() - fit_started,
                }
            )

        evaluation_sets = {
            "clean": test_clean,
            "linked_rewrites": test_linked,
            "external_openai": external,
            "spam_hard_negative": spam_clean,
        }
        if seed == PRIMARY_SEED:
            for condition, frame in evaluation_sets.items():
                output = pd.DataFrame(
                    {
                        "condition": condition,
                        "row_id": [f"{condition}-{index}" for index in range(len(frame))],
                        "group_id": frame["group_id"].astype(str).to_numpy(),
                        "label": (
                            frame["label"].fillna(0).astype(int).to_numpy()
                            if condition != "spam_hard_negative"
                            else np.zeros(len(frame), dtype=int)
                        ),
                        "word_count": frame["text"].map(lambda text: len(text.split())).to_numpy(),
                    }
                )
                if "Entity_Type" in frame:
                    output["entity_type"] = frame["Entity_Type"].to_numpy()
                    output["length_type"] = frame["Length_Type"].to_numpy()
                else:
                    output["entity_type"] = "not_available"
                    output["length_type"] = "not_available"
                prediction_frames[condition] = output

        for model_name, model in models.items():
            if isinstance(model, EmbeddingModel):
                val_matrix = np.stack(
                    [primary_embedding_by_index[index] for index in val.index]
                )
                val_probability = model.predict_proba(val_matrix)
            else:
                val_probability = model.predict_proba(val["text"].tolist())
            recall_threshold = threshold_for_recall(
                val["label"].to_numpy(), val_probability
            )

            for condition, frame in evaluation_sets.items():
                predict_started = time.perf_counter()
                if isinstance(model, EmbeddingModel):
                    if condition == "external_openai":
                        probability = model.predict_proba(external_embeddings)
                    else:
                        matrix = np.stack(
                            [primary_embedding_by_index[index] for index in frame.index]
                        )
                        probability = model.predict_proba(matrix)
                else:
                    probability = model.predict_proba(frame["text"].tolist())
                cost_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "stage": (
                            "classifier_predict_from_cached_embeddings"
                            if isinstance(model, EmbeddingModel)
                            else "vectorize_and_predict"
                        ),
                        "condition": condition,
                        "n": len(frame),
                        "seconds": time.perf_counter() - predict_started,
                    }
                )

                if condition == "spam_hard_negative":
                    result = {
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
                else:
                    y_true = frame["label"].to_numpy().astype(int)
                    result = metrics(y_true, probability)
                    negative = y_true == 0
                    result["fpr_at_0_5"] = float(
                        (probability[negative] >= 0.5).mean()
                    )
                    result["fpr_at_95_recall"] = float(
                        (probability[negative] >= recall_threshold).mean()
                    )
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "condition": condition,
                        "n": len(frame),
                        "n_groups": frame["group_id"].nunique(),
                        "threshold_95_recall": recall_threshold,
                        **result,
                    }
                )

                if condition == "linked_rewrites":
                    for (entity, message_length), subgroup in frame.groupby(
                        ["Entity_Type", "Length_Type"], sort=True
                    ):
                        positions = frame.index.get_indexer(subgroup.index)
                        subgroup_result = metrics(
                            subgroup["label"].to_numpy().astype(int),
                            probability[positions],
                        )
                        variant_rows.append(
                            {
                                "seed": seed,
                                "model": model_name,
                                "entity_type": entity,
                                "length_type": message_length,
                                "n": len(subgroup),
                                "n_groups": subgroup["group_id"].nunique(),
                                **subgroup_result,
                            }
                        )

                if seed == PRIMARY_SEED:
                    prediction_frames[condition][
                        f"probability_{model_slug(model_name)}"
                    ] = probability
                    y_for_error = (
                        frame["label"].to_numpy().astype(int)
                        if condition != "spam_hard_negative"
                        else np.zeros(len(frame), dtype=int)
                    )
                    bins = frame["text"].map(length_bin).to_numpy()
                    predicted = (probability >= 0.5).astype(int)
                    for label_value in np.unique(y_for_error):
                        for bin_name in ["<75", "75-149", "150+"]:
                            mask = (y_for_error == label_value) & (bins == bin_name)
                            if not mask.any():
                                continue
                            errors = predicted[mask] != y_for_error[mask]
                            error_rows.append(
                                {
                                    "seed": seed,
                                    "model": model_name,
                                    "condition": condition,
                                    "true_label": int(label_value),
                                    "length_bin": bin_name,
                                    "n": int(mask.sum()),
                                    "errors": int(errors.sum()),
                                    "error_rate": float(errors.mean()),
                                    "mean_probability": float(probability[mask].mean()),
                                }
                            )

                if seed == PRIMARY_SEED and condition == "linked_rewrites":
                    primary_predictions[model_name] = probability
                    primary_linked = frame.copy()

    detailed = pd.DataFrame(rows)
    detailed.to_csv(RESULTS / "metrics_by_seed.csv", index=False)
    pd.DataFrame(variant_rows).to_csv(RESULTS / "variant_metrics_by_seed.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(RESULTS / "runtime_costs.csv", index=False)
    pd.DataFrame(split_rows).to_csv(RESULTS / "split_assignments.csv", index=False)
    pd.DataFrame(error_rows).to_csv(RESULTS / "error_analysis_seed42.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(RESULTS / "feature_audit_seed42.csv", index=False)
    pd.concat(prediction_frames.values(), ignore_index=True).to_csv(
        RESULTS / "predictions_seed42.csv", index=False
    )
    numeric_columns = [
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
    summary = (
        detailed.groupby(["model", "condition"], as_index=False)[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.drop(columns=["index"], errors="ignore")
    summary.to_csv(RESULTS / "summary.csv", index=False)
    variant_summary = (
        pd.DataFrame(variant_rows)
        .groupby(["model", "entity_type", "length_type"], as_index=False)[
            ["precision", "recall", "f1", "brier", "ece"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    variant_summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in variant_summary.columns
    ]
    variant_summary = variant_summary.drop(columns=["index"], errors="ignore")
    variant_summary.to_csv(RESULTS / "variant_summary.csv", index=False)

    if primary_linked is None:
        raise RuntimeError("Primary-seed predictions were not produced")
    comparisons = {}
    pairs = [
        ("Word LR, clean-only", "Word LR, linked-augmented"),
        ("MiniLM LR, clean-only", "MiniLM LR, linked-augmented"),
        ("Word LR, linked-augmented", "Char+word LR, linked-augmented"),
    ]
    for first, second in pairs:
        comparisons[f"{second} minus {first}"] = grouped_bootstrap_delta(
            primary_linked,
            primary_predictions[first],
            primary_predictions[second],
            seed=PRIMARY_SEED,
        )
    (RESULTS / "paired_group_bootstrap.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )

    run_info = {
        "wall_clock_seconds": time.time() - started,
        "primary_seed": PRIMARY_SEED,
        "embedding_cache": embedding_info,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in [
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "scipy",
                    "scikit-learn",
                    "sentence-transformers",
                    "transformers",
                    "torch",
                    "huggingface-hub",
                ]
            },
        },
        "models": detailed["model"].drop_duplicates().tolist(),
        "conditions": detailed["condition"].drop_duplicates().tolist(),
    }
    (RESULTS / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
