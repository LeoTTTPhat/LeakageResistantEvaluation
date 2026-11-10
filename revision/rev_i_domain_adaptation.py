#!/usr/bin/env python3
"""Decomposing the human-authored (L3) collapse into a ceiling and a transfer gap.

This revision, as first drafted, acknowledged but did not measure two things
about the L3 result it had just added:

* that the collapse "measures the combined effect of authorship and distribution
  shift, and the design cannot apportion it between the two"; and
* that rewrite augmentation "is not a substitute for target-domain data when the
  target is real attacks", without saying how much target-domain data would be
  required.

This script replaces both acknowledgements with measurements. It asks four
questions on one fixed, near-duplicate-disjoint partition of the human-authored
corpus:

1. ``synthetic_transfer``  -- what the deployed detectors score on the human test
   split (the collapse, reproduced on this partition).
2. ``human_in_domain``     -- what the *same estimator family* scores when it is
   trained on human-authored mail instead. This is the corpus ceiling. It
   separates "the target task is hard" from "the transfer fails": whatever the
   ceiling reaches is available signal that transfer failed to capture.
3. ``mixed_k`` / ``human_k`` -- two few-shot adaptation curves over the same k
   human-authored messages, k = 0 ... 4000: one adding them to the synthetic
   augmented training set, one training on them alone. The first converts the
   qualitative claim about target-domain data into an exchange rate. The second
   is the control that decides whether the synthetic corpus contributes anything
   at all once real data is available -- if ``mixed_k`` and ``human_k`` coincide,
   rewrite augmentation is not merely insufficient for the real task but
   inert for it. A third arm, ``mixed_balanced_k``, repeats the mixture with the
   k target messages oversampled to the mass of the synthetic set, so that any
   harm from mixing cannot be attributed to naive proportions alone.
4. ``reverse``             -- human-trained detectors evaluated on L1 and L2.
   If synthetic->human fails but human->synthetic does not, the collapse is
   asymmetric and the direction of that asymmetry is itself evidence about which
   corpus carries the narrower notion of phishing.

Splitting. The human corpus publishes no campaign identifier, so this script
applies the same latent-provenance procedure used for L2 in ``rev_d``: messages
are clustered into connected components at cosine 0.90 in MiniLM space and the
partition is taken over components, never over rows. An in-domain ceiling
computed on a row-wise split would be inflated by near-duplicates and would
overstate the transfer gap.

Neither the human test split nor SpamAssassin is used for feature selection,
calibration, or threshold choice in any condition.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from rev_common import (
    RESULTS,
    SEEDS,
    base,
    encode_minilm,
    load_corpora,
    load_embeddings,
    split_frames,
)
from rev_realworld_data import load_human_phishing

COSINE_THRESHOLD = 0.90
HOLDOUT_FRACTION = 0.30
FEWSHOT_SIZES = [0, 25, 50, 100, 250, 500, 1000, 2000, 4000]
BLOCK = 512


def connected_components(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Component label per row, linking any pair at cosine >= threshold.

    Union-find over blocked cosine similarity; rows are L2-normalized already,
    so the inner product is the cosine.
    """
    n = len(embeddings)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        similarity = embeddings[start:stop] @ embeddings.T
        for offset in range(stop - start):
            row = start + offset
            neighbours = np.nonzero(similarity[offset] >= threshold)[0]
            for neighbour in neighbours:
                if neighbour > row:
                    union(row, int(neighbour))
    return np.array([find(i) for i in range(n)])


def component_split(components: np.ndarray, labels: np.ndarray, seed: int):
    """Hold out HOLDOUT_FRACTION of components, stratified by component majority."""
    frame = pd.DataFrame({"component": components, "label": labels})
    majority = frame.groupby("component")["label"].mean().round().astype(int)
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for value in (0, 1):
        ids = majority.index[majority == value].to_numpy()
        ids = rng.permutation(ids)
        cut = int(round(len(ids) * (1.0 - HOLDOUT_FRACTION)))
        train_parts.append(ids[:cut])
        test_parts.append(ids[cut:])
    train_ids = set(np.concatenate(train_parts).tolist())
    test_ids = set(np.concatenate(test_parts).tolist())
    is_train = np.array([c in train_ids for c in components])
    is_test = np.array([c in test_ids for c in components])
    return is_train, is_test


def oracle_threshold_f1(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 over a threshold sweep, and the threshold attaining it."""
    positives = int(y.sum())
    if positives == 0:
        return float("nan"), float("nan")
    order = np.argsort(-probability)
    true_positive = np.cumsum(y[order])
    predicted = np.arange(1, len(y) + 1)
    f1_curve = 2 * true_positive / (predicted + positives)
    index = int(np.argmax(f1_curve))
    return float(f1_curve[index]), float(probability[order][index])


def evaluate(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    result = base.metrics(y, probability)
    f1_oracle, threshold = oracle_threshold_f1(y, probability)
    result["f1_at_oracle"] = f1_oracle
    result["oracle_threshold"] = threshold
    result["fpr_at_0_5"] = float((probability[y == 0] >= 0.5).mean()) if (y == 0).any() else float("nan")
    return result


def main() -> None:
    started = time.time()
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)

    human = load_human_phishing()
    print(f"human corpus: {len(human)} messages, prevalence {human['label'].mean():.4f}")
    human_embeddings = encode_minilm(human["text"])
    human_texts = human["text"].tolist()
    human_labels = human["label"].to_numpy()

    print(f"clustering at cosine {COSINE_THRESHOLD} ...", flush=True)
    components = connected_components(human_embeddings, COSINE_THRESHOLD)
    n_components = len(np.unique(components))
    print(f"  {len(human)} rows -> {n_components} components")

    structure = {
        "rows": int(len(human)),
        "components": int(n_components),
        "cosine_threshold": COSINE_THRESHOLD,
        "prevalence": float(human_labels.mean()),
        "largest_component": int(pd.Series(components).value_counts().iloc[0]),
    }
    (RESULTS / "revision_human_cluster_structure.json").write_text(
        json.dumps(structure, indent=2) + "\n"
    )

    rows: list[dict] = []
    for seed in SEEDS:
        is_train, is_test = component_split(components, human_labels, seed)
        train_index = np.nonzero(is_train)[0]
        test_index = np.nonzero(is_test)[0]
        y_test = human_labels[test_index]
        test_texts = [human_texts[i] for i in test_index]
        test_embeddings = human_embeddings[test_index]

        frames = split_frames(binary, external, spam, seed)
        train_all = frames["train_all"]
        train_clean = frames["train_clean"]
        linked = frames["linked_rewrites"]

        synthetic = {
            "clean_only": (
                train_clean["text"].tolist(),
                train_clean["label"].to_numpy(),
                primary_embeddings[train_clean.index.to_numpy()],
            ),
            "linked_augmented": (
                train_all["text"].tolist(),
                train_all["label"].to_numpy(),
                primary_embeddings[train_all.index.to_numpy()],
            ),
        }

        def record(condition: str, training: str, model: str, n_train: int,
                   n_human_train: int, y: np.ndarray, probability: np.ndarray) -> None:
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "training": training,
                    "model": model,
                    "n_train": int(n_train),
                    "n_human_train": int(n_human_train),
                    "n_eval": int(len(y)),
                    **evaluate(y, probability),
                }
            )

        # -- 1. synthetic -> human transfer, on this partition -----------------
        for training, (texts, labels, matrix) in synthetic.items():
            word = base.WordModel.fit(texts, labels)
            record("synthetic_transfer", training, "word_lr", len(texts), 0,
                   y_test, word.predict_proba(test_texts))
            embed = base.EmbeddingModel.fit(matrix, labels)
            record("synthetic_transfer", training, "minilm_lr", len(texts), 0,
                   y_test, embed.predict_proba(test_embeddings))

        # -- 2. human in-domain ceiling ---------------------------------------
        human_train_texts = [human_texts[i] for i in train_index]
        human_train_labels = human_labels[train_index]
        human_train_embeddings = human_embeddings[train_index]
        word = base.WordModel.fit(human_train_texts, human_train_labels)
        record("human_in_domain", "human_only", "word_lr", len(train_index),
               len(train_index), y_test, word.predict_proba(test_texts))
        embed = base.EmbeddingModel.fit(human_train_embeddings, human_train_labels)
        record("human_in_domain", "human_only", "minilm_lr", len(train_index),
               len(train_index), y_test, embed.predict_proba(test_embeddings))

        # -- 4. reverse transfer: human-trained, evaluated on L1 and L2 --------
        record("reverse_to_L1", "human_only", "word_lr", len(train_index),
               len(train_index), linked["label"].to_numpy(),
               word.predict_proba(linked["text"].tolist()))
        record("reverse_to_L2", "human_only", "word_lr", len(train_index),
               len(train_index), external["label"].to_numpy().astype(int),
               word.predict_proba(external["text"].tolist()))
        record("reverse_to_L1", "human_only", "minilm_lr", len(train_index),
               len(train_index), linked["label"].to_numpy(),
               embed.predict_proba(primary_embeddings[linked.index.to_numpy()]))
        record("reverse_to_L2", "human_only", "minilm_lr", len(train_index),
               len(train_index), external["label"].to_numpy().astype(int),
               embed.predict_proba(external_embeddings))

        # -- 3. few-shot adaptation curve -------------------------------------
        rng = np.random.default_rng(seed + 7919)
        base_texts, base_labels, base_matrix = synthetic["linked_augmented"]
        positive_pool = train_index[human_labels[train_index] == 1]
        negative_pool = train_index[human_labels[train_index] == 0]
        prevalence = float(human_labels.mean())
        for k in FEWSHOT_SIZES:
            if k == 0:
                chosen = np.array([], dtype=int)
            else:
                n_positive = max(1, int(round(k * prevalence)))
                n_negative = max(1, k - n_positive)
                chosen = np.concatenate(
                    [
                        rng.choice(positive_pool, size=min(n_positive, len(positive_pool)), replace=False),
                        rng.choice(negative_pool, size=min(n_negative, len(negative_pool)), replace=False),
                    ]
                )
            mixed_texts = base_texts + [human_texts[i] for i in chosen]
            mixed_labels = np.concatenate([base_labels, human_labels[chosen]])
            word = base.WordModel.fit(mixed_texts, mixed_labels)
            record("mixed", f"augmented_plus_{k}", "word_lr", len(mixed_texts),
                   len(chosen), y_test, word.predict_proba(test_texts))
            mixed_matrix = np.vstack([base_matrix, human_embeddings[chosen]]) if len(chosen) else base_matrix
            embed = base.EmbeddingModel.fit(mixed_matrix, mixed_labels)
            record("mixed", f"augmented_plus_{k}", "minilm_lr", len(mixed_texts),
                   len(chosen), y_test, embed.predict_proba(test_embeddings))

            # control: the same k human messages, oversampled to the mass of the
            # synthetic set, so the mixture is not dominated by proportion alone
            if k > 0:
                repeats = max(1, int(round(len(base_texts) / len(chosen))))
                oversampled = np.tile(chosen, repeats)
                balanced_texts = base_texts + [human_texts[i] for i in oversampled]
                balanced_labels = np.concatenate([base_labels, human_labels[oversampled]])
                word_bal = base.WordModel.fit(balanced_texts, balanced_labels)
                record("mixed_balanced", f"augmented_plus_{k}x{repeats}", "word_lr",
                       len(balanced_texts), len(chosen), y_test,
                       word_bal.predict_proba(test_texts))
                embed_bal = base.EmbeddingModel.fit(
                    np.vstack([base_matrix, human_embeddings[oversampled]]),
                    balanced_labels,
                )
                record("mixed_balanced", f"augmented_plus_{k}x{repeats}", "minilm_lr",
                       len(balanced_texts), len(chosen), y_test,
                       embed_bal.predict_proba(test_embeddings))

            # control: the same k human messages with no synthetic data at all
            if k > 0 and len(np.unique(human_labels[chosen])) == 2:
                word_only = base.WordModel.fit(
                    [human_texts[i] for i in chosen], human_labels[chosen]
                )
                record("human_only_k", f"human_{k}", "word_lr", len(chosen),
                       len(chosen), y_test, word_only.predict_proba(test_texts))
                embed_only = base.EmbeddingModel.fit(
                    human_embeddings[chosen], human_labels[chosen]
                )
                record("human_only_k", f"human_{k}", "minilm_lr", len(chosen),
                       len(chosen), y_test, embed_only.predict_proba(test_embeddings))
        print(f"seed {seed} complete ({time.time() - started:.0f}s)", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "revision_domain_adaptation_by_seed.csv", index=False)

    numeric = [c for c in detail.columns
               if c not in {"seed", "condition", "training", "model"}]
    summary = (
        detail.groupby(["condition", "training", "model"], as_index=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.drop(columns=["index"], errors="ignore")
    summary.to_csv(RESULTS / "revision_domain_adaptation_summary.csv", index=False)

    show = summary[summary["condition"] != "mixed"][
        ["condition", "training", "model", "f1_mean", "f1_std",
         "precision_mean", "recall_mean", "pr_auc_mean", "f1_at_oracle_mean"]
    ]
    print()
    print(show.to_string(index=False))
    print()
    curve = summary[summary["condition"].isin(["mixed", "mixed_balanced", "human_only_k"])][
        ["condition", "training", "model", "n_human_train_mean", "f1_mean",
         "f1_std", "recall_mean", "pr_auc_mean"]
    ].sort_values(["model", "condition", "n_human_train_mean"])
    curve.to_csv(RESULTS / "revision_fewshot_curve.csv", index=False)
    print(curve.to_string(index=False))
    print(f"\ndone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
