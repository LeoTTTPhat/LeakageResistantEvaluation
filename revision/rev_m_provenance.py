#!/usr/bin/env python3
"""What the generator-disjoint corpus can and cannot establish (round-3 comment 1).

The corpus used at level L2 publishes no source-group, campaign, prompt or
model-version identifier. Round-3 comment 1 asks that the transfer claim either
be supported by stronger provenance evidence or be narrowed to match the
evidence. This script measures the three things that *are* checkable without
publisher metadata, so that the narrowed claim in the manuscript is a measured
statement rather than a concession:

1. Pipeline separability.  A source-grouped classifier is trained to tell a
   PhishFuzzer message from an L2 message.  Near-perfect separability shows the
   two corpora are distinct generative populations -- which is what
   "generator-disjoint" asserts -- while saying nothing about whether L2's own
   rows are independent of each other.

2. Proximity-stratified transfer.  L2 rows are stratified by their maximum
   cosine similarity to any training message, and the augmentation contrast is
   recomputed inside each stratum.  If the contrast survives in the least
   similar stratum, the L2 result is not an artefact of residual overlap with
   training material.

3. Latent-redundancy sensitivity.  Component counts, effective sample size
   (design effect) and cluster-deduplicated F1 are recomputed across linkage
   thresholds 0.80-0.95, so the audit's conclusions do not depend on the single
   0.90 threshold reported previously.

Outputs (results/):
  revision_r3_pipeline_separability.csv
  revision_r3_proximity_strata.csv
  revision_r3_redundancy_sensitivity.csv
  revision_r3_provenance_ledger.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from rev_common import RESULTS, base, load_corpora, load_embeddings, split_frames

SEEDS = [17, 29, 42, 71, 113]
LINKAGE_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
PAIRS = [
    ("Word TF-IDF", "probability_word_lr_clean_only",
     "probability_word_lr_linked_augmented"),
    ("MiniLM", "probability_minilm_lr_clean_only",
     "probability_minilm_lr_linked_augmented"),
]


def pipeline_separability(primary, external, primary_embeddings, external_embeddings):
    """Can a classifier tell the two generation pipelines apart?

    Folds are grouped by PhishFuzzer source item, so no source group's variants
    straddle the fold boundary; L2 rows have no identifiers and are split
    row-wise, which is the most favourable assumption for the corpus.
    """
    rows = []
    text = primary["text"].tolist() + external["text"].tolist()
    embeddings = np.vstack([primary_embeddings, external_embeddings])
    origin = np.concatenate([np.zeros(len(primary)), np.ones(len(external))]).astype(int)
    groups = np.concatenate(
        [
            primary["group_id"].astype(str).to_numpy(),
            np.array([f"external-{i}" for i in range(len(external))]),
        ]
    )
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(embeddings, origin, groups)
    ):
        word = base.WordModel.fit(
            [text[i] for i in train_index], origin[train_index]
        )
        word_probability = word.predict_proba([text[i] for i in test_index])
        embedding = LogisticRegression(
            C=2.0, class_weight="balanced", max_iter=3000, solver="liblinear"
        ).fit(embeddings[train_index], origin[train_index])
        embedding_probability = embedding.predict_proba(embeddings[test_index])[:, 1]
        for name, probability in (
            ("Word TF-IDF", word_probability),
            ("MiniLM", embedding_probability),
        ):
            rows.append(
                {
                    "representation": name,
                    "fold": fold,
                    "n_test": len(test_index),
                    "roc_auc": float(roc_auc_score(origin[test_index], probability)),
                    "f1": float(
                        f1_score(origin[test_index], probability >= 0.5, zero_division=0)
                    ),
                    "accuracy": float(
                        ((probability >= 0.5).astype(int) == origin[test_index]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def proximity_strata(external, external_embeddings, primary_embeddings, binary, spam):
    """Augmentation contrast inside strata of similarity to the training set."""
    probabilities = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    probabilities = probabilities[probabilities["condition"] == "external_openai"]
    rows = []
    for seed in SEEDS:
        frames = split_frames(binary, external, spam, seed)
        train_matrix = primary_embeddings[frames["train_all"].index.to_numpy()]
        maximum = (external_embeddings @ train_matrix.T).max(axis=1)
        block = probabilities[probabilities["seed"] == seed].sort_values("row_index")
        if len(block) != len(maximum):
            raise ValueError("external probability rows do not align with the corpus")
        y = block["label"].to_numpy().astype(int)
        edges = np.quantile(maximum, [0.0, 0.25, 0.5, 0.75, 1.0])
        for index in range(4):
            low, high = edges[index], edges[index + 1]
            mask = (
                (maximum >= low) & (maximum <= high)
                if index == 3
                else (maximum >= low) & (maximum < high)
            )
            if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                continue
            for family, clean_column, augmented_column in PAIRS:
                clean = block[clean_column].to_numpy()[mask]
                augmented = block[augmented_column].to_numpy()[mask]
                rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "quartile": index + 1,
                        "similarity_low": float(low),
                        "similarity_high": float(high),
                        "n": int(mask.sum()),
                        "f1_clean": float(f1_score(y[mask], clean >= 0.5, zero_division=0)),
                        "f1_augmented": float(
                            f1_score(y[mask], augmented >= 0.5, zero_division=0)
                        ),
                        "delta": float(
                            f1_score(y[mask], augmented >= 0.5, zero_division=0)
                            - f1_score(y[mask], clean >= 0.5, zero_division=0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def components(similarity: np.ndarray, threshold: float) -> np.ndarray:
    """Single-linkage components of a dense cosine-similarity matrix."""
    n = similarity.shape[0]
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rows, columns = np.where(np.triu(similarity >= threshold, k=1))
    for a, b in zip(rows, columns):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def redundancy_sensitivity(external, external_embeddings):
    similarity = external_embeddings @ external_embeddings.T
    probabilities = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    probabilities = probabilities[probabilities["condition"] == "external_openai"]
    rows = []
    for threshold in LINKAGE_THRESHOLDS:
        labels = components(similarity, threshold)
        unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        design_effect = float((counts**2).sum() / counts.sum())
        representative = pd.Series(inverse).drop_duplicates().index.to_numpy()
        for family, clean_column, augmented_column in PAIRS:
            per_seed_clean, per_seed_augmented = [], []
            for seed in SEEDS:
                block = probabilities[probabilities["seed"] == seed].sort_values(
                    "row_index"
                )
                y = block["label"].to_numpy().astype(int)[representative]
                clean = block[clean_column].to_numpy()[representative]
                augmented = block[augmented_column].to_numpy()[representative]
                per_seed_clean.append(f1_score(y, clean >= 0.5, zero_division=0))
                per_seed_augmented.append(f1_score(y, augmented >= 0.5, zero_division=0))
            rows.append(
                {
                    "linkage_threshold": threshold,
                    "family": family,
                    "components": int(len(unique)),
                    "largest_component": int(counts.max()),
                    "rows_in_multi_member_components": int(counts[counts > 1].sum()),
                    "design_effect": design_effect,
                    "effective_sample_size": float(len(labels) / design_effect),
                    "dedup_f1_clean": float(np.mean(per_seed_clean)),
                    "dedup_f1_augmented": float(np.mean(per_seed_augmented)),
                    "dedup_delta": float(
                        np.mean(per_seed_augmented) - np.mean(per_seed_clean)
                    ),
                }
            )
    return pd.DataFrame(rows)


def ledger(separability, strata, redundancy):
    word = separability[separability["representation"] == "Word TF-IDF"]
    least = strata[(strata["quartile"] == 1) & (strata["family"] == "Word TF-IDF")]
    most = strata[(strata["quartile"] == 4) & (strata["family"] == "Word TF-IDF")]
    row = redundancy[
        (redundancy["linkage_threshold"] == 0.90)
        & (redundancy["family"] == "Word TF-IDF")
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "property": "Different generation pipeline from the training corpus",
                "status": "measured",
                "evidence": f"grouped-CV origin classifier separates the two corpora "
                f"at ROC-AUC {word['roc_auc'].mean():.3f}",
            },
            {
                "property": "Not derived from the training corpus",
                "status": "measured",
                "evidence": "augmentation contrast in the least-similar quartile "
                f"{least['delta'].mean():+.3f} vs {most['delta'].mean():+.3f} in the "
                "most-similar quartile",
            },
            {
                "property": "Rows independent at source level",
                "status": "not verifiable; audited and rejected",
                "evidence": f"{int(row['components'])} single-linkage components at "
                f"cosine 0.90, design effect {row['design_effect']:.2f}, effective "
                f"sample size {row['effective_sample_size']:.0f} of 3,746 rows",
            },
            {
                "property": "Source intents disjoint from the training corpus",
                "status": "not verifiable",
                "evidence": "no source-group, campaign or prompt identifier is "
                "published; similarity is an upper bound on overlap, not a proof "
                "of disjointness",
            },
            {
                "property": "Generator model and version",
                "status": "not verifiable",
                "evidence": "described by the publisher as OpenAI-generated with no "
                "model or version string; no per-row metadata",
            },
            {
                "property": "Prompt provenance / campaign structure",
                "status": "not verifiable",
                "evidence": "absent from the release; latent structure is estimated "
                "by clustering rather than read from identifiers",
            },
        ]
    )


def main() -> None:
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)

    separability = pipeline_separability(
        primary, external, primary_embeddings, external_embeddings
    )
    separability.to_csv(RESULTS / "revision_r3_pipeline_separability.csv", index=False)
    print(separability.groupby("representation")[["roc_auc", "f1", "accuracy"]].mean())

    strata = proximity_strata(
        external, external_embeddings, primary_embeddings, binary, spam
    )
    strata.to_csv(RESULTS / "revision_r3_proximity_strata.csv", index=False)
    print()
    print(
        strata.groupby(["family", "quartile"])[
            ["similarity_low", "similarity_high", "n", "f1_clean", "f1_augmented", "delta"]
        ]
        .mean()
        .to_string()
    )

    redundancy = redundancy_sensitivity(external, external_embeddings)
    redundancy.to_csv(RESULTS / "revision_r3_redundancy_sensitivity.csv", index=False)
    print()
    print(redundancy.to_string(index=False))

    table = ledger(separability, strata, redundancy)
    table.to_csv(RESULTS / "revision_r3_provenance_ledger.csv", index=False)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
