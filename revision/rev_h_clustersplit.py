#!/usr/bin/env python3
"""Template-disjoint splitting: a stricter partition than source grouping.

Reviewer 1 observes that binding the six variants of a source item together
assumes those variants behave like the original, and asks whether source-group
independence genuinely improves the estimate. The direct test is to make the
partition *stricter* and see whether the estimate moves.

Source groups are clustered by canonical-message similarity (single-linkage
connected components over a MiniLM cosine graph). Every group in a cluster is
then assigned to the same partition, so no template family can span train and
test. Three progressively stricter thresholds are compared with the manuscript's
source-group split under otherwise identical training and evaluation code.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rev_common import RESULTS, SEEDS, base, load_corpora, load_embeddings
from rev_d_similarity import cluster_components

THRESHOLDS = [0.95, 0.90, 0.85]


def cluster_split(binary, cluster_of_group, seed):
    """Stratified 70/15/15 split over template clusters rather than groups."""
    table = (
        binary[["group_id", "label"]]
        .drop_duplicates()
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    table["cluster"] = table["group_id"].map(cluster_of_group)
    clusters = (
        table.groupby("cluster")["label"].mean().round().astype(int).reset_index()
    )
    train_clusters, holdout = train_test_split(
        clusters["cluster"],
        test_size=0.30,
        random_state=seed,
        stratify=clusters["label"],
    )
    holdout_table = clusters[clusters["cluster"].isin(holdout)]
    validation_clusters, test_clusters = train_test_split(
        holdout_table["cluster"],
        test_size=0.50,
        random_state=seed + 1000,
        stratify=holdout_table["label"],
    )
    assign = {}
    for name, chosen in [
        ("train", set(train_clusters)),
        ("validation", set(validation_clusters)),
        ("test", set(test_clusters)),
    ]:
        for group, cluster in cluster_of_group.items():
            if cluster in chosen:
                assign[group] = name
    return assign


def evaluate(binary, primary_embeddings, external, external_embeddings, assign, seed):
    partition = binary["group_id"].map(assign)
    train_all = binary[partition == "train"]
    train_clean = base.canonical(train_all)
    test_groups = binary[partition == "test"]
    test_linked = test_groups[
        ~(
            test_groups["Entity_Type"].eq("well_known")
            & test_groups["Length_Type"].eq("short")
        )
    ]
    results = []
    fits = {
        "Word LR, clean-only": (base.WordModel, train_clean),
        "Word LR, linked-augmented": (base.WordModel, train_all),
    }
    for name, (factory, frame) in fits.items():
        model = factory.fit(frame["text"].tolist(), frame["label"].to_numpy())
        for condition, target in [
            ("linked_rewrites", test_linked),
            ("external_openai", external),
        ]:
            probability = model.predict_proba(target["text"].tolist())
            results.append(
                {
                    "seed": seed,
                    "model": name,
                    "condition": condition,
                    "n": len(target),
                    "n_train": len(frame),
                    **base.metrics(target["label"].to_numpy().astype(int), probability),
                }
            )
    for name, frame in [
        ("MiniLM LR, clean-only", train_clean),
        ("MiniLM LR, linked-augmented", train_all),
    ]:
        model = base.EmbeddingModel.fit(
            primary_embeddings[frame.index.to_numpy()], frame["label"].to_numpy()
        )
        for condition, target, matrix in [
            (
                "linked_rewrites",
                test_linked,
                primary_embeddings[test_linked.index.to_numpy()],
            ),
            ("external_openai", external, external_embeddings),
        ]:
            probability = model.predict_proba(matrix)
            results.append(
                {
                    "seed": seed,
                    "model": name,
                    "condition": condition,
                    "n": len(target),
                    "n_train": len(frame),
                    **base.metrics(target["label"].to_numpy().astype(int), probability),
                }
            )
    return results


def main() -> None:
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)

    canonical = base.canonical(binary)
    canonical_embeddings = primary_embeddings[canonical.index.to_numpy()]
    groups = canonical["group_id"].to_numpy()

    structure = []
    rows = []
    for threshold in THRESHOLDS:
        assignment = cluster_components(canonical_embeddings, threshold)
        cluster_of_group = dict(zip(groups, assignment))
        sizes = np.bincount(assignment)
        sizes = sizes[sizes > 0]
        structure.append(
            {
                "threshold": threshold,
                "source_groups": int(len(groups)),
                "template_clusters": int(len(sizes)),
                "largest_cluster": int(sizes.max()),
                "groups_in_multi_group_clusters": int(sizes[sizes > 1].sum()),
            }
        )
        print(
            f"threshold {threshold}: {len(sizes)} clusters over {len(groups)} groups",
            flush=True,
        )
        for seed in SEEDS:
            assign = cluster_split(binary, cluster_of_group, seed)
            for record in evaluate(
                binary,
                primary_embeddings,
                external,
                external_embeddings,
                assign,
                seed,
            ):
                record["threshold"] = threshold
                record["partition"] = f"template-cluster ({threshold})"
                rows.append(record)

    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "revision_template_split_metrics.csv", index=False)
    summary = (
        detail.groupby(["partition", "model", "condition"], as_index=False)[
            ["f1", "precision", "recall", "brier", "ece", "n_train"]
        ]
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
    summary.to_csv(RESULTS / "revision_template_split_summary.csv", index=False)
    (RESULTS / "revision_template_cluster_structure.json").write_text(
        json.dumps(structure, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
