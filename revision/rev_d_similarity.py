#!/usr/bin/env python3
"""Similarity and latent-provenance audits.

Reviewer 1 asks how similar the six variants of a source group actually are,
and whether binding them together can itself bias the estimate. Reviewer 2 asks
what can be established about source-level independence in corpora that publish
no group, campaign, or prompt identifiers.

Neither question can be answered by assertion, so this script measures:

1. within-source-group similarity against the nearest and the average *other*
   source group, for MiniLM cosine, TF-IDF cosine, and token Jaccard;
2. how exchangeable the six variants are to a trained detector, via the
   intraclass correlation of predicted probabilities within a source group;
3. latent near-duplicate structure in every held-out corpus, recovered by
   single-linkage connected components over a cosine graph, together with the
   effective number of independent units it implies;
4. cluster-aware versus row-wise bootstrap intervals for held-out F1, which
   shows how much the missing group identifiers actually cost;
5. cross-corpus dependence: the maximum similarity of every held-out message to
   any training message, and held-out F1 restricted to the least similar
   messages.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from rev_common import (
    PRIMARY_SEED,
    RESULTS,
    SEEDS,
    base,
    encode_minilm,
    load_corpora,
    load_embeddings,
    split_frames,
)
from rev_realworld_data import realworld_conditions

CHUNK = 512
CLUSTER_THRESHOLDS = [0.95, 0.90, 0.85]


def pairwise_upper_mean(matrix: np.ndarray) -> float:
    """Mean cosine over distinct pairs of L2-normalized rows."""
    if len(matrix) < 2:
        return float("nan")
    similarity = matrix @ matrix.T
    n = len(matrix)
    return float((similarity.sum() - np.trace(similarity)) / (n * (n - 1)))


def nearest_other(matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max and mean cosine of each row to same-label rows other than itself."""
    best = np.empty(len(matrix), dtype=float)
    average = np.empty(len(matrix), dtype=float)
    for start in range(0, len(matrix), CHUNK):
        block = matrix[start : start + CHUNK] @ matrix.T
        for offset in range(len(block)):
            index = start + offset
            same = labels == labels[index]
            same[index] = False
            values = block[offset][same]
            best[index] = values.max() if values.size else np.nan
            average[index] = values.mean() if values.size else np.nan
    return best, average


def cluster_components(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Single-linkage clusters as connected components of a cosine graph."""
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for start in range(0, len(matrix), CHUNK):
        block = matrix[start : start + CHUNK] @ matrix.T
        local_rows, local_columns = np.nonzero(block >= threshold)
        keep = (local_rows + start) != local_columns
        rows.append(local_rows[keep] + start)
        columns.append(local_columns[keep])
    if rows:
        row_index = np.concatenate(rows)
        column_index = np.concatenate(columns)
    else:  # pragma: no cover
        row_index = column_index = np.array([], dtype=int)
    graph = coo_matrix(
        (np.ones(len(row_index), dtype=np.int8), (row_index, column_index)),
        shape=(len(matrix), len(matrix)),
    )
    _, assignment = connected_components(graph, directed=False)
    return assignment


def max_similarity_to(matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    best = np.empty(len(matrix), dtype=float)
    for start in range(0, len(matrix), CHUNK):
        block = matrix[start : start + CHUNK] @ reference.T
        best[start : start + len(block)] = block.max(axis=1)
    return best


def f1_at(y: np.ndarray, probability: np.ndarray) -> float:
    predicted = probability >= 0.5
    tp = float(np.sum((y == 1) & predicted))
    fp = float(np.sum((y == 0) & predicted))
    fn = float(np.sum((y == 1) & ~predicted))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def bootstrap_f1(
    y: np.ndarray, probability: np.ndarray, units: np.ndarray | None, rng, n=2000
):
    """Percentile interval for F1, resampling rows or whole clusters."""
    if units is None:
        index = np.arange(len(y))
        blocks = [np.array([i]) for i in index]
    else:
        order = np.argsort(units, kind="stable")
        sorted_units = units[order]
        boundaries = np.flatnonzero(np.diff(sorted_units)) + 1
        blocks = np.split(order, boundaries)
    values = np.empty(n, dtype=float)
    count = len(blocks)
    for replicate in range(n):
        chosen = rng.integers(0, count, size=count)
        rows = np.concatenate([blocks[c] for c in chosen])
        values[replicate] = f1_at(y[rows], probability[rows])
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), float(values.std(ddof=1))


def variant_similarity(primary, binary, primary_embeddings):
    """Within-group versus between-group similarity for the primary corpus."""
    groups = binary["group_id"].to_numpy()
    unique_groups, first_index = np.unique(groups, return_index=True)
    labels = binary["label"].to_numpy()[first_index]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1), min_df=2, sublinear_tf=True, strip_accents="unicode"
    )
    tfidf = vectorizer.fit_transform(binary["text"].tolist())
    tfidf = tfidf.multiply(1.0 / np.sqrt(tfidf.multiply(tfidf).sum(axis=1)))
    tfidf = tfidf.tocsr()

    token_sets = [set(text.lower().split()) for text in binary["text"]]
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}

    rows = []
    for group in unique_groups:
        index = positions[group]
        embedding_block = primary_embeddings[binary.index.to_numpy()[index]]
        tfidf_block = tfidf[index]
        tfidf_similarity = (tfidf_block @ tfidf_block.T).toarray()
        n = len(index)
        jaccard = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = token_sets[index[i]], token_sets[index[j]]
                union = len(a | b)
                jaccard.append(len(a & b) / union if union else np.nan)
        rows.append(
            {
                "group_id": group,
                "n_variants": n,
                "minilm_within": pairwise_upper_mean(embedding_block),
                "tfidf_within": float(
                    (tfidf_similarity.sum() - np.trace(tfidf_similarity))
                    / (n * (n - 1))
                )
                if n > 1
                else np.nan,
                "jaccard_within": float(np.nanmean(jaccard)) if jaccard else np.nan,
            }
        )
    within = pd.DataFrame(rows)

    canonical_mask = (
        binary["Entity_Type"].eq("well_known") & binary["Length_Type"].eq("short")
    ).to_numpy()
    canonical_rows = np.flatnonzero(canonical_mask)
    canonical_groups = groups[canonical_rows]
    canonical_labels = binary["label"].to_numpy()[canonical_rows]
    canonical_embeddings = primary_embeddings[binary.index.to_numpy()[canonical_rows]]
    best, average = nearest_other(canonical_embeddings, canonical_labels)
    between = pd.DataFrame(
        {
            "group_id": canonical_groups,
            "minilm_nearest_other_group": best,
            "minilm_mean_other_group": average,
        }
    )

    canonical_tfidf = tfidf[canonical_rows]
    tfidf_best = np.empty(len(canonical_rows), dtype=float)
    for start in range(0, len(canonical_rows), CHUNK):
        block = (canonical_tfidf[start : start + CHUNK] @ canonical_tfidf.T).toarray()
        for offset in range(len(block)):
            index = start + offset
            same = canonical_labels == canonical_labels[index]
            same[index] = False
            tfidf_best[index] = block[offset][same].max()
    between["tfidf_nearest_other_group"] = tfidf_best

    merged = within.merge(between, on="group_id", how="left")
    merged["label"] = pd.Series(canonical_labels, index=canonical_groups).reindex(
        merged["group_id"]
    ).to_numpy()
    return merged


def variant_exchangeability(probability_table):
    """Intraclass correlation of predicted probabilities within source groups."""
    rows = []
    for (seed, column), block in _iterate_linked(probability_table):
        frame = block[["group_id", "label"]].copy()
        frame["value"] = block[column].to_numpy()
        grand = frame["value"].mean()
        means = frame.groupby("group_id")["value"].mean()
        sizes = frame.groupby("group_id")["value"].size()
        between = float(((means - grand) ** 2 * sizes).sum() / (len(means) - 1))
        within = float(
            frame.assign(center=frame["group_id"].map(means))
            .eval("(value - center) ** 2")
            .sum()
            / (len(frame) - len(means))
        )
        k = float(sizes.mean())
        icc = (between - within) / (between + (k - 1) * within)
        rows.append(
            {
                "seed": seed,
                "model": column.replace("probability_", ""),
                "icc_within_group": float(icc),
                "mean_within_group_sd": float(
                    frame.groupby("group_id")["value"].std(ddof=1).mean()
                ),
                "between_group_sd": float(means.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _iterate_linked(table):
    linked = table[table["condition"] == "linked_rewrites"]
    columns = [c for c in table.columns if c.startswith("probability_")]
    for seed in sorted(linked["seed"].unique()):
        block = linked[linked["seed"] == seed]
        for column in columns:
            yield (seed, column), block


def corpus_audit(name, frame, embeddings, probability_frame, train_embeddings, rng):
    """Latent-cluster, bootstrap, and cross-corpus dependence audit."""
    report = {"corpus": name, "n": int(len(frame))}
    duplicates = int(frame["text"].duplicated().sum())
    report["exact_duplicate_rows"] = duplicates

    for threshold in CLUSTER_THRESHOLDS:
        assignment = cluster_components(embeddings, threshold)
        sizes = np.bincount(assignment)
        report[f"clusters_at_{threshold}"] = int(len(sizes))
        report[f"largest_cluster_at_{threshold}"] = int(sizes.max())
        report[f"rows_in_multi_member_clusters_at_{threshold}"] = int(
            sizes[sizes > 1].sum()
        )
        report[f"effective_units_at_{threshold}"] = float(
            len(embeddings) / np.mean(sizes[assignment])
        )

    nearest = np.empty(len(embeddings), dtype=float)
    for start in range(0, len(embeddings), CHUNK):
        block = embeddings[start : start + CHUNK] @ embeddings.T
        for offset in range(len(block)):
            block[offset, start + offset] = -1.0
        nearest[start : start + len(block)] = block.max(axis=1)
    report["nearest_neighbour_cosine_median"] = float(np.median(nearest))
    report["nearest_neighbour_cosine_p90"] = float(np.percentile(nearest, 90))
    report["fraction_nn_cosine_above_0_9"] = float((nearest >= 0.9).mean())

    cross = max_similarity_to(embeddings, train_embeddings)
    report["max_cosine_to_training_median"] = float(np.median(cross))
    report["max_cosine_to_training_p95"] = float(np.percentile(cross, 95))
    report["fraction_max_cosine_to_training_above_0_9"] = float((cross >= 0.9).mean())

    assignment = cluster_components(embeddings, 0.90)
    y = frame["label"].to_numpy().astype(int)
    detail = []
    for column in [c for c in probability_frame.columns if c.startswith("probability_")]:
        probability = probability_frame[column].to_numpy()
        row_low, row_high, row_sd = bootstrap_f1(y, probability, None, rng)
        cluster_low, cluster_high, cluster_sd = bootstrap_f1(
            y, probability, assignment, rng
        )
        representative = pd.Series(assignment).drop_duplicates().index.to_numpy()
        far = cross <= np.median(cross)
        detail.append(
            {
                "corpus": name,
                "model": column.replace("probability_", ""),
                "f1": f1_at(y, probability),
                "f1_row_bootstrap_ci_low": row_low,
                "f1_row_bootstrap_ci_high": row_high,
                "f1_row_bootstrap_sd": row_sd,
                "f1_cluster_bootstrap_ci_low": cluster_low,
                "f1_cluster_bootstrap_ci_high": cluster_high,
                "f1_cluster_bootstrap_sd": cluster_sd,
                "ci_width_ratio_cluster_over_row": (cluster_high - cluster_low)
                / max(row_high - row_low, 1e-12),
                "f1_cluster_deduplicated": f1_at(
                    y[representative], probability[representative]
                ),
                "n_cluster_deduplicated": int(len(representative)),
                "f1_least_similar_half": f1_at(y[far], probability[far]),
                "n_least_similar_half": int(far.sum()),
            }
        )
    return report, pd.DataFrame(detail)


def main() -> None:
    rng = np.random.default_rng(20260808)
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)
    probability_table = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")

    print("variant similarity ...", flush=True)
    similarity = variant_similarity(primary, binary, primary_embeddings)
    similarity.to_csv(RESULTS / "revision_variant_similarity.csv", index=False)
    summary = {
        "n_groups": int(len(similarity)),
        "minilm_within_group_mean": float(similarity["minilm_within"].mean()),
        "minilm_within_group_median": float(similarity["minilm_within"].median()),
        "minilm_nearest_other_group_mean": float(
            similarity["minilm_nearest_other_group"].mean()
        ),
        "minilm_mean_other_group_mean": float(
            similarity["minilm_mean_other_group"].mean()
        ),
        "tfidf_within_group_mean": float(similarity["tfidf_within"].mean()),
        "tfidf_nearest_other_group_mean": float(
            similarity["tfidf_nearest_other_group"].mean()
        ),
        "jaccard_within_group_mean": float(similarity["jaccard_within"].mean()),
        "fraction_groups_within_above_nearest_other": float(
            (
                similarity["minilm_within"]
                > similarity["minilm_nearest_other_group"]
            ).mean()
        ),
    }

    print("variant exchangeability ...", flush=True)
    exchangeability = variant_exchangeability(probability_table)
    exchangeability.to_csv(
        RESULTS / "revision_variant_exchangeability.csv", index=False
    )

    print("corpus audits ...", flush=True)
    frames = split_frames(binary, external, spam, PRIMARY_SEED)
    train_embeddings = primary_embeddings[frames["train_all"].index.to_numpy()]

    audits = []
    details = []

    external_probabilities = probability_table[
        (probability_table["condition"] == "external_openai")
        & (probability_table["seed"] == PRIMARY_SEED)
    ].reset_index(drop=True)
    report, detail = corpus_audit(
        "external_openai",
        external,
        external_embeddings,
        external_probabilities,
        train_embeddings,
        rng,
    )
    audits.append(report)
    details.append(detail)

    realworld_path = RESULTS / "revision_realworld_probabilities.csv"
    if realworld_path.exists():
        realworld = pd.read_csv(realworld_path)
        conditions = realworld_conditions()
        human = conditions["human_phishing"]
        human_embeddings = encode_minilm(human["text"])
        human_probabilities = realworld[
            (realworld["condition"] == "human_phishing")
            & (realworld["seed"] == PRIMARY_SEED)
        ].reset_index(drop=True)
        report, detail = corpus_audit(
            "human_phishing",
            human,
            human_embeddings,
            human_probabilities,
            train_embeddings,
            rng,
        )
        audits.append(report)
        details.append(detail)

    pd.DataFrame(audits).to_csv(
        RESULTS / "revision_corpus_provenance_audit.csv", index=False
    )
    pd.concat(details, ignore_index=True).to_csv(
        RESULTS / "revision_cluster_aware_intervals.csv", index=False
    )
    (RESULTS / "revision_variant_similarity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(audits).to_string(index=False))


if __name__ == "__main__":
    main()
