#!/usr/bin/env python3
"""Detection as a function of proximity to the synthetic training data (round 4).

Comments R1.3 and R1.9. For each target corpus -- L2 generator-disjoint
phishing, L3 human-authored phishing (2000s), and era-matched real phishing
(Nazario 2015-2025) -- every message's maximum MiniLM cosine similarity to any
message of the augmented training set is computed on each of the five published
partitions, and the augmented word model's recall at threshold 0.5 is reported
within quartiles of that similarity (the L3 corpus is reported as published and
with the rows that duplicate a SpamAssassin message removed; see
rev_t3_l3_sa_overlap.py). Round 3 applied the same stratification to
L2 only, as a test of overlap; applied to all three corpora it shows whether one
quantity -- semantic proximity to the synthetic training distribution -- orders
detection across corpora and within each of them.

Outputs (results/):
  revision_r4_proximity_recall.csv   partition x corpus x quartile
  revision_r4_proximity_summary.csv  means over partitions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rev_common import RESULTS, base, encode_minilm, load_corpora, load_embeddings, split_frames
from rev_era_data import load_era_matched
from rev_realworld_data import load_human_phishing

SEEDS = [17, 29, 42, 71, 113]


def main() -> None:
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)
    nazario = load_era_matched()
    nazario_embeddings = encode_minilm(nazario["text_full"])
    human = load_human_phishing()
    human_embeddings = encode_minilm(human["text"])
    positive = (human["label"] == 1).to_numpy()
    l2_positive = (external["label"] == 1).to_numpy()
    duplicates_path = RESULTS / "revision_r4_l3_sa_duplicates.csv"
    not_duplicate = (~pd.read_csv(duplicates_path)["duplicate"].to_numpy()
                     if duplicates_path.exists() else np.ones(len(human), dtype=bool))
    dedup_positive = positive & not_duplicate
    corpora = {
        "L2 generator-disjoint phishing": (external_embeddings[l2_positive],
                                           external.loc[l2_positive, "text"].tolist(), None),
        "L3 human-authored phishing (2000s)": (human_embeddings[positive],
                                               human.loc[positive, "text"].tolist(), None),
        "L3 human-authored phishing (2000s), SpamAssassin duplicates removed": (
            human_embeddings[dedup_positive], human.loc[dedup_positive, "text"].tolist(), None),
        "Nazario 2015-2022": (nazario_embeddings[(nazario["year"] <= 2022).to_numpy()],
                              nazario.loc[nazario["year"] <= 2022, "text_full"].tolist(), None),
        "Nazario 2023-2025": (nazario_embeddings[(nazario["year"] >= 2023).to_numpy()],
                              nazario.loc[nazario["year"] >= 2023, "text_full"].tolist(), None),
    }
    rows = []
    for seed in SEEDS:
        frames = split_frames(binary, external, spam, seed)
        train = frames["train_all"]
        model = base.WordModel.fit(train["text"].tolist(), train["label"].to_numpy())
        train_matrix = primary_embeddings[train.index.to_numpy()]
        for corpus, (embedded, texts, _) in corpora.items():
            maximum = (embedded @ train_matrix.T).max(axis=1)
            probability = model.predict_proba(texts)
            edges = np.quantile(maximum, [0.25, 0.5, 0.75])
            quartile = np.digitize(maximum, edges)
            rows.append({"seed": seed, "corpus": corpus, "quartile": "all", "n": len(maximum),
                         "similarity_median": float(np.median(maximum)),
                         "share_at_or_above_0.9": float((maximum >= 0.9).mean()),
                         "recall_0.5": float((probability >= 0.5).mean())})
            for q in range(4):
                mask = quartile == q
                rows.append({"seed": seed, "corpus": corpus, "quartile": str(q + 1),
                             "n": int(mask.sum()),
                             "similarity_low": float(maximum[mask].min()),
                             "similarity_high": float(maximum[mask].max()),
                             "similarity_median": float(np.median(maximum[mask])),
                             "recall_0.5": float((probability[mask] >= 0.5).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "revision_r4_proximity_recall.csv", index=False)
    summary = table.groupby(["corpus", "quartile"]).mean(numeric_only=True).reset_index()
    summary.to_csv(RESULTS / "revision_r4_proximity_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
