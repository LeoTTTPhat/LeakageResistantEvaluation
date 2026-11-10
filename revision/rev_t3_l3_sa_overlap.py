#!/usr/bin/env python3
"""The human-authored aggregate contains the SpamAssassin corpus (round 4).

Found while answering R1.3/R1.9. The public L3 aggregate
(``zefang-liu/phishing-email-dataset``) was assembled from older public corpora,
and one of them is the Apache SpamAssassin corpus that this study also uses as a
separate all-negative stress test. SpamAssassin *spam* appears in the aggregate
labelled "Phishing Email" and SpamAssassin *ham* labelled "Safe Email". Two
consequences: (i) part of the L3 positive class is commercial spam by
construction; (ii) the round-3 human-authored cost stream, which appended all
SpamAssassin messages to the aggregate as negatives, contained the same messages
twice with contradictory labels.

This script measures the overlap, records which L3 rows duplicate a SpamAssassin
message (row indices only; no text), recomputes the L3 detection metrics with
those rows removed from the saved probabilities of every detector, and repeats
the in-domain ceiling and the synthetic-trained comparison on the deduplicated
corpus with the component-disjoint split of ``rev_i_domain_adaptation.py``.

A row is a duplicate if its MiniLM embedding has cosine >= 0.95 to a SpamAssassin
message (body-only view, which matches the aggregate's body-only layout) or if
the first 400 normalized characters coincide exactly.

Outputs (results/):
  revision_r4_l3_sa_overlap.json
  revision_r4_l3_sa_duplicates.csv      L3 row_index, label, max cosine, SA stream
  revision_r4_l3_dedup_metrics.csv      per detector x partition, full vs dedup
  revision_r4_l3_dedup_ceiling.csv      in-domain ceiling, full vs dedup corpus
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from rev_common import RESULTS, base, encode_minilm, load_corpora, split_frames
from rev_i_domain_adaptation import component_split, connected_components, oracle_threshold_f1
from rev_realworld_data import load_human_phishing
from rev_u_era_format import spamassassin_body_only

THRESHOLD = 0.95
SEEDS = [17, 29, 42, 71, 113]


def normalized_prefix(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()[:400]


def overlap():
    human = load_human_phishing()
    sa = spamassassin_body_only()
    human_embeddings = encode_minilm(human["text"])
    sa_embeddings = encode_minilm(sa["text_body"])
    maximum = np.zeros(len(human))
    argmax = np.zeros(len(human), dtype=int)
    for start in range(0, len(human), 2048):
        similarity = human_embeddings[start:start + 2048] @ sa_embeddings.T
        maximum[start:start + 2048] = similarity.max(1)
        argmax[start:start + 2048] = similarity.argmax(1)
    prefix = set(sa["text_body"].map(normalized_prefix))
    exact = human["text"].map(normalized_prefix).isin(prefix).to_numpy()
    label = human["label"].to_numpy()
    stream = sa["stream"].to_numpy()[argmax]
    summary = {"l3_rows": int(len(human)), "spamassassin_rows": int(len(sa)), "by_threshold": {}}
    for threshold in (0.99, 0.97, 0.95, 0.90):
        mask = maximum >= threshold
        summary["by_threshold"][str(threshold)] = {
            "l3_rows_matched": int(mask.sum()),
            "l3_phishing_matched": int((mask & (label == 1)).sum()),
            "l3_phishing_matched_to_sa_spam": int((mask & (label == 1) & (stream == "spam")).sum()),
            "l3_legitimate_matched": int((mask & (label == 0)).sum()),
            "l3_legitimate_matched_to_sa_ham": int((mask & (label == 0) & (stream == "ham")).sum()),
        }
    duplicate = (maximum >= THRESHOLD) | exact
    summary["exact_prefix_matches"] = int(exact.sum())
    summary["duplicate_rule"] = f"cosine >= {THRESHOLD} or identical first 400 normalized characters"
    summary["duplicates"] = int(duplicate.sum())
    summary["duplicates_phishing"] = int((duplicate & (label == 1)).sum())
    summary["duplicates_legitimate"] = int((duplicate & (label == 0)).sum())
    summary["dedup_rows"] = int((~duplicate).sum())
    summary["dedup_phishing"] = int(((~duplicate) & (label == 1)).sum())
    summary["dedup_prevalence"] = float(label[~duplicate].mean())
    table = pd.DataFrame({"row_index": np.arange(len(human)), "label": label, "max_cosine_to_sa": maximum,
                          "nearest_sa_stream": stream, "exact_prefix_match": exact, "duplicate": duplicate})
    return human, human_embeddings, table, summary


def detector_metrics(duplicate: np.ndarray, label: np.ndarray):
    rows = []
    realworld = pd.read_csv(RESULTS / "revision_realworld_probabilities.csv")
    realworld = realworld[realworld["condition"] == "human_phishing"]
    for seed, block in realworld.groupby("seed"):
        block = block.sort_values("row_index")
        for column in [c for c in block.columns if c.startswith("probability_")]:
            rows.append((column.replace("probability_", ""), int(seed), block[column].to_numpy()))
    for path in ("revision_distilbert_stream_probabilities.csv", "revision_r4_distilbert_multisplit_probabilities.csv"):
        table = pd.read_csv(RESULTS / path)
        table = table[table["condition"] == "human_phishing"]
        for (seed, training), block in table.groupby(["seed", "training"]):
            block = block.sort_values("row_index")
            rows.append((f"distilbert_{training.replace('-', '_')}", int(seed), block["probability"].to_numpy()))
    out = []
    for model, seed, probability in rows:
        if len(probability) != len(label):
            raise ValueError(f"{model} seed {seed}: {len(probability)} rows, expected {len(label)}")
        for scope, keep in (("full", np.ones(len(label), dtype=bool)), ("dedup", ~duplicate)):
            y, p = label[keep], probability[keep]
            out.append({"model": model, "seed": seed, "scope": scope, "n": int(keep.sum()),
                        "prevalence": float(y.mean()),
                        "f1_0.5": float(f1_score(y, p >= 0.5)), "recall_0.5": float((p[y == 1] >= 0.5).mean()),
                        "fpr_0.5": float((p[y == 0] >= 0.5).mean()), "roc_auc": float(roc_auc_score(y, p)),
                        "pr_auc": float(average_precision_score(y, p)), "ceiling_f1": oracle_threshold_f1(y, p)[0]})
    return pd.DataFrame(out)


def ceiling(human, human_embeddings, duplicate):
    primary, external, binary, spam = load_corpora()
    out = []
    texts = human["text"].tolist()
    label = human["label"].to_numpy()
    for scope, keep in (("full", np.ones(len(label), dtype=bool)), ("dedup", ~duplicate)):
        index = np.flatnonzero(keep)
        components = connected_components(human_embeddings[index], 0.90)
        for seed in SEEDS:
            train, test = component_split(components, label[index], seed)
            tr, te = index[train], index[test]
            word = base.WordModel.fit([texts[i] for i in tr], label[tr])
            p = word.predict_proba([texts[i] for i in te])
            frames = split_frames(binary, external, spam, seed)
            synthetic = base.WordModel.fit(frames["train_all"]["text"].tolist(), frames["train_all"]["label"].to_numpy())
            q = synthetic.predict_proba([texts[i] for i in te])
            out.append({"scope": scope, "seed": seed, "components": int(len(np.unique(components))),
                        "n_train": int(len(tr)), "n_test": int(len(te)),
                        "ceiling_f1": float(f1_score(label[te], p >= 0.5)),
                        "synthetic_f1": float(f1_score(label[te], q >= 0.5)),
                        "synthetic_recall": float((q[label[te] == 1] >= 0.5).mean()),
                        "synthetic_oracle_f1": oracle_threshold_f1(label[te], q)[0],
                        "synthetic_pr_auc": float(average_precision_score(label[te], q))})
    return pd.DataFrame(out)


def main() -> None:
    human, human_embeddings, table, summary = overlap()
    table.to_csv(RESULTS / "revision_r4_l3_sa_duplicates.csv", index=False)
    duplicate = table["duplicate"].to_numpy()
    metrics = detector_metrics(duplicate, human["label"].to_numpy())
    metrics.to_csv(RESULTS / "revision_r4_l3_dedup_metrics.csv", index=False)
    ceilings = ceiling(human, human_embeddings, duplicate)
    ceilings.to_csv(RESULTS / "revision_r4_l3_dedup_ceiling.csv", index=False)
    summary["metrics_mean"] = (metrics.groupby(["model", "scope"])[["f1_0.5", "recall_0.5", "roc_auc", "pr_auc",
                                                                    "ceiling_f1"]].mean().round(4)
                               .reset_index().to_dict(orient="records"))
    summary["ceiling_mean"] = ceilings.groupby("scope").mean(numeric_only=True).round(4).reset_index().to_dict(orient="records")
    (RESULTS / "revision_r4_l3_sa_overlap.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.set_option("display.width", 250)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("metrics_mean", "ceiling_mean")}, indent=2))
    print(metrics.groupby(["model", "scope"])[["n", "f1_0.5", "recall_0.5", "roc_auc", "pr_auc", "ceiling_f1"]]
          .mean().round(4).to_string())
    print(ceilings.groupby("scope").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
