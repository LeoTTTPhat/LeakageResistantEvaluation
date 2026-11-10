#!/usr/bin/env python3
"""Dump per-split probabilities for every lightweight model and condition.

The published artifact stored seed-42 test probabilities only. The revision
analyses additionally need validation probabilities (for cost-sensitive
threshold selection) and every split (for multi-split inference), so this
script refits the same five estimators on all five grouped splits and writes a
single long-format probability table.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from rev_common import (
    EVALUATION_CONDITIONS,
    RESULTS,
    SEEDS,
    base,
    load_corpora,
    load_embeddings,
    split_frames,
)


def main() -> None:
    started = time.time()
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)

    records: list[pd.DataFrame] = []
    for seed in SEEDS:
        frames = split_frames(binary, external, spam, seed)
        train_all = frames["train_all"]
        train_clean = frames["train_clean"]

        models: dict[str, object] = {
            "Word LR, clean-only": base.WordModel.fit(
                train_clean["text"].tolist(), train_clean["label"].to_numpy()
            ),
            "Word LR, linked-augmented": base.WordModel.fit(
                train_all["text"].tolist(), train_all["label"].to_numpy()
            ),
            "Char+word LR, linked-augmented": base.CharWordModel.fit(
                train_all["text"].tolist(), train_all["label"].to_numpy()
            ),
            "MiniLM LR, clean-only": base.EmbeddingModel.fit(
                primary_embeddings[train_clean.index.to_numpy()],
                train_clean["label"].to_numpy(),
            ),
            "MiniLM LR, linked-augmented": base.EmbeddingModel.fit(
                primary_embeddings[train_all.index.to_numpy()],
                train_all["label"].to_numpy(),
            ),
        }

        for condition in EVALUATION_CONDITIONS:
            frame = frames[condition]
            block = pd.DataFrame(
                {
                    "seed": seed,
                    "condition": condition,
                    "row_index": np.arange(len(frame)),
                    "group_id": frame["group_id"].astype(str).to_numpy(),
                    "label": (
                        np.zeros(len(frame), dtype=int)
                        if condition == "spam_hard_negative"
                        else frame["label"].astype(int).to_numpy()
                    ),
                }
            )
            if "Entity_Type" in frame.columns:
                block["entity_type"] = frame["Entity_Type"].to_numpy()
                block["length_type"] = frame["Length_Type"].to_numpy()
            else:
                block["entity_type"] = "not_available"
                block["length_type"] = "not_available"

            for name, model in models.items():
                if isinstance(model, base.EmbeddingModel):
                    if condition == "external_openai":
                        matrix = external_embeddings
                    else:
                        matrix = primary_embeddings[frame.index.to_numpy()]
                    probability = model.predict_proba(matrix)
                else:
                    probability = model.predict_proba(frame["text"].tolist())
                block[f"probability_{base.model_slug(name)}"] = probability
            records.append(block)
        print(f"seed {seed} complete", flush=True)

    table = pd.concat(records, ignore_index=True)
    output = RESULTS / "revision_probabilities_all_seeds.csv"
    table.to_csv(output, index=False)
    print(f"wrote {output} rows={len(table)} seconds={time.time() - started:.1f}")


if __name__ == "__main__":
    main()
