#!/usr/bin/env python3
"""Twenty independently refitted grouped partitions for the lightweight families.

Reviewer comment 2 (round 3): inference should reflect variability from
alternative data partitions and repeated model fitting, not only resampling
within one pre-designated split. The previously submitted version answered this
with a two-stage bootstrap over five refitted partitions; with five outer units
the outer resampling stage is coarse, which the manuscript acknowledged. This
script removes that caveat for the two lightweight families by refitting every
estimator on twenty independent source-grouped partitions.

Every split is drawn with the study's own splitter (``split_groups``), so each
seed yields a different 70/15/15 partition *by source group*, and every model is
refitted from scratch on that partition's training groups.

Outputs (results/):
  revision_r3_multisplit_metrics.csv        per split x model x condition
  revision_r3_multisplit_probabilities.csv  per row, long format, all splits
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from rev_common import (
    EVALUATION_CONDITIONS,
    RESULTS,
    base,
    load_corpora,
    load_embeddings,
    split_frames,
)

# The five published partitions first, then fifteen added in this revision.
SPLIT_SEEDS = [
    17, 29, 42, 71, 113,
    211, 307, 419, 523, 631,
    743, 857, 971, 1087, 1201,
    1319, 1433, 1553, 1667, 1789,
]


def main() -> None:
    started = time.time()
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)

    metric_rows: list[dict[str, object]] = []
    probability_frames: list[pd.DataFrame] = []

    for order, seed in enumerate(SPLIT_SEEDS):
        frames = split_frames(binary, external, spam, seed)
        train_all = frames["train_all"]
        train_clean = frames["train_clean"]

        fit_seconds: dict[str, float] = {}
        models: dict[str, object] = {}
        for name, builder in (
            ("Word LR, clean-only", lambda: base.WordModel.fit(
                train_clean["text"].tolist(), train_clean["label"].to_numpy())),
            ("Word LR, linked-augmented", lambda: base.WordModel.fit(
                train_all["text"].tolist(), train_all["label"].to_numpy())),
            ("Char+word LR, linked-augmented", lambda: base.CharWordModel.fit(
                train_all["text"].tolist(), train_all["label"].to_numpy())),
            ("MiniLM LR, clean-only", lambda: base.EmbeddingModel.fit(
                primary_embeddings[train_clean.index.to_numpy()],
                train_clean["label"].to_numpy())),
            ("MiniLM LR, linked-augmented", lambda: base.EmbeddingModel.fit(
                primary_embeddings[train_all.index.to_numpy()],
                train_all["label"].to_numpy())),
        ):
            fit_started = time.perf_counter()
            models[name] = builder()
            fit_seconds[name] = time.perf_counter() - fit_started

        def predict(model, condition, frame):
            if isinstance(model, base.EmbeddingModel):
                matrix = (
                    external_embeddings
                    if condition == "external_openai"
                    else primary_embeddings[frame.index.to_numpy()]
                )
                return model.predict_proba(matrix)
            return model.predict_proba(frame["text"].tolist())

        thresholds = {
            name: base.threshold_for_recall(
                frames["validation"]["label"].to_numpy().astype(int),
                predict(model, "validation", frames["validation"]),
            )
            for name, model in models.items()
        }

        for condition in EVALUATION_CONDITIONS:
            frame = frames[condition]
            labels = (
                np.zeros(len(frame), dtype=int)
                if condition == "spam_hard_negative"
                else frame["label"].to_numpy().astype(int)
            )
            block = pd.DataFrame(
                {
                    "seed": seed,
                    "split_order": order,
                    "condition": condition,
                    "group_id": frame["group_id"].astype(str).to_numpy(),
                    "label": labels,
                }
            )
            for name, model in models.items():
                probability = predict(model, condition, frame)
                block[f"probability_{base.model_slug(name)}"] = probability
                row = {
                    "seed": seed,
                    "split_order": order,
                    "model": name,
                    "condition": condition,
                    "n": len(frame),
                    "n_groups": int(frame["group_id"].nunique()),
                    "fit_seconds": fit_seconds[name],
                    "threshold_95_recall": thresholds[name],
                }
                if condition == "spam_hard_negative":
                    row.update(
                        {key: float("nan") for key in
                         ("precision", "recall", "f1", "brier", "ece",
                          "roc_auc", "pr_auc")}
                    )
                else:
                    row.update(base.metrics(labels, probability))
                row["fpr_at_0_5"] = float((probability[labels == 0] >= 0.5).mean())
                row["fpr_at_95_recall"] = float(
                    (probability[labels == 0] >= thresholds[name]).mean()
                )
                metric_rows.append(row)
            probability_frames.append(block)
        print(
            f"split {order + 1}/{len(SPLIT_SEEDS)} (seed {seed}) "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    pd.DataFrame(metric_rows).to_csv(
        RESULTS / "revision_r3_multisplit_metrics.csv", index=False
    )
    pd.concat(probability_frames, ignore_index=True).to_csv(
        RESULTS / "revision_r3_multisplit_probabilities.csv", index=False
    )
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
