#!/usr/bin/env python3
"""One hundred independently refitted grouped partitions for the lightweight families.

Round-4 comment R1.2. Moving from five to twenty partitions raised the estimated
between-partition standard deviation of the word model's augmentation effect by
a factor of 3.6, and the reviewer asks whether twenty is itself enough: either
add partitions or show that the interval width has stabilized. This script does
the first so that ``rev_r_r4_inference.py`` can do the second on data rather
than on an extrapolation. It adds eighty partitions to the twenty of
``rev_k_multisplit.py`` (seeds 2001-2080, chosen as a consecutive block so that
no seed is selected after seeing a result).

Every partition is drawn with the study's own splitter (``split_groups``), and
the four paired models (word TF-IDF and MiniLM, clean-only and
linked-augmented) are refitted from scratch on its training groups. The
augmented-only character+word model is not part of the paired inference and is
not refitted here.

Partitions are fitted in parallel worker processes, so the recorded fit times
are inflated by contention and are not used for any cost claim.

Outputs (results/):
  revision_r4_multisplit100_metrics.csv        per split x model x condition
  revision_r4_multisplit100_probabilities.csv  validation and linked rows only
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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

NEW_SEEDS = list(range(2001, 2081))
MODELS = [
    "Word LR, clean-only",
    "Word LR, linked-augmented",
    "MiniLM LR, clean-only",
    "MiniLM LR, linked-augmented",
]
STORED_CONDITIONS = ("validation", "linked_rewrites")

_STATE: dict[str, object] = {}


def _init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)
    _STATE.update(
        binary=binary,
        external=external,
        spam=spam,
        primary_embeddings=primary_embeddings,
        external_embeddings=external_embeddings,
    )


def run_partition(seed: int, order: int):
    binary = _STATE["binary"]
    external = _STATE["external"]
    spam = _STATE["spam"]
    primary_embeddings = _STATE["primary_embeddings"]
    external_embeddings = _STATE["external_embeddings"]

    frames = split_frames(binary, external, spam, seed)
    train_all, train_clean = frames["train_all"], frames["train_clean"]
    builders = {
        "Word LR, clean-only": lambda: base.WordModel.fit(
            train_clean["text"].tolist(), train_clean["label"].to_numpy()
        ),
        "Word LR, linked-augmented": lambda: base.WordModel.fit(
            train_all["text"].tolist(), train_all["label"].to_numpy()
        ),
        "MiniLM LR, clean-only": lambda: base.EmbeddingModel.fit(
            primary_embeddings[train_clean.index.to_numpy()],
            train_clean["label"].to_numpy(),
        ),
        "MiniLM LR, linked-augmented": lambda: base.EmbeddingModel.fit(
            primary_embeddings[train_all.index.to_numpy()],
            train_all["label"].to_numpy(),
        ),
    }
    models, fit_seconds = {}, {}
    for name in MODELS:
        fit_started = time.perf_counter()
        models[name] = builders[name]()
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

    validation = frames["validation"]
    thresholds = {
        name: base.threshold_for_recall(
            validation["label"].to_numpy().astype(int),
            predict(model, "validation", validation),
        )
        for name, model in models.items()
    }
    metric_rows, blocks = [], []
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
            block[f"probability_{base.model_slug(name)}"] = probability.astype(np.float32)
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
                    {
                        key: float("nan")
                        for key in (
                            "precision",
                            "recall",
                            "f1",
                            "brier",
                            "ece",
                            "roc_auc",
                            "pr_auc",
                        )
                    }
                )
            else:
                row.update(base.metrics(labels, probability))
            row["fpr_at_0_5"] = float((probability[labels == 0] >= 0.5).mean())
            row["fpr_at_95_recall"] = float(
                (probability[labels == 0] >= thresholds[name]).mean()
            )
            metric_rows.append(row)
        if condition in STORED_CONDITIONS:
            blocks.append(block)
    return metric_rows, pd.concat(blocks, ignore_index=True)


def main() -> None:
    started = time.time()
    workers = int(os.environ.get("REV_Q_WORKERS", "12"))
    metric_rows: list[dict] = []
    blocks: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futures = {
            pool.submit(run_partition, seed, 20 + index): seed
            for index, seed in enumerate(NEW_SEEDS)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            rows, block = future.result()
            metric_rows.extend(rows)
            blocks.append(block)
            print(
                f"{done}/{len(NEW_SEEDS)} partitions (seed {futures[future]}) "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
    metrics = pd.DataFrame(metric_rows).sort_values(["split_order", "model", "condition"])
    metrics.to_csv(RESULTS / "revision_r4_multisplit100_metrics.csv", index=False)
    probabilities = pd.concat(blocks, ignore_index=True).sort_values(
        ["split_order", "condition"], kind="stable"
    )
    probabilities.to_csv(
        RESULTS / "revision_r4_multisplit100_probabilities.csv",
        index=False,
        float_format="%.6g",
    )
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
