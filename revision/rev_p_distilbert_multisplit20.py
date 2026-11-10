#!/usr/bin/env python3
"""Extend the DistilBERT evidence from eight grouped partitions to twenty.

Round-4 comments R1.2 and R2.2. Both reviewers note that the eight-partition
prediction interval for DistilBERT's augmentation effect, -0.016 to +0.126,
includes zero, and that the between-partition share of variance (87%) makes the
effect strongly dependent on partition composition. Round 3 stopped at eight
partitions because of fine-tuning cost. This script adds the twelve partitions
needed to put DistilBERT on exactly the same twenty source-grouped partitions as
the two lightweight families, so that all three families are compared on
identical splits and the exact sign-flip test can reach the same resolution
(2 / 2**20) for every family.

Each split is drawn with the study's splitter (``split_groups``), and both the
clean-only and the linked-augmented model are fine-tuned from scratch with the
pre-registered configuration of ``run_transformer.py`` (distilbert-base-uncased,
one epoch, batch 32, lr 2e-5, weight decay 0.01, clipping 1.0, max length 160),
through the same ``train_one`` routine that ``rev_k_distilbert_multisplit.py``
used for the five partitions added in round 3.

Unlike round 3, validation probabilities and the human-authored streams are also
stored, so that the deployment analyses can use these partitions as well.

Outputs (results/), rewritten after every partition so that an interrupted run
keeps what it finished:
  revision_r4_distilbert_multisplit_metrics.csv
  revision_r4_distilbert_multisplit_probabilities.csv
  revision_r4_distilbert_multisplit_run_info.json
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rev_common import RESULTS, base, load_corpora, split_frames
from rev_realworld_data import realworld_conditions

sys.path.insert(0, str(base.HERE))
import run_transformer as rt  # noqa: E402

# Partitions 1-8 already exist (three published, five added in round 3); these
# twelve complete the lightweight families' twenty.
EXISTING_SEEDS = [29, 42, 71, 17, 113, 211, 307, 419]
NEW_SEEDS = [523, 631, 743, 857, 971, 1087, 1201, 1319, 1433, 1553, 1667, 1789]
STUDY_CONDITIONS = [
    "validation",
    "clean",
    "linked_rewrites",
    "external_openai",
    "spam_hard_negative",
]
HUMAN_CONDITIONS = ["human_phishing", "spamassassin_ham", "spamassassin_spam"]


def evaluate(model, frame: pd.DataFrame, tokenizer, device, batch_size: int = 64):
    loader = DataLoader(
        rt.EmailDataset(
            frame.assign(label=frame["label"].fillna(0).astype(int)), tokenizer
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return rt.probabilities(model, loader, device)


def main() -> None:
    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)
    tokenizer = rt.AutoTokenizer.from_pretrained(rt.MODEL_NAME)

    primary, external, binary, spam = load_corpora()
    human = realworld_conditions()

    metric_rows: list[dict[str, object]] = []
    probability_frames: list[pd.DataFrame] = []
    completed: list[int] = []

    for seed in NEW_SEEDS:
        frames = split_frames(binary, external, spam, seed)
        frames.update(human)
        validation = frames["validation"]
        for training, train_frame in (
            ("clean-only", frames["train_clean"]),
            ("linked-augmented", frames["train_all"]),
        ):
            fit_started = time.time()
            model, best_validation_f1 = rt.train_one(
                train_frame,
                validation,
                tokenizer,
                seed=seed,
                epochs=1,
                batch_size=32,
                device=device,
            )
            fit_seconds = time.time() - fit_started
            scored: dict[str, np.ndarray] = {}
            for condition in STUDY_CONDITIONS + HUMAN_CONDITIONS:
                scored[condition] = evaluate(model, frames[condition], tokenizer, device)
            threshold95 = base.threshold_for_recall(
                validation["label"].to_numpy().astype(int), scored["validation"]
            )
            for condition, probability in scored.items():
                frame = frames[condition]
                labels = (
                    np.zeros(len(frame), dtype=int)
                    if condition == "spam_hard_negative"
                    else frame["label"].to_numpy().astype(int)
                )
                row = {
                    "seed": seed,
                    "training": training,
                    "condition": condition,
                    "n": len(frame),
                    "n_groups": frame["group_id"].nunique(),
                    "best_validation_f1": best_validation_f1,
                    "threshold_95_recall": threshold95,
                    "fit_seconds": fit_seconds,
                }
                if labels.max() == 0:
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
                    (probability[labels == 0] >= threshold95).mean()
                )
                metric_rows.append(row)
                probability_frames.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "training": training,
                            "condition": condition,
                            "row_index": np.arange(len(frame)),
                            "group_id": frame["group_id"].astype(str).to_numpy(),
                            "label": labels,
                            "probability": probability.astype(np.float32),
                        }
                    )
                )
            print(
                f"seed={seed} training={training} fit={fit_seconds:.0f}s "
                f"linked F1={metric_rows[-6]['f1']:.4f} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        completed.append(seed)
        pd.DataFrame(metric_rows).to_csv(
            RESULTS / "revision_r4_distilbert_multisplit_metrics.csv", index=False
        )
        pd.concat(probability_frames, ignore_index=True).to_csv(
            RESULTS / "revision_r4_distilbert_multisplit_probabilities.csv",
            index=False,
            float_format="%.6g",
        )
        info = {
            "model": rt.MODEL_NAME,
            "existing_seeds": EXISTING_SEEDS,
            "new_seeds_completed": completed,
            "total_splits": len(EXISTING_SEEDS) + len(completed),
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "max_sequence_length": 160,
            "device": str(device),
            "wall_clock_seconds": time.time() - started,
        }
        (RESULTS / "revision_r4_distilbert_multisplit_run_info.json").write_text(
            json.dumps(info, indent=2) + "\n"
        )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
