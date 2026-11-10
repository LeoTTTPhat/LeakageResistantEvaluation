#!/usr/bin/env python3
"""Extend the DistilBERT evidence from three grouped splits to eight.

Reviewer comment 2 (round 3) asks that inference reflect variability across
alternative grouped train/test partitions *and* repeated model fitting, not
only resampling within one partition. The lightweight families are extended to
twenty splits by ``rev_k_multisplit.py``; a fine-tuned transformer cannot be
refitted twenty times at reasonable cost, so this script adds the five
partitions needed to bring DistilBERT from three to eight.

Each split is an independent source-grouped partition (``split_groups``) with a
fresh fine-tune of both the clean-only and the linked-augmented condition, using
the pre-registered configuration of ``run_transformer.py``: distilbert-base-
uncased, one epoch, batch 32, lr 2e-5, max sequence length 160.

Outputs (results/):
  revision_r3_distilbert_multisplit_metrics.csv        per split x condition
  revision_r3_distilbert_multisplit_probabilities.csv  per row, with group_id
  revision_r3_distilbert_multisplit_run_info.json      device, timing, seeds
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

sys.path.insert(0, str(base.HERE))
import run_transformer as rt  # noqa: E402

# The three published splits plus the five added in this revision.
PUBLISHED_SEEDS = [29, 42, 71]
NEW_SEEDS = [17, 113, 211, 307, 419]
CONDITIONS = ["clean", "linked_rewrites", "external_openai", "spam_hard_negative"]


def evaluate(model, frame: pd.DataFrame, tokenizer, device, batch_size: int = 64):
    loader = DataLoader(
        rt.EmailDataset(frame.assign(label=frame["label"].fillna(0).astype(int)), tokenizer),
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

    metric_rows: list[dict[str, object]] = []
    probability_frames: list[pd.DataFrame] = []

    for seed in NEW_SEEDS:
        frames = split_frames(binary, external, spam, seed)
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
            validation_probability = evaluate(model, validation, tokenizer, device)
            threshold95 = base.threshold_for_recall(
                validation["label"].to_numpy().astype(int), validation_probability
            )
            for condition in CONDITIONS:
                frame = frames[condition]
                probability = evaluate(model, frame, tokenizer, device)
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
                if condition == "spam_hard_negative":
                    row.update(
                        {
                            "precision": float("nan"),
                            "recall": float("nan"),
                            "f1": float("nan"),
                            "brier": float("nan"),
                            "ece": float("nan"),
                            "roc_auc": float("nan"),
                            "pr_auc": float("nan"),
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
                            "group_id": frame["group_id"].astype(str).to_numpy(),
                            "label": labels,
                            "probability": probability,
                        }
                    )
                )
            print(
                f"seed={seed} training={training} fit={fit_seconds:.0f}s "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

        pd.DataFrame(metric_rows).to_csv(
            RESULTS / "revision_r3_distilbert_multisplit_metrics.csv", index=False
        )
        pd.concat(probability_frames, ignore_index=True).to_csv(
            RESULTS / "revision_r3_distilbert_multisplit_probabilities.csv", index=False
        )

    info = {
        "model": rt.MODEL_NAME,
        "published_seeds": PUBLISHED_SEEDS,
        "new_seeds": NEW_SEEDS,
        "total_splits": len(PUBLISHED_SEEDS) + len(NEW_SEEDS),
        "epochs": 1,
        "batch_size": 32,
        "max_sequence_length": 160,
        "device": str(device),
        "wall_clock_seconds": time.time() - started,
    }
    (RESULTS / "revision_r3_distilbert_multisplit_run_info.json").write_text(
        json.dumps(info, indent=2)
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
