#!/usr/bin/env python3
"""Is the 160- versus 320-token window effect specific to DistilBERT?

Round-4 comment R1.4. The round-3 manuscript reports that DistilBERT's
generator-disjoint (L2) augmentation deficit -- in ranking, calibration and
trajectory stability -- is present at a 160-token window and absent at 320, and
interprets that as a window effect. The reviewer asks whether the effect would
appear in another transformer with a similar sequence-length constraint, since
otherwise it could be a property of DistilBERT rather than of the window.

This script repeats the round-2 sequence-length experiment
(``rev_e_distilbert.py``) with a second backbone, ``FacebookAI/roberta-base``,
which differs from DistilBERT in tokenizer (byte-level BPE vs WordPiece),
pretraining corpus and objective, depth (12 vs 6 layers) and in not being
distilled, but shares its 512-position limit. Everything else is held fixed:
the same three source-grouped partitions (seeds 29, 42, 71), the same clean-only
and linked-augmented training sets, AdamW at lr 2e-5 with weight decay 0.01,
batch 32, gradient clipping 1.0, one epoch (the pre-registered stopping point),
and a checkpoint every tenth of an epoch evaluated on the validation, linked,
generator-disjoint and spam streams.

Outputs (results/):
  revision_r4_backbone_learning_curve.csv     every checkpoint x stream
  revision_r4_backbone_probabilities.csv      end-of-epoch probabilities
  revision_r4_backbone_truncation_audit.csv   token-length audit, this tokenizer
  revision_r4_backbone_run_info.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rev_common import RESULTS, TRANSFORMER_SEEDS, base, load_corpora, split_frames
from rev_e_distilbert import encode, evaluate, predict, truncation_audit

EVAL_CONDITIONS = ["validation", "linked_rewrites", "external_openai", "spam_hard_negative"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--seeds", nargs="+", type=int, default=TRANSFORMER_SEEDS)
    parser.add_argument("--max-lengths", nargs="+", type=int, default=[160, 320])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoints-per-epoch", type=int, default=10)
    parser.add_argument("--tag", default="backbone")
    args = parser.parse_args()

    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    primary, external, binary, spam = load_corpora()

    curve_rows: list[dict] = []
    probability_rows: list[pd.DataFrame] = []
    audit_written = False

    def save() -> None:
        pd.DataFrame(curve_rows).to_csv(
            RESULTS / f"revision_r4_{args.tag}_learning_curve.csv", index=False
        )
        if probability_rows:
            pd.concat(probability_rows, ignore_index=True).to_csv(
                RESULTS / f"revision_r4_{args.tag}_probabilities.csv",
                index=False,
                float_format="%.6g",
            )

    for max_length in args.max_lengths:
        for seed in args.seeds:
            frames = split_frames(binary, external, spam, seed)
            if not audit_written:
                audit = truncation_audit(tokenizer, frames, external)
                audit.insert(0, "model", args.model)
                audit.to_csv(
                    RESULTS / f"revision_r4_{args.tag}_truncation_audit.csv", index=False
                )
                audit_written = True

            encoded, labels = {}, {}
            for condition in EVAL_CONDITIONS:
                frame = frames[condition]
                encoded[condition] = encode(tokenizer, frame["text"], max_length)
                labels[condition] = (
                    np.zeros(len(frame), dtype=int)
                    if condition == "spam_hard_negative"
                    else frame["label"].to_numpy().astype(int)
                )

            for training_name, training_frame in [
                ("clean-only", frames["train_clean"]),
                ("linked-augmented", frames["train_all"]),
            ]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model, num_labels=2
                ).to(device)
                optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
                train_ids, train_mask = encode(
                    tokenizer, training_frame["text"], max_length
                )
                train_labels = torch.tensor(
                    training_frame["label"].to_numpy().astype(int), dtype=torch.long
                )
                n_train = len(train_ids)
                steps_per_epoch = int(np.ceil(n_train / args.batch_size))
                eval_every = max(1, int(round(steps_per_epoch / args.checkpoints_per_epoch)))
                generator = torch.Generator().manual_seed(seed)
                print(
                    f"{args.model} max_len={max_length} seed={seed} {training_name} "
                    f"n={n_train} steps/epoch={steps_per_epoch}",
                    flush=True,
                )
                fit_started = time.time()
                eval_seconds = 0.0
                global_step = 0
                model.train()
                for epoch in range(args.epochs):
                    order = torch.randperm(n_train, generator=generator)
                    for start in range(0, n_train, args.batch_size):
                        index = order[start : start + args.batch_size]
                        loss = model(
                            input_ids=train_ids[index].to(device),
                            attention_mask=train_mask[index].to(device),
                            labels=train_labels[index].to(device),
                        ).loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        global_step += 1

                        end_of_epoch = start + args.batch_size >= n_train
                        if global_step % eval_every and not end_of_epoch:
                            continue
                        eval_started = time.time()
                        probability = {
                            condition: predict(model, *encoded[condition], device)
                            for condition in EVAL_CONDITIONS
                        }
                        threshold = base.threshold_for_recall(
                            labels["validation"], probability["validation"]
                        )
                        for condition in EVAL_CONDITIONS:
                            result = evaluate(
                                labels[condition],
                                probability[condition],
                                condition == "spam_hard_negative",
                            )
                            negative = labels[condition] == 0
                            result["fpr_at_95_recall"] = float(
                                (probability[condition][negative] >= threshold).mean()
                                if condition != "spam_hard_negative"
                                else (probability[condition] >= threshold).mean()
                            )
                            curve_rows.append(
                                {
                                    "model": args.model,
                                    "max_length": max_length,
                                    "seed": seed,
                                    "training": training_name,
                                    "step": global_step,
                                    "epoch_progress": global_step / steps_per_epoch,
                                    "train_rows": n_train,
                                    "eval_condition": condition,
                                    "n": int(len(labels[condition])),
                                    "threshold_95_recall": threshold,
                                    **result,
                                }
                            )
                        eval_seconds += time.time() - eval_started
                        if end_of_epoch:
                            for condition in EVAL_CONDITIONS:
                                probability_rows.append(
                                    pd.DataFrame(
                                        {
                                            "model": args.model,
                                            "max_length": max_length,
                                            "seed": seed,
                                            "training": training_name,
                                            "epoch": epoch + 1,
                                            "condition": condition,
                                            "label": labels[condition],
                                            "probability": probability[condition],
                                        }
                                    )
                                )
                            print(
                                f"  epoch {epoch + 1}: linked F1="
                                f"{curve_rows[-3]['f1']:.4f} external F1="
                                f"{curve_rows[-2]['f1']:.4f} external AUC="
                                f"{curve_rows[-2]['roc_auc']:.4f} external ECE="
                                f"{curve_rows[-2]['ece']:.4f} "
                                f"fit={time.time() - fit_started - eval_seconds:.0f}s",
                                flush=True,
                            )
                del model, optimizer, train_ids, train_mask
                if device.type == "mps":
                    torch.mps.empty_cache()
                save()

    save()
    (RESULTS / f"revision_r4_{args.tag}_run_info.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "seeds": args.seeds,
                "max_lengths": args.max_lengths,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": 2e-5,
                "checkpoints_per_epoch": args.checkpoints_per_epoch,
                "device": str(device),
                "wall_clock_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
