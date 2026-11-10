#!/usr/bin/env python3
"""DistilBERT diagnostics requested in revision round 2.

Three questions are addressed experimentally rather than by interpretation:

1. *Calibration pattern.* Probabilities are stored for the linked and external
   conditions so reliability curves and confidence histograms can be drawn.
2. *Overfitting inflection.* Training runs for two epochs with a checkpoint
   every tenth of an epoch, evaluated on validation, linked, external, and spam
   streams, so the trajectories can be compared directly.
3. *Truncation and training length.* Every configuration is repeated at
   maximum sequence length 160 (the pre-registered setting) and 320.

The end-of-epoch-1 checkpoint reproduces the pre-specified configuration
reported in the main tables.
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

MODEL_NAME = "distilbert-base-uncased"
EVAL_CONDITIONS = ["validation", "linked_rewrites", "external_openai", "spam_hard_negative"]


def encode(tokenizer, texts, max_length):
    batch = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return batch["input_ids"], batch["attention_mask"]


def predict(model, ids, mask, device, batch_size=128):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            logits = model(
                input_ids=ids[start : start + batch_size].to(device),
                attention_mask=mask[start : start + batch_size].to(device),
            ).logits
            output.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    model.train()
    return np.concatenate(output)


def evaluate(y_true, probability, is_spam):
    if is_spam:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "fpr_at_0_5": float((probability >= 0.5).mean()),
        }
    result = base.metrics(y_true, probability)
    negative = y_true == 0
    result["fpr_at_0_5"] = float((probability[negative] >= 0.5).mean())
    return result


def truncation_audit(tokenizer, frames, external):
    rows = []
    sources = {
        "primary_canonical": frames["clean"]["text"],
        "primary_linked": frames["linked_rewrites"]["text"],
        "external_openai": external["text"],
    }
    for name, texts in sources.items():
        lengths = np.array(
            [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts]
        )
        rows.append(
            {
                "corpus": name,
                "n": int(len(lengths)),
                "median_tokens": float(np.median(lengths)),
                "mean_tokens": float(lengths.mean()),
                "p90_tokens": float(np.percentile(lengths, 90)),
                "max_tokens": int(lengths.max()),
                "fraction_truncated_at_160": float((lengths > 160).mean()),
                "fraction_truncated_at_320": float((lengths > 320).mean()),
                "mean_retained_fraction_at_160": float(
                    np.minimum(lengths, 160).sum() / lengths.sum()
                ),
                "mean_retained_fraction_at_320": float(
                    np.minimum(lengths, 320).sum() / lengths.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=TRANSFORMER_SEEDS)
    parser.add_argument("--max-lengths", nargs="+", type=int, default=[160, 320])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoints-per-epoch", type=int, default=10)
    args = parser.parse_args()

    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    primary, external, binary, spam = load_corpora()

    curve_rows: list[dict] = []
    probability_rows: list[pd.DataFrame] = []
    audit_written = False

    for max_length in args.max_lengths:
        for seed in args.seeds:
            frames = split_frames(binary, external, spam, seed)
            if not audit_written:
                truncation_audit(tokenizer, frames, external).to_csv(
                    RESULTS / "revision_distilbert_truncation_audit.csv", index=False
                )
                audit_written = True

            encoded = {}
            labels = {}
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
                base_module = base
                base_module  # keep reference for linters
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=2
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
                eval_every = max(
                    1, int(round(steps_per_epoch / args.checkpoints_per_epoch))
                )
                generator = torch.Generator().manual_seed(seed)
                print(
                    f"max_len={max_length} seed={seed} {training_name} "
                    f"n={n_train} steps/epoch={steps_per_epoch}",
                    flush=True,
                )

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
                        probability = {
                            condition: predict(
                                model, *encoded[condition], device
                            )
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
                                    "max_length": max_length,
                                    "seed": seed,
                                    "training": training_name,
                                    "step": global_step,
                                    "epoch_progress": global_step / steps_per_epoch,
                                    "examples_seen": global_step * args.batch_size,
                                    "train_rows": n_train,
                                    "eval_condition": condition,
                                    "n": int(len(labels[condition])),
                                    "threshold_95_recall": threshold,
                                    **result,
                                }
                            )
                        if end_of_epoch:
                            for condition in ["linked_rewrites", "external_openai"]:
                                probability_rows.append(
                                    pd.DataFrame(
                                        {
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
                                f"  epoch {epoch + 1}: "
                                f"linked F1={curve_rows[-3]['f1']:.4f} "
                                f"external F1={curve_rows[-2]['f1']:.4f} "
                                f"external ECE={curve_rows[-2]['ece']:.4f}",
                                flush=True,
                            )
                del model, optimizer, train_ids, train_mask
                if device.type == "mps":
                    torch.mps.empty_cache()

    pd.DataFrame(curve_rows).to_csv(
        RESULTS / "revision_distilbert_learning_curve.csv", index=False
    )
    pd.concat(probability_rows, ignore_index=True).to_csv(
        RESULTS / "revision_distilbert_probabilities.csv", index=False
    )
    (RESULTS / "revision_distilbert_run_info.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "seeds": args.seeds,
                "max_lengths": args.max_lengths,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "checkpoints_per_epoch": args.checkpoints_per_epoch,
                "device": str(device),
                "wall_clock_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
