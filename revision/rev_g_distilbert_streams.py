#!/usr/bin/env python3
"""DistilBERT probabilities for every evaluation stream, including validation.

The pre-registered configuration is reproduced (one epoch, maximum sequence
length 160, seeds 29/42/71). Validation probabilities are stored because the
cost-sensitive threshold analysis must select operating points without touching
any target stream, and the human-authored corpora are added so the transformer
appears in the real-attack comparison as well.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rev_common import RESULTS, TRANSFORMER_SEEDS, base, load_corpora, split_frames
from rev_realworld_data import realworld_conditions

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 160
BATCH_SIZE = 32


def encode(tokenizer, texts):
    batch = tokenizer(
        list(texts),
        truncation=True,
        max_length=MAX_LENGTH,
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
    return np.concatenate(output)


def main() -> None:
    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    primary, external, binary, spam = load_corpora()
    realworld = realworld_conditions()

    blocks: list[pd.DataFrame] = []
    for seed in TRANSFORMER_SEEDS:
        frames = split_frames(binary, external, spam, seed)
        streams = {
            "validation": frames["validation"],
            "clean": frames["clean"],
            "linked_rewrites": frames["linked_rewrites"],
            "external_openai": frames["external_openai"],
            "spam_hard_negative": frames["spam_hard_negative"],
            **realworld,
        }
        encoded = {name: encode(tokenizer, frame["text"]) for name, frame in streams.items()}

        for training_name, training_frame in [
            ("clean-only", frames["train_clean"]),
            ("linked-augmented", frames["train_all"]),
        ]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            ids, mask = encode(tokenizer, training_frame["text"])
            labels = torch.tensor(
                training_frame["label"].to_numpy().astype(int), dtype=torch.long
            )
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(ids), generator=generator)
            model.train()
            for start in range(0, len(ids), BATCH_SIZE):
                index = order[start : start + BATCH_SIZE]
                loss = model(
                    input_ids=ids[index].to(device),
                    attention_mask=mask[index].to(device),
                    labels=labels[index].to(device),
                ).loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for name, frame in streams.items():
                probability = predict(model, *encoded[name], device)
                blocks.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "training": training_name,
                            "condition": name,
                            "row_index": np.arange(len(frame)),
                            "group_id": frame["group_id"].astype(str).to_numpy(),
                            "label": (
                                np.zeros(len(frame), dtype=int)
                                if name == "spam_hard_negative"
                                else frame["label"].to_numpy().astype(int)
                            ),
                            "probability": probability,
                        }
                    )
                )
            print(f"seed={seed} {training_name} done", flush=True)
            del model, optimizer, ids, mask
            if device.type == "mps":
                torch.mps.empty_cache()

    table = pd.concat(blocks, ignore_index=True)
    table.to_csv(RESULTS / "revision_distilbert_stream_probabilities.csv", index=False)

    rows = []
    for (condition, seed, training), block in table.groupby(
        ["condition", "seed", "training"]
    ):
        y = block["label"].to_numpy()
        probability = block["probability"].to_numpy()
        if y.max() == 0:
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "training": training,
                    "n": len(y),
                    "fpr_at_0_5": float((probability >= 0.5).mean()),
                }
            )
        else:
            result = base.metrics(y, probability)
            result["fpr_at_0_5"] = float((probability[y == 0] >= 0.5).mean())
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "training": training,
                    "n": len(y),
                    **result,
                }
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "revision_distilbert_stream_metrics.csv", index=False)
    (RESULTS / "revision_distilbert_stream_run_info.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "seeds": TRANSFORMER_SEEDS,
                "epochs": 1,
                "max_sequence_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "device": str(device),
                "wall_clock_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    numeric = [c for c in detail.columns if c not in {"condition", "seed", "training"}]
    print(
        detail.groupby(["condition", "training"])[numeric].mean().to_string()
    )


if __name__ == "__main__":
    main()
