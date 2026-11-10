#!/usr/bin/env python3
"""Fine-tune DistilBERT on clean-only and linked-rewrite training sets."""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from run_experiment import (
    DATA,
    RESULTS,
    canonical,
    load_external,
    load_primary,
    metrics,
    grouped_bootstrap_delta,
    model_slug,
    split_groups,
    threshold_for_recall,
)


MODEL_NAME = "distilbert-base-uncased"


class EmailDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 160):
        self.text = frame["text"].tolist()
        self.labels = (
            frame["label"].astype(int).tolist() if frame["label"].notna().all() else None
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.tokenizer(
            self.text[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        result = {key: value.squeeze(0) for key, value in item.items()}
        if self.labels is not None:
            result["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def probabilities(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            output.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(output)


def train_one(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tokenizer,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
):
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    train_loader = DataLoader(
        EmailDataset(train, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        EmailDataset(validation, tokenizer),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    best_f1 = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            if step % 100 == 0:
                print(
                    f"seed={seed} epoch={epoch + 1} step={step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )
        val_probability = probabilities(model, validation_loader, device)
        val_f1 = f1_score(
            validation["label"].to_numpy(), val_probability >= 0.5, zero_division=0
        )
        print(
            f"seed={seed} epoch={epoch + 1} validation_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("No transformer checkpoint was selected")
    model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[29, 42, 71])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-prefix", default="transformer")
    args = parser.parse_args()

    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    primary = load_primary().reset_index(drop=True)
    binary = primary[primary["label"].notna()].copy()
    binary["label"] = binary["label"].astype(int)
    spam = canonical(primary[primary["Type"].eq("Spam")].copy())
    external = load_external()

    rows = []
    cost_rows = []
    variant_rows = []
    seed42_predictions: dict[str, np.ndarray] = {}
    seed42_linked = None
    for seed in args.seeds:
        train_ids, val_ids, test_ids = split_groups(binary, seed)
        train_all = binary[binary["group_id"].isin(train_ids)]
        train_clean = canonical(train_all)
        validation = canonical(binary[binary["group_id"].isin(val_ids)])
        test_groups = binary[binary["group_id"].isin(test_ids)]
        test_clean = canonical(test_groups)
        test_linked = test_groups[
            ~(
                test_groups["Entity_Type"].eq("well_known")
                & test_groups["Length_Type"].eq("short")
            )
        ]
        conditions = {
            "clean": test_clean,
            "linked_rewrites": test_linked,
            "external_openai": external,
            "spam_hard_negative": spam,
        }
        for training_name, training_frame in [
            ("DistilBERT, clean-only", train_clean),
            ("DistilBERT, linked-augmented", train_all),
        ]:
            print(
                f"\n{training_name}, seed={seed}, n_train={len(training_frame)}",
                flush=True,
            )
            train_started = time.perf_counter()
            model, best_val_f1 = train_one(
                training_frame,
                validation,
                tokenizer,
                seed,
                args.epochs,
                args.batch_size,
                device,
            )
            cost_rows.append(
                {
                    "seed": seed,
                    "model": training_name,
                    "stage": "fine_tune",
                    "condition": "training",
                    "n": len(training_frame),
                    "seconds": time.perf_counter() - train_started,
                }
            )
            validation_probability = probabilities(
                model,
                DataLoader(
                    EmailDataset(validation, tokenizer),
                    batch_size=args.batch_size * 2,
                    shuffle=False,
                ),
                device,
            )
            recall_threshold = threshold_for_recall(
                validation["label"].to_numpy(), validation_probability
            )
            for condition, frame in conditions.items():
                predict_started = time.perf_counter()
                probability = probabilities(
                    model,
                    DataLoader(
                        EmailDataset(frame, tokenizer),
                        batch_size=args.batch_size * 2,
                        shuffle=False,
                    ),
                    device,
                )
                cost_rows.append(
                    {
                        "seed": seed,
                        "model": training_name,
                        "stage": "tokenize_and_predict",
                        "condition": condition,
                        "n": len(frame),
                        "seconds": time.perf_counter() - predict_started,
                    }
                )
                if condition == "spam_hard_negative":
                    result = {
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "f1": float("nan"),
                        "roc_auc": float("nan"),
                        "pr_auc": float("nan"),
                        "brier": float("nan"),
                        "ece": float("nan"),
                        "fpr_at_0_5": float((probability >= 0.5).mean()),
                        "fpr_at_95_recall": float(
                            (probability >= recall_threshold).mean()
                        ),
                    }
                else:
                    y_true = frame["label"].to_numpy().astype(int)
                    result = metrics(y_true, probability)
                    negative = y_true == 0
                    result["fpr_at_0_5"] = float(
                        (probability[negative] >= 0.5).mean()
                    )
                    result["fpr_at_95_recall"] = float(
                        (probability[negative] >= recall_threshold).mean()
                    )
                rows.append(
                    {
                        "seed": seed,
                        "model": training_name,
                        "condition": condition,
                        "n": len(frame),
                        "n_groups": frame["group_id"].nunique(),
                        "best_validation_f1": best_val_f1,
                        "threshold_95_recall": recall_threshold,
                        **result,
                    }
                )
                if condition == "linked_rewrites":
                    for (entity, message_length), subgroup in frame.groupby(
                        ["Entity_Type", "Length_Type"], sort=True
                    ):
                        positions = frame.index.get_indexer(subgroup.index)
                        variant_rows.append(
                            {
                                "seed": seed,
                                "model": training_name,
                                "entity_type": entity,
                                "length_type": message_length,
                                "n": len(subgroup),
                                "n_groups": subgroup["group_id"].nunique(),
                                **metrics(
                                    subgroup["label"].to_numpy().astype(int),
                                    probability[positions],
                                ),
                            }
                        )
                    if seed == 42:
                        seed42_predictions[training_name] = probability
                        seed42_linked = frame.copy()
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    detailed = pd.DataFrame(rows)
    detailed.to_csv(
        RESULTS / f"{args.output_prefix}_metrics_by_seed.csv", index=False
    )
    pd.DataFrame(cost_rows).to_csv(
        RESULTS / f"{args.output_prefix}_runtime_costs.csv", index=False
    )
    pd.DataFrame(variant_rows).to_csv(
        RESULTS / f"{args.output_prefix}_variant_metrics_by_seed.csv", index=False
    )
    numeric = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
        "ece",
        "fpr_at_0_5",
        "fpr_at_95_recall",
    ]
    summary = (
        detailed.groupby(["model", "condition"], as_index=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.drop(columns=["index"], errors="ignore")
    summary.to_csv(RESULTS / f"{args.output_prefix}_summary.csv", index=False)
    variant_summary = (
        pd.DataFrame(variant_rows)
        .groupby(["model", "entity_type", "length_type"], as_index=False)[
            ["precision", "recall", "f1", "brier", "ece"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    variant_summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in variant_summary.columns
    ]
    variant_summary = variant_summary.drop(columns=["index"], errors="ignore")
    variant_summary.to_csv(
        RESULTS / f"{args.output_prefix}_variant_summary.csv", index=False
    )
    if seed42_linked is not None:
        first = "DistilBERT, clean-only"
        second = "DistilBERT, linked-augmented"
        bootstrap = grouped_bootstrap_delta(
            seed42_linked,
            seed42_predictions[first],
            seed42_predictions[second],
            seed=42,
        )
        (RESULTS / f"{args.output_prefix}_paired_group_bootstrap.json").write_text(
            json.dumps({f"{second} minus {first}": bootstrap}, indent=2) + "\n",
            encoding="utf-8",
        )
        prediction_output = seed42_linked[
            ["group_id", "label", "Entity_Type", "Length_Type"]
        ].copy()
        prediction_output.columns = [
            "group_id",
            "label",
            "entity_type",
            "length_type",
        ]
        for model_name, probability in seed42_predictions.items():
            prediction_output[f"probability_{model_slug(model_name)}"] = probability
        prediction_output.to_csv(
            RESULTS / f"{args.output_prefix}_predictions_seed42.csv", index=False
        )
    (RESULTS / f"{args.output_prefix}_run_info.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "seeds": args.seeds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_sequence_length": 160,
                "device": str(device),
                "wall_clock_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
