#!/usr/bin/env python3
"""DistilBERT on the era-matched corpus and under the header-format control.

The transformer counterpart of ``rev_u_era_format.py`` (round-4 R1.9, R2.4).
For the five published source-grouped partitions, DistilBERT is fine-tuned with
the pre-registered configuration (one epoch, batch 32, lr 2e-5, length 160) in
two text formats -- as published (``Subject / Sender / body``) and body-only --
and every model scores the same streams as the lightweight script: L1, L2, the
L3 human-authored corpus, SpamAssassin ham and spam, and Nazario's 2015-2025
real phishing in the matching view (and, for the published format, also the
body-only view of Nazario, which isolates the header effect on one model).

Outputs (results/):
  revision_r4_era_format_distilbert_metrics.csv
  revision_r4_era_format_distilbert_by_year.csv
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from rev_common import RESULTS, base, split_frames
from rev_u_era_format import build_inputs, thresholds

sys.path.insert(0, str(base.HERE))
import run_transformer as rt  # noqa: E402

SEEDS = [17, 29, 42, 71, 113]


def scores_for(model, texts, tokenizer, device):
    frame = pd.DataFrame({"text": texts, "label": 0})
    loader = DataLoader(rt.EmailDataset(frame, tokenizer), batch_size=64, shuffle=False)
    return rt.probabilities(model, loader, device)


def main() -> None:
    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = rt.AutoTokenizer.from_pretrained(rt.MODEL_NAME)
    primary, external, binary, spam, human, sa, nazario, _ = build_inputs()
    metric_rows, year_rows = [], []
    for seed in SEEDS:
        frames = split_frames(binary, external, spam, seed)
        for fmt in ("published", "body_only"):
            column = "text" if fmt == "published" else "text_body"

            def view(frame):
                out = frame.copy()
                out["text"] = primary.loc[frame.index, column].to_numpy()
                return out

            sa_column = "text_full" if fmt == "published" else "text_body"
            streams = {
                "validation": (frames["validation"]["label"].to_numpy(), view(frames["validation"])["text"].tolist()),
                "L1_linked": (frames["linked_rewrites"]["label"].to_numpy(), view(frames["linked_rewrites"])["text"].tolist()),
                "L2": (external["label"].to_numpy().astype(int),
                       external["text" if fmt == "published" else "text_body"].tolist()),
                "L3_human": (human["label"].to_numpy(), human["text"].tolist()),
                "sa_ham": (np.zeros(int((sa["stream"] == "ham").sum()), dtype=int),
                           sa.loc[sa["stream"] == "ham", sa_column].tolist()),
                "sa_spam": (np.zeros(int((sa["stream"] == "spam").sum()), dtype=int),
                            sa.loc[sa["stream"] == "spam", sa_column].tolist()),
                "nazario": (np.ones(len(nazario), dtype=int),
                            nazario["text_full" if fmt == "published" else "text_body"].tolist()),
            }
            if fmt == "published":
                streams["nazario_body_view"] = (np.ones(len(nazario), dtype=int), nazario["text_body"].tolist())
            for training, frame in (("clean", frames["train_clean"]), ("augmented", frames["train_all"])):
                fit_started = time.time()
                model, _ = rt.train_one(view(frame), view(frames["validation"]), tokenizer, seed=seed,
                                        epochs=1, batch_size=32, device=device)
                fit_seconds = time.time() - fit_started
                scored = {name: (y, scores_for(model, t, tokenizer, device)) for name, (y, t) in streams.items()}
                cut = thresholds(*scored["validation"])
                legit_pool = np.r_[scored["L3_human"][1][scored["L3_human"][0] == 0], scored["sa_ham"][1]]
                for name, (y, p) in scored.items():
                    row = {"seed": seed, "format": fmt, "family": "distilbert", "training": training,
                           "stream": name, "n": len(y), "fit_seconds": fit_seconds,
                           **{f"threshold_{k}": v for k, v in cut.items()}}
                    for key, t in cut.items():
                        if (y == 1).any():
                            row[f"recall_{key}"] = float((p[y == 1] >= t).mean())
                        if (y == 0).any():
                            row[f"fpr_{key}"] = float((p[y == 0] >= t).mean())
                    if (y == 1).any() and (y == 0).any():
                        row["roc_auc"] = float(roc_auc_score(y, p))
                        row["pr_auc"] = float(average_precision_score(y, p))
                        pred = p >= 0.5
                        row["f1"] = 2 * float((pred & (y == 1)).sum()) / max(1.0, pred.sum() + (y == 1).sum())
                    if name.startswith("nazario"):
                        row["roc_auc_vs_real_legit"] = float(roc_auc_score(
                            np.r_[np.ones(len(p)), np.zeros(len(legit_pool))], np.r_[p, legit_pool]))
                        for era, mask in (("2015-2022", nazario["year"] <= 2022), ("2023-2025", nazario["year"] >= 2023)):
                            m = mask.to_numpy()
                            for key, t in cut.items():
                                row[f"recall_{key}_{era}"] = float((p[m] >= t).mean())
                        for year, part in nazario.groupby("year"):
                            idx = part.index.to_numpy()
                            year_rows.append({"seed": seed, "format": fmt, "family": "distilbert",
                                              "training": training, "stream": name, "year": int(year),
                                              "n": len(idx), **{f"recall_{k}": float((p[idx] >= t).mean())
                                                                for k, t in cut.items()}})
                    metric_rows.append(row)
                print(f"seed={seed} {fmt} {training} fit={fit_seconds:.0f}s "
                      f"nazario recall@0.5={metric_rows[-1 if fmt == 'body_only' else -2]['recall_t0.5']:.3f} "
                      f"L3 recall@0.5={[r for r in metric_rows if r['stream'] == 'L3_human'][-1]['recall_t0.5']:.3f} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
        pd.DataFrame(metric_rows).to_csv(RESULTS / "revision_r4_era_format_distilbert_metrics.csv", index=False)
        pd.DataFrame(year_rows).to_csv(RESULTS / "revision_r4_era_format_distilbert_by_year.csv", index=False)
    (RESULTS / "revision_r4_era_format_distilbert_run_info.json").write_text(json.dumps(
        {"model": rt.MODEL_NAME, "seeds": SEEDS, "epochs": 1, "batch_size": 32, "max_sequence_length": 160,
         "device": str(device), "wall_clock_seconds": time.time() - started}, indent=2) + "\n")
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
