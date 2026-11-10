#!/usr/bin/env python3
"""Fit and inference timing for every family, measured together (round 4).

Round 3 compared fitting costs taken from different runs: the word model's fit
time from ``run_experiment.py`` and DistilBERT's from the original
``run_transformer.py`` run. Re-running DistilBERT in round 4 on the same machine
gave fit times about six times shorter than that run recorded, so the ratio the
manuscript quoted depended on when each number was taken. This script measures
all families in one session, on one partition (seed 42), sequentially, with the
system load recorded, so that the reported ratios compare like with like.

* Word TF-IDF and character+word models: fit (clean / augmented), five repeats.
* MiniLM: classifier fit from cached embeddings (five repeats) and, separately,
  the sentence encoding of the augmented training set and of the L2 corpus.
* DistilBERT: one-epoch fine-tune (clean / augmented), two repeats, with the
  pre-registered configuration; inference on the L2 corpus.
Inference throughput is measured on the 3,746-message generator-disjoint corpus.

Outputs (results/):
  revision_r4_timing.csv
  revision_r4_timing_run_info.json
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from rev_common import RESULTS, base, encode_minilm, load_corpora, load_embeddings, split_frames

sys.path.insert(0, str(base.HERE))
import run_transformer as rt  # noqa: E402

SEED = 42


def load_average() -> str:
    try:
        return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unavailable"


def main() -> None:
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)
    frames = split_frames(binary, external, spam, SEED)
    rows = []

    def record(family, condition, stage, seconds, n):
        rows.append({"family": family, "condition": condition, "stage": stage, "seconds": seconds, "n": n,
                     "per_second": n / seconds if seconds > 0 else np.nan})
        print(rows[-1], flush=True)

    texts_l2 = external["text"].tolist()
    for condition, frame in (("clean", frames["train_clean"]), ("augmented", frames["train_all"])):
        y = frame["label"].to_numpy()
        for _ in range(5):
            started = time.perf_counter()
            word = base.WordModel.fit(frame["text"].tolist(), y)
            record("Word TF-IDF", condition, "fit", time.perf_counter() - started, len(y))
        started = time.perf_counter()
        word.predict_proba(texts_l2)
        record("Word TF-IDF", condition, "infer_L2", time.perf_counter() - started, len(texts_l2))
        if condition == "augmented":
            for _ in range(3):
                started = time.perf_counter()
                charword = base.CharWordModel.fit(frame["text"].tolist(), y)
                record("Char+word TF-IDF", condition, "fit", time.perf_counter() - started, len(y))
            started = time.perf_counter()
            charword.predict_proba(texts_l2)
            record("Char+word TF-IDF", condition, "infer_L2", time.perf_counter() - started, len(texts_l2))
        x = primary_embeddings[frame.index.to_numpy()]
        for _ in range(5):
            started = time.perf_counter()
            embed = base.EmbeddingModel.fit(x, y)
            record("MiniLM", condition, "fit_classifier", time.perf_counter() - started, len(y))
        started = time.perf_counter()
        embed.predict_proba(external_embeddings)
        record("MiniLM", condition, "infer_L2_classifier", time.perf_counter() - started, len(texts_l2))

    # MiniLM encoding cost, measured on fresh text so that no cache is hit
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    salted = [t + " " for t in frames["train_all"]["text"].tolist()]
    started = time.perf_counter()
    encode_minilm(salted, device=device)
    record("MiniLM", "augmented", "encode_training_set", time.perf_counter() - started, len(salted))
    salted_l2 = [t + "  " for t in texts_l2]
    started = time.perf_counter()
    encode_minilm(salted_l2, device=device)
    record("MiniLM", "augmented", "encode_L2", time.perf_counter() - started, len(salted_l2))
    import hashlib

    for texts in (salted, salted_l2):  # remove the two caches this benchmark created
        fingerprint = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        (base.CACHE / f"minilm-revision-{fingerprint}.npy").unlink(missing_ok=True)

    torch_device = torch.device(device)
    tokenizer = rt.AutoTokenizer.from_pretrained(rt.MODEL_NAME)
    for condition, frame in (("clean", frames["train_clean"]), ("augmented", frames["train_all"])):
        for _ in range(2):
            started = time.perf_counter()
            model, _ = rt.train_one(frame, frames["validation"], tokenizer, seed=SEED, epochs=1,
                                    batch_size=32, device=torch_device)
            record("DistilBERT", condition, "fit", time.perf_counter() - started, len(frame))
        loader = torch.utils.data.DataLoader(rt.EmailDataset(external, tokenizer), batch_size=64, shuffle=False)
        started = time.perf_counter()
        rt.probabilities(model, loader, torch_device)
        record("DistilBERT", condition, "infer_L2", time.perf_counter() - started, len(texts_l2))
        del model
        if torch_device.type == "mps":
            torch.mps.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "revision_r4_timing.csv", index=False)
    summary = table.groupby(["family", "condition", "stage"])[["seconds", "per_second"]].median().reset_index()
    info = {
        "seed": SEED,
        "platform": platform.platform(),
        "processor": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                    text=True).stdout.strip() if sys.platform == "darwin" else platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "device": device,
        "load_before": load_average(),
        "note": "shared machine; other users' jobs were running; medians reported",
        "medians": summary.to_dict(orient="records"),
    }
    (RESULTS / "revision_r4_timing_run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
