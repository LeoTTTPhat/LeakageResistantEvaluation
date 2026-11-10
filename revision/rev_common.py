#!/usr/bin/env python3
"""Shared helpers for the revision experiments.

The revision scripts reuse the exact loaders, splitters, and estimators of
``run_experiment.py``. ``sentence_transformers`` is only required when the
MiniLM embedding cache is absent; a stub keeps the import working on
interpreters where the package is unavailable but the cache is present.
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY / "results"
CACHE = STUDY / "cache"
FIGURES = STUDY / "figures"

if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

try:  # pragma: no cover - exercised only on interpreters without the package
    import sentence_transformers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # noqa: D401 - placeholder
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "sentence-transformers is unavailable and the MiniLM cache is missing"
            )

    stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = stub

import run_experiment as base  # noqa: E402

SEEDS = base.SEEDS
TRANSFORMER_SEEDS = [29, 42, 71]
PRIMARY_SEED = base.PRIMARY_SEED


def load_corpora():
    """Return (primary, external, binary, spam_canonical) with positional indices."""
    primary = base.load_primary().reset_index(drop=True)
    external = base.load_external().reset_index(drop=True)
    binary = primary[primary["label"].notna()].copy()
    binary["label"] = binary["label"].astype(int)
    spam = base.canonical(primary[primary["Type"].eq("Spam")].copy())
    return primary, external, binary, spam


def load_embeddings(primary, external):
    """Load the pinned MiniLM cache and split it into primary/external blocks."""
    matrices = sorted(CACHE.glob("minilm-*.npy"))
    if not matrices:
        raise FileNotFoundError(f"No MiniLM cache under {CACHE}")
    matrix = np.load(matrices[0])
    expected = len(primary) + len(external)
    if matrix.shape[0] != expected:
        raise ValueError(
            f"Cache has {matrix.shape[0]} rows but the corpora need {expected}"
        )
    return matrix[: len(primary)], matrix[len(primary) :]


MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def encode_minilm(texts, batch_size: int = 64, device: str = "cpu"):
    """Mean-pooled, L2-normalized MiniLM embeddings with a disk cache.

    This reproduces ``SentenceTransformer(all-MiniLM-L6-v2).encode(...,
    normalize_embeddings=True)`` to within 3e-7 on the cached study corpora, so
    the revision corpora are embedded in the same space as the published run
    without requiring the sentence-transformers package.
    """
    import hashlib

    import torch
    from transformers import AutoModel, AutoTokenizer

    texts = list(texts)
    fingerprint = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"minilm-revision-{fingerprint}.npy"
    if path.exists():
        return np.load(path)

    tokenizer = AutoTokenizer.from_pretrained(MINILM_NAME)
    model = AutoModel.from_pretrained(MINILM_NAME).to(device).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.float().cpu().numpy())
    matrix = np.concatenate(chunks)
    np.save(path, matrix)
    return matrix


def split_frames(binary, external, spam, seed):
    """Reproduce the study's source-grouped partition for one seed."""
    train_ids, val_ids, test_ids = base.split_groups(binary, seed)
    train_all = binary[binary["group_id"].isin(train_ids)]
    train_clean = base.canonical(train_all)
    validation = base.canonical(binary[binary["group_id"].isin(val_ids)])
    test_groups = binary[binary["group_id"].isin(test_ids)]
    test_clean = base.canonical(test_groups)
    test_linked = test_groups[
        ~(
            test_groups["Entity_Type"].eq("well_known")
            & test_groups["Length_Type"].eq("short")
        )
    ].copy()
    return {
        "train_all": train_all,
        "train_clean": train_clean,
        "validation": validation,
        "clean": test_clean,
        "linked_rewrites": test_linked,
        "external_openai": external,
        "spam_hard_negative": spam,
    }


EVALUATION_CONDITIONS = [
    "validation",
    "clean",
    "linked_rewrites",
    "external_openai",
    "spam_hard_negative",
]

MODEL_ORDER = [
    "Word LR, clean-only",
    "Word LR, linked-augmented",
    "Char+word LR, linked-augmented",
    "MiniLM LR, clean-only",
    "MiniLM LR, linked-augmented",
]

PAIRS = [
    ("Word TF-IDF", "Word LR, clean-only", "Word LR, linked-augmented"),
    ("MiniLM", "MiniLM LR, clean-only", "MiniLM LR, linked-augmented"),
]
