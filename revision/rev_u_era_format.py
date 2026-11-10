#!/usr/bin/env python3
"""Era-matched real phishing, and a header-format control (round-4 R1.9, R2.4).

Both reviewers note that the L3 collapse is a compound shift: the human-authored
corpus differs from the synthetic training data in authorship, but also in era
(2000s mail), formatting and topic. Reviewer 1 asks for an era-matched human
corpus or a detailed analysis of the differences. This script supplies the
first and the header half of the second; ``rev_t_human_error_analysis.py``
supplies the rest (length, topics, vocabulary, coverage).

1. Era.  Nazario's real-phishing corpus, 2015-2025 (``rev_era_data``), is scored
   by the same synthetic-trained detectors, separately for 2015-2022 (pre-LLM,
   treated as human-authored) and 2023-2025 (authorship unverifiable), and per
   year. The corpus is all-positive, so the era-matched quantity is recall at
   the deployed operating points: threshold 0.5, the validation 95%-recall
   threshold, and the validation cost-minimizing threshold at 1% prevalence and
   r = 10. Ranking metrics against the only real legitimate mail available (the
   2000s-era L3 legitimate messages and SpamAssassin ham) are reported as well,
   with that era mismatch on the negative side stated.

2. Header format.  Every synthetic message carries a ``Subject: ... / Sender:
   ...`` block (PhishFuzzer by construction, L2 in 97% of rows) while the L3
   human-authored corpus carries none. Two controls separate formatting from
   authorship:
   (a) the published detectors score Nazario both with its own headers rendered
       in the training layout and body-only;
   (b) every detector is refitted on body-only text (PhishFuzzer ``Body``; L2
       with its ``Subject:`` label removed) and evaluated on body-only versions
       of every stream, so that no corpus anywhere carries the header block.

Partitions: the twenty source-grouped partitions of ``rev_k_multisplit.py``.

Outputs (results/):
  revision_r4_era_format_metrics.csv     partition x format x model x stream
  revision_r4_era_by_year.csv            partition x format x model x year
  revision_r4_era_format_summary.csv     means across partitions
  revision_r4_era_corpus.json            corpus composition
"""

from __future__ import annotations

import email
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from rev_common import RESULTS, base, encode_minilm, load_corpora, load_embeddings, split_frames
from rev_era_data import load_era_matched
from rev_k_multisplit import SPLIT_SEEDS
from rev_realworld_data import DATA as RW_DATA
from rev_realworld_data import MAX_CHARS, MIN_CHARS, SPAMASSASSIN_FOLDERS, _body, load_human_phishing, load_spamassassin

PREVALENCE, COST_RATIO = 0.01, 10.0
GRID = np.unique(np.concatenate([np.linspace(0.0, 1.0, 2001), [0.5]]))
SUBJECT_LABEL = re.compile(r"^\s*\**\s*subject\s*:\s*\**\s*", re.I)
_STATE: dict[str, object] = {}


def spamassassin_body_only() -> pd.DataFrame:
    """SpamAssassin with the same cleaning as ``load_spamassassin`` but no header block."""
    rows = []
    for folder, stream in SPAMASSASSIN_FOLDERS:
        directory = RW_DATA / "spamassassin" / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.name.startswith("cmds"):
                continue
            try:
                message = email.message_from_bytes(path.read_bytes())
                subject = message.get("Subject", "") or ""
                sender = message.get("From", "") or ""
                body = _body(message)
            except Exception:  # pragma: no cover
                continue
            full = base.neutralize(f"Subject: {subject}\nSender: {sender}\n{body}")[:MAX_CHARS]
            if len(full) < MIN_CHARS:
                continue
            rows.append({"text_full": full, "text_body": base.neutralize(body)[:MAX_CHARS],
                         "stream": stream})
    frame = pd.DataFrame(rows).drop_duplicates(subset=["text_full"]).reset_index(drop=True)
    frame.loc[frame["text_body"].str.len() < MIN_CHARS, "text_body"] = "(empty)"
    return frame


def build_inputs():
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)
    primary["text_body"] = primary["Body"].map(base.neutralize)
    raw = pd.read_csv(base.DATA / "gpt_phishing_email.csv").dropna(subset=["content", "label"])
    external = external.copy()
    external["text_body"] = [base.neutralize(SUBJECT_LABEL.sub("", str(t), count=1))
                             for t in raw["content"]]
    human = load_human_phishing()
    sa = spamassassin_body_only()
    sa_check = load_spamassassin()
    if len(sa) != len(sa_check) or not (sa["text_full"].to_numpy() == sa_check["text"].to_numpy()).all():
        raise ValueError("SpamAssassin body-only view does not align with the published loader")
    nazario = load_era_matched()

    device = "mps" if os.environ.get("REV_U_MPS") else "cpu"
    embed = {
        ("primary", "published"): primary_embeddings,
        ("external", "published"): external_embeddings,
        ("primary", "body_only"): encode_minilm(primary["text_body"], device=device),
        ("external", "body_only"): encode_minilm(external["text_body"], device=device),
        ("human", "published"): encode_minilm(human["text"], device=device),
        ("sa", "published"): encode_minilm(sa["text_full"], device=device),
        ("sa", "body_only"): encode_minilm(sa["text_body"], device=device),
        ("nazario", "published"): encode_minilm(nazario["text_full"], device=device),
        ("nazario", "body_only"): encode_minilm(nazario["text_body"], device=device),
    }
    embed[("human", "body_only")] = embed[("human", "published")]
    return primary, external, binary, spam, human, sa, nazario, embed


def _init():
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    primary, external, binary, spam, human, sa, nazario, embed = build_inputs()
    _STATE.update(primary=primary, external=external, binary=binary, spam=spam, human=human,
                  sa=sa, nazario=nazario, embed=embed)


def thresholds(y, p):
    positives = np.sort(p[y == 1])
    t95 = float(positives[max(0, int(np.floor(0.05 * len(positives))))])
    tpr = (p[y == 1][None, :] >= GRID[:, None]).mean(1)
    fpr = (p[y == 0][None, :] >= GRID[:, None]).mean(1)
    cost = (1 - PREVALENCE) * fpr + PREVALENCE * COST_RATIO * (1 - tpr)
    return {"t0.5": 0.5, "t95R": t95, "tcost": float(GRID[int(np.argmin(cost))])}


def run_partition(seed):
    primary, external, binary, spam = (_STATE[k] for k in ("primary", "external", "binary", "spam"))
    human, sa, nazario, embed = (_STATE[k] for k in ("human", "sa", "nazario", "embed"))
    frames = split_frames(binary, external, spam, seed)
    metric_rows, year_rows = [], []
    for fmt in ("published", "body_only"):
        column = "text" if fmt == "published" else "text_body"

        def texts(frame):
            return primary.loc[frame.index, column].tolist()

        def vectors(frame):
            return embed[("primary", fmt)][frame.index.to_numpy()]

        models = {}
        for training, frame in (("clean", frames["train_clean"]), ("augmented", frames["train_all"])):
            y = frame["label"].to_numpy()
            models[("word", training)] = base.WordModel.fit(texts(frame), y)
            models[("minilm", training)] = base.EmbeddingModel.fit(vectors(frame), y)
        validation = frames["validation"]
        linked = frames["linked_rewrites"]
        sa_column = "text_full" if fmt == "published" else "text_body"
        naz_column = "text_full" if fmt == "published" else "text_body"
        streams = {
            "validation": (validation["label"].to_numpy(), texts(validation), vectors(validation)),
            "L1_linked": (linked["label"].to_numpy(), texts(linked), vectors(linked)),
            "L2": (external["label"].to_numpy().astype(int),
                   external["text" if fmt == "published" else "text_body"].tolist(),
                   embed[("external", fmt)]),
            "L3_human": (human["label"].to_numpy(), human["text"].tolist(), embed[("human", fmt)]),
            "sa_ham": (np.zeros(int((sa["stream"] == "ham").sum()), dtype=int),
                       sa.loc[sa["stream"] == "ham", sa_column].tolist(),
                       embed[("sa", fmt)][(sa["stream"] == "ham").to_numpy()]),
            "sa_spam": (np.zeros(int((sa["stream"] == "spam").sum()), dtype=int),
                        sa.loc[sa["stream"] == "spam", sa_column].tolist(),
                        embed[("sa", fmt)][(sa["stream"] == "spam").to_numpy()]),
            "nazario": (np.ones(len(nazario), dtype=int), nazario[naz_column].tolist(),
                        embed[("nazario", fmt)]),
        }
        if fmt == "published":
            streams["nazario_body_view"] = (np.ones(len(nazario), dtype=int),
                                            nazario["text_body"].tolist(),
                                            embed[("nazario", "body_only")])
        for (family, training), model in models.items():
            def score(stream):
                y, t, v = streams[stream]
                return y, (model.predict_proba(v) if family == "minilm" else model.predict_proba(t))

            yv, pv = score("validation")
            cut = thresholds(yv, pv)
            scored = {name: score(name) for name in streams}
            legit_pool = np.r_[scored["L3_human"][1][scored["L3_human"][0] == 0],
                               scored["sa_ham"][1]]
            for name, (y, p) in scored.items():
                row = {"seed": seed, "format": fmt, "family": family, "training": training,
                       "stream": name, "n": len(y), **{f"threshold_{k}": v for k, v in cut.items()}}
                for key, t in cut.items():
                    if (y == 1).any():
                        row[f"recall_{key}"] = float((p[y == 1] >= t).mean())
                    if (y == 0).any():
                        row[f"fpr_{key}"] = float((p[y == 0] >= t).mean())
                if (y == 1).any() and (y == 0).any():
                    row["roc_auc"] = float(roc_auc_score(y, p))
                    row["pr_auc"] = float(average_precision_score(y, p))
                    pred = p >= 0.5
                    tp = float((pred & (y == 1)).sum())
                    row["f1"] = 2 * tp / max(1.0, pred.sum() + (y == 1).sum())
                if name.startswith("nazario"):
                    truth = np.r_[np.ones(len(p)), np.zeros(len(legit_pool))]
                    combined = np.r_[p, legit_pool]
                    row["roc_auc_vs_real_legit"] = float(roc_auc_score(truth, combined))
                    for era, mask in (("2015-2022", nazario["year"] <= 2022),
                                      ("2023-2025", nazario["year"] >= 2023)):
                        m = mask.to_numpy()
                        row[f"roc_auc_vs_real_legit_{era}"] = float(roc_auc_score(
                            np.r_[np.ones(m.sum()), np.zeros(len(legit_pool))],
                            np.r_[p[m], legit_pool]))
                        for key, t in cut.items():
                            row[f"recall_{key}_{era}"] = float((p[m] >= t).mean())
                    for year, part in nazario.groupby("year"):
                        idx = part.index.to_numpy()
                        year_rows.append({"seed": seed, "format": fmt, "family": family,
                                          "training": training, "stream": name, "year": int(year),
                                          "n": len(idx),
                                          **{f"recall_{k}": float((p[idx] >= t).mean())
                                             for k, t in cut.items()},
                                          "median_probability": float(np.median(p[idx]))})
                if name == "L3_human":
                    row["median_probability_positive"] = float(np.median(p[y == 1]))
                metric_rows.append(row)
    return seed, metric_rows, year_rows


def main():
    started = time.time()
    # Build (and cache) every embedding once in the parent before forking.
    _, _, _, _, human, sa, nazario, _ = build_inputs()
    composition = {
        "nazario_messages": int(len(nazario)),
        "nazario_by_year": {int(k): int(v) for k, v in nazario["year"].value_counts().sort_index().items()},
        "nazario_pre_llm": int((nazario["year"] <= 2022).sum()),
        "nazario_llm_era": int((nazario["year"] >= 2023).sum()),
        "nazario_html_only_share": float((nazario["body_source"] == "text/html").mean()),
        "nazario_median_body_chars": float(nazario["text_body"].str.len().median()),
        "sa_body_only_empty": int((sa["text_body"] == "(empty)").sum()),
    }
    (RESULTS / "revision_r4_era_corpus.json").write_text(json.dumps(composition, indent=2) + "\n")
    print(composition, flush=True)
    workers = int(os.environ.get("REV_U_WORKERS", "10"))
    metric_rows, year_rows = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futures = [pool.submit(run_partition, seed) for seed in SPLIT_SEEDS]
        for done, future in enumerate(as_completed(futures), start=1):
            seed, m, y = future.result()
            metric_rows += m
            year_rows += y
            print(f"{done}/{len(SPLIT_SEEDS)} seed {seed} elapsed {time.time() - started:.0f}s",
                  flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "revision_r4_era_format_metrics.csv", index=False)
    years = pd.DataFrame(year_rows)
    years.to_csv(RESULTS / "revision_r4_era_by_year.csv", index=False)
    numeric = [c for c in metrics.columns if c not in ("seed", "format", "family", "training", "stream")]
    summary = metrics.groupby(["format", "family", "training", "stream"])[numeric].mean().reset_index()
    summary.to_csv(RESULTS / "revision_r4_era_format_summary.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["format", "family", "training", "stream", "recall_t0.5", "recall_tcost", "recall_t95R",
            "fpr_t0.5", "f1", "roc_auc", "pr_auc", "roc_auc_vs_real_legit"]
    print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
