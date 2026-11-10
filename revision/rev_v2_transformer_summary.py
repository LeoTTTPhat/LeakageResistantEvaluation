#!/usr/bin/env python3
"""Derived numbers for the transformer section (round 4), from stored outputs.

Round-4 comment R1.4 asks that the account of DistilBERT's generator-disjoint
(L2) behaviour be stated as an interpretation rather than a mechanism, and
whether the 160- versus 320-token window effect is specific to DistilBERT;
Reviewer 2 notes that DistilBERT's augmentation effect is heterogeneous across
partitions. The manuscript answers both with three bodies of stored evidence:
the diagnostic re-runs on the three reference partitions (seeds 29, 42, 71;
``rev_e_distilbert.py``, ``rev_g_distilbert_streams.py``,
``rev_n_distilbert_mechanism.py``), the 20 refitted DistilBERT partitions
(``rev_g``, ``rev_k_distilbert_multisplit.py``,
``rev_p_distilbert_multisplit20.py``), and RoBERTa-base fine-tuned with
DistilBERT's recipe on the reference partitions (``rev_v_backbone_window.py``).

This script recomputes every derived number that section quotes: end-of-epoch
means and per-partition contrasts, within-epoch and two-epoch volatility,
temperatures fitted on validation and on L2, oracle thresholds, shares of
extreme probabilities, per-partition counts, and corrected resampled t-tests
over the refitted partitions. It reads result files only: no model is trained,
and neither torch nor ``rev_common`` (which imports sentence-transformers) is
imported, so it can run next to a timing benchmark. The helpers below
reimplement, line for line, the definitions of ``run_experiment.py`` (15-bin
ECE) and ``rev_n_distilbert_mechanism.py`` (temperature fit, oracle threshold).

Definitions
  End of the pre-registered epoch: ``epoch_progress == 1.0``. The optimizer has
    no schedule, so that checkpoint of a two-epoch run equals a one-epoch run.
  Within-epoch volatility: checkpoints with 0.4 < epoch_progress <= 1.0 (seven
    per run); two-epoch volatility (round-3 definition): 0.4 < progress <= 2.0.
    SD uses ddof=1; maximum drawdown is the largest fall below the running
    maximum inside the window; "checkpoints below" counts checkpoints more than
    0.05 below that running maximum. Values are per partition, then averaged.
  Temperature: minimizer of the mean log loss of sigmoid(logit(p)/T), T in
    [0.05, 20]; "L2 temperature" is fitted on L2 itself (needs target labels).
  Oracle threshold: best F1 over the 197-point grid 0.01..0.99 on L2.
  Corrected resampled t (Nadeau and Bengio, 2003): se = sd * sqrt(1/K +
    n_test/n_train) with n_test/n_train = 324/1509, df = K - 1, as in
    ``rev_r_r4_inference.py``.

Inputs (results/):
  revision_distilbert_learning_curve.csv          rev_e: reference partitions, 160/320, 2 epochs
  revision_distilbert_probabilities.csv           rev_e: end-of-epoch probabilities
  revision_distilbert_stream_metrics.csv          rev_g: reference partitions, every stream
  revision_distilbert_stream_probabilities.csv    rev_g: probabilities incl. validation
  revision_r3_distilbert_multisplit_metrics.csv   rev_k: five partitions
  revision_r3_distilbert_multisplit_probabilities.csv  (no validation stream)
  revision_r4_distilbert_multisplit_metrics.csv   rev_p: twelve partitions
  revision_r4_distilbert_multisplit_probabilities.csv  (with validation)
  revision_r3_distilbert_truncation_strata.csv    rev_n: L2 F1 by length quartile
  revision_distilbert_truncation_audit.csv        rev_e: DistilBERT tokenizer
  revision_r4_backbone_learning_curve.csv         rev_v: RoBERTa, 160/320, 1 epoch
  revision_r4_backbone_probabilities.csv          rev_v: end-of-epoch probabilities
  revision_r4_backbone_truncation_audit.csv       rev_v: RoBERTa tokenizer
  revision_r4_partition_effects.csv               rev_r: per-partition effects (word model)
  revision_r4_partition_effect_summary.csv        rev_r: cross-check of the tests below
  revision_r4_heterogeneity.json                  rev_r: random-effects I^2 (read, not recomputed)

Outputs (results/):
  revision_r4_backbone_summary.csv      DistilBERT and RoBERTa on the reference
                                        partitions: one row per backbone x
                                        window x partition (and mean) x training
                                        (clean-only, linked-augmented,
                                        aug_minus_clean)
  revision_r4_transformer_summary.csv   every other quoted quantity, long format:
                                        block, quantity, clean, augmented,
                                        contrast, n_positive, k, corrected_p,
                                        corrected_low, corrected_high, source
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

REFERENCE = [29, 42, 71]
CLEAN, AUG = "clean-only", "linked-augmented"
NB_FACTOR = 324 / 1509  # n_test / n_train, as in rev_r_r4_inference.py
EPSILON = 1e-6
L2 = "external_openai"
SPAM = "spam_hard_negative"
LINKED = "linked_rewrites"


# --------------------------------------------------------------------------
# metric helpers (same definitions as run_experiment.py / rev_n)
# --------------------------------------------------------------------------
def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p <= upper if upper == 1.0 else p < upper)
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def f1_at(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    predicted = p >= threshold
    tp = float(np.sum(predicted & (y == 1)))
    fp = float(np.sum(predicted & (y == 0)))
    fn = float(np.sum(~predicted & (y == 1)))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator > 0 else 0.0


def clip_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, EPSILON, 1.0 - EPSILON))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    def loss(t: float) -> float:
        q = np.clip(expit(z / t), EPSILON, 1.0 - EPSILON)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    return float(minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded").x)


GRID = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))


def oracle_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    scores = [f1_at(y, p, t) for t in GRID]
    index = int(np.argmax(scores))
    return float(GRID[index]), float(scores[index])


def corrected_t(deltas) -> dict[str, float]:
    d = np.asarray(deltas, dtype=float)
    k, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    se = sd * np.sqrt(1.0 / k + NB_FACTOR)
    t_crit = float(stats.t.ppf(0.975, k - 1))
    p = float(2 * (1 - stats.t.cdf(abs(mean / se), k - 1))) if se > 0 else 0.0
    return {"contrast": mean, "n_positive": int((d > 0).sum()), "k": k,
            "corrected_p": p, "corrected_low": mean - t_crit * se,
            "corrected_high": mean + t_crit * se}


def probability_block(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    z = clip_logit(p)
    t_l2 = fit_temperature(z, y)
    threshold, best = oracle_threshold(y, p)
    return {"l2_temperature": t_l2,
            "l2_ece_l2_scaled": ece(y, expit(z / t_l2)),
            "oracle_threshold": threshold, "oracle_threshold_f1": best,
            "share_below_0_05": float((p < 0.05).mean()),
            "share_above_0_95": float((p > 0.95).mean())}


def volatility(curve: pd.DataFrame, condition: str, high: float) -> dict[str, float]:
    block = curve[(curve["eval_condition"] == condition)].sort_values("step")
    if block["epoch_progress"].max() < high - 1e-9:
        return {}
    window = block[(block["epoch_progress"] > 0.4)
                   & (block["epoch_progress"] <= high + 1e-9)]["f1"].to_numpy()
    running = np.maximum.accumulate(window)
    return {"sd": float(window.std(ddof=1)), "drawdown": float((running - window).max()),
            "below_0_05": float(((running - window) > 0.05).sum())}


# --------------------------------------------------------------------------
# reference partitions: DistilBERT (rev_e) and RoBERTa (rev_v)
# --------------------------------------------------------------------------
def backbone_rows() -> pd.DataFrame:
    sources = {
        "DistilBERT": ("revision_distilbert_learning_curve.csv",
                       "revision_distilbert_probabilities.csv"),
        "RoBERTa": ("revision_r4_backbone_learning_curve.csv",
                    "revision_r4_backbone_probabilities.csv"),
    }
    rows = []
    for backbone, (curve_file, probability_file) in sources.items():
        curve = pd.read_csv(RESULTS / curve_file)
        probability = pd.read_csv(RESULTS / probability_file)
        probability = probability[probability["epoch"] == 1]
        for (length, seed, training), run in curve.groupby(["max_length", "seed", "training"]):
            end = run[np.isclose(run["epoch_progress"], 1.0)].set_index("eval_condition")
            if end.empty:
                continue
            row = {"backbone": backbone, "max_length": int(length), "seed": str(seed),
                   "training": training, "n_partitions": 1,
                   "linked_f1": end.at[LINKED, "f1"], "l2_f1": end.at[L2, "f1"],
                   "l2_precision": end.at[L2, "precision"], "l2_recall": end.at[L2, "recall"],
                   "l2_roc_auc": end.at[L2, "roc_auc"], "l2_ece": end.at[L2, "ece"],
                   "spam_fpr_at_0_5": end.at[SPAM, "fpr_at_0_5"]}
            for condition, tag in ((L2, "l2"), (LINKED, "linked")):
                for high, window in ((1.0, "epoch1"), (2.0, "two_epochs")):
                    for key, value in volatility(run, condition, high).items():
                        row[f"{tag}_f1_{key}_{window}"] = value
            second = run[np.isclose(run["epoch_progress"], 2.0)].set_index("eval_condition")
            if not second.empty:
                row["epoch2_linked_f1"] = second.at[LINKED, "f1"]
                row["epoch2_l2_f1"] = second.at[L2, "f1"]
                row["epoch2_l2_ece"] = second.at[L2, "ece"]
            stored = probability[(probability["max_length"] == length)
                                 & (probability["seed"] == seed)
                                 & (probability["training"] == training)]
            l2 = stored[stored["condition"] == L2]
            y, p = l2["label"].to_numpy().astype(int), l2["probability"].to_numpy()
            row.update(probability_block(y, p))
            validation = stored[stored["condition"] == "validation"]
            if len(validation):  # rev_v stores validation; rev_e does not
                t_val = fit_temperature(clip_logit(validation["probability"].to_numpy()),
                                        validation["label"].to_numpy().astype(int))
                row["validation_temperature"] = t_val
                row["l2_ece_validation_scaled"] = ece(y, expit(clip_logit(p) / t_val))
            rows.append(row)
    table = pd.DataFrame(rows)
    value_columns = [c for c in table.columns
                     if c not in ("backbone", "max_length", "seed", "training", "n_partitions")]
    out = [table]
    for (backbone, length), block in table.groupby(["backbone", "max_length"]):
        complete = block.groupby("seed")["training"].nunique().eq(2)
        seeds = complete[complete].index
        for training, part in block.groupby("training"):
            mean = part[value_columns].mean(numeric_only=True).to_dict()
            out.append(pd.DataFrame([{"backbone": backbone, "max_length": length, "seed": "mean",
                                      "training": training, "n_partitions": len(part), **mean}]))
        clean = block[block["training"] == CLEAN].set_index("seed").loc[seeds, value_columns]
        aug = block[block["training"] == AUG].set_index("seed").loc[seeds, value_columns]
        delta = (aug - clean).reset_index()
        delta.insert(0, "backbone", backbone)
        delta.insert(1, "max_length", length)
        delta.insert(3, "training", "aug_minus_clean")
        delta.insert(4, "n_partitions", 1)
        out.append(delta)
        out.append(pd.DataFrame([{"backbone": backbone, "max_length": length, "seed": "mean",
                                  "training": "aug_minus_clean", "n_partitions": len(seeds),
                                  **(aug - clean).mean().to_dict()}]))
        difference = aug - clean
        positive = (difference > 0).sum().astype(float).where(difference.notna().any(), np.nan)
        out.append(pd.DataFrame([{"backbone": backbone, "max_length": length,
                                  "seed": "count_positive", "training": "aug_minus_clean",
                                  "n_partitions": len(seeds), **positive.to_dict()}]))
    result = pd.concat(out, ignore_index=True)
    order = {"29": 0, "42": 1, "71": 2, "mean": 3, "count_positive": 4}
    result["_o"] = result["seed"].map(order)
    result["_t"] = result["training"].map({CLEAN: 0, AUG: 1, "aug_minus_clean": 2})
    result = result.sort_values(["backbone", "max_length", "_t", "_o"]).drop(columns=["_o", "_t"])
    return result.reset_index(drop=True)


# --------------------------------------------------------------------------
# everything else, long format
# --------------------------------------------------------------------------
records: list[dict] = []


def record(block, quantity, clean=np.nan, augmented=np.nan, source="", **extra):
    records.append({"block": block, "quantity": quantity, "clean": clean,
                    "augmented": augmented, "source": source, **extra})


def paired(block, quantity, frame, source):
    """frame: index seed, columns CLEAN/AUG."""
    frame = frame.dropna()
    record(block, quantity, frame[CLEAN].mean(), frame[AUG].mean(), source,
           **corrected_t(frame[AUG] - frame[CLEAN]))


def refit_metrics() -> pd.DataFrame:
    files = ["revision_distilbert_stream_metrics.csv",
             "revision_r3_distilbert_multisplit_metrics.csv",
             "revision_r4_distilbert_multisplit_metrics.csv"]
    frame = pd.concat([pd.read_csv(RESULTS / f) for f in files], ignore_index=True)
    return frame.drop_duplicates(["seed", "training", "condition"])


def refit_probabilities() -> pd.DataFrame:
    files = ["revision_distilbert_stream_probabilities.csv",
             "revision_r3_distilbert_multisplit_probabilities.csv",
             "revision_r4_distilbert_multisplit_probabilities.csv"]
    keep = ("seed", "training", "condition", "label", "probability")
    frames = []
    for name in files:
        frame = pd.read_csv(RESULTS / name, usecols=lambda c: c in keep)
        frame = frame[frame["condition"].isin(["validation", L2])]
        frame["file"] = name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    first = frame.groupby("seed")["file"].first()
    return frame[frame["file"] == frame["seed"].map(first)]


def twenty_partitions(metrics: pd.DataFrame, probabilities: pd.DataFrame) -> None:
    src_m = "rev_g + rev_k + rev_p metrics"
    src_p = "rev_g + rev_k + rev_p probabilities"
    seeds = sorted(metrics["seed"].unique())
    assert len(seeds) == 20, seeds
    block = "DistilBERT, 20 refitted partitions"
    for condition, label in ((LINKED, "linked"), (L2, "L2")):
        part = metrics[metrics["condition"] == condition]
        for metric in ("f1", "roc_auc", "pr_auc", "ece", "precision", "recall"):
            paired(block, f"{label} {metric}", part.pivot(index="seed", columns="training",
                                                          values=metric), src_m)
    spam = metrics[metrics["condition"] == SPAM]
    paired(block, "spam FPR@0.5", spam.pivot(index="seed", columns="training",
                                             values="fpr_at_0_5"), src_m)
    roc = metrics[metrics["condition"] == L2].pivot(index="seed", columns="training",
                                                     values="roc_auc")
    delta = (roc[AUG] - roc[CLEAN]).sort_values()
    for rank, (seed, value) in enumerate(delta.items(), start=1):
        if rank <= 6 or seed in REFERENCE:
            record(block, f"L2 ROC-AUC effect, rank {rank} (ascending), seed {seed}",
                   source=src_m, contrast=value)
    l2 = metrics[metrics["condition"] == L2]
    for metric in ("roc_auc", "precision", "f1", "ece", "pr_auc"):
        piv = l2.pivot(index="seed", columns="training", values=metric)
        d = piv[AUG] - piv[CLEAN]
        record("DistilBERT, reference partitions within the 20", f"L2 {metric}",
               piv.loc[REFERENCE, CLEAN].mean(), piv.loc[REFERENCE, AUG].mean(), src_m,
               contrast=d.loc[REFERENCE].mean(), k=3)
        record("DistilBERT, other 17 partitions", f"L2 {metric}",
               piv.drop(REFERENCE)[CLEAN].mean(), piv.drop(REFERENCE)[AUG].mean(), src_m,
               contrast=d.drop(REFERENCE).mean(), k=17)

    per = []
    for (seed, training), run in probabilities.groupby(["seed", "training"]):
        l2p = run[run["condition"] == L2]
        y, p = l2p["label"].to_numpy().astype(int), l2p["probability"].to_numpy()
        z = clip_logit(p)
        row = {"seed": seed, "training": training, "raw_ece": ece(y, p), "raw_f1": f1_at(y, p, 0.5),
               **probability_block(y, p)}
        validation = run[run["condition"] == "validation"]
        if len(validation):
            t_val = fit_temperature(clip_logit(validation["probability"].to_numpy()),
                                    validation["label"].to_numpy().astype(int))
            row["validation_temperature"] = t_val
            row["l2_ece_validation_scaled"] = ece(y, expit(z / t_val))
        per.append(row)
    per = pd.DataFrame(per)
    assert per["seed"].nunique() == 20
    for key, label in (("share_below_0_05", "L2 share of probabilities < 0.05"),
                       ("share_above_0_95", "L2 share of probabilities > 0.95"),
                       ("l2_temperature", "L2 temperature (fitted on L2)"),
                       ("raw_f1", "L2 F1 at 0.5 (from probabilities)"),
                       ("oracle_threshold_f1", "L2 F1 at the oracle threshold")):
        paired(block, label, per.pivot(index="seed", columns="training", values=key), src_p)

    with_val = per.dropna(subset=["validation_temperature"])
    seeds15 = with_val.groupby("seed")["training"].nunique()
    seeds15 = seeds15[seeds15 == 2].index
    v = with_val[with_val["seed"].isin(seeds15)]
    b15 = f"DistilBERT, {len(seeds15)} partitions with stored validation probabilities"
    pv = {key: v.pivot(index="seed", columns="training", values=key)
          for key in ("validation_temperature", "l2_temperature", "raw_ece",
                      "l2_ece_validation_scaled", "l2_ece_l2_scaled")}
    for key, frame in pv.items():
        record(b15, key, frame[CLEAN].mean(), frame[AUG].mean(), src_p, k=len(seeds15))
    for training in (CLEAN, AUG):
        n = int((pv["l2_temperature"][training] > pv["validation_temperature"][training]).sum())
        record(b15, f"count L2 temperature > validation temperature ({training})",
               source=src_p, n_positive=n, k=len(seeds15))
    n = int((pv["l2_ece_validation_scaled"][AUG] > pv["raw_ece"][CLEAN]).sum())
    record(b15, "count augmented validation-scaled ECE > clean unscaled ECE",
           source=src_p, n_positive=n, k=len(seeds15))
    record(b15, "seeds", source=" ".join(str(s) for s in seeds15))


def reference_stored_scores(metrics: pd.DataFrame) -> None:
    """rev_n's tests on rev_g's stored probabilities, and rev_e vs rev_g agreement."""
    block = "DistilBERT, reference partitions, stored scores (rev_g)"
    table = pd.read_csv(RESULTS / "revision_distilbert_stream_probabilities.csv",
                        usecols=["seed", "training", "condition", "label", "probability"])
    per = []
    for (seed, training), run in table.groupby(["seed", "training"]):
        l2p = run[run["condition"] == L2]
        y, p = l2p["label"].to_numpy().astype(int), l2p["probability"].to_numpy()
        z = clip_logit(p)
        validation = run[run["condition"] == "validation"]
        t_val = fit_temperature(clip_logit(validation["probability"].to_numpy()),
                                validation["label"].to_numpy().astype(int))
        per.append({"seed": seed, "training": training, "raw_f1": f1_at(y, p, 0.5),
                    "raw_ece": ece(y, p), "validation_temperature": t_val,
                    "l2_ece_validation_scaled": ece(y, expit(z / t_val)),
                    **probability_block(y, p)})
    per = pd.DataFrame(per)
    for key in ("raw_f1", "raw_ece", "validation_temperature", "l2_temperature",
                "l2_ece_validation_scaled", "l2_ece_l2_scaled", "oracle_threshold",
                "oracle_threshold_f1"):
        frame = per.pivot(index="seed", columns="training", values=key)
        record(block, key, frame[CLEAN].mean(), frame[AUG].mean(),
               "revision_distilbert_stream_probabilities.csv",
               contrast=(frame[AUG] - frame[CLEAN]).mean(), k=3)
    raw = per.pivot(index="seed", columns="training", values="raw_f1")
    orc = per.pivot(index="seed", columns="training", values="oracle_threshold_f1")
    same_sign = int((np.sign(raw[AUG] - raw[CLEAN]) == np.sign(orc[AUG] - orc[CLEAN])).sum())
    record(block, "partitions on which the condition ahead at 0.5 stays ahead at the oracle threshold",
           source="revision_distilbert_stream_probabilities.csv", n_positive=same_sign, k=3)
    for seed in REFERENCE:
        record(block, f"F1 contrast at 0.5 -> oracle threshold, seed {seed}",
               raw.at[seed, AUG] - raw.at[seed, CLEAN], orc.at[seed, AUG] - orc.at[seed, CLEAN],
               "revision_distilbert_stream_probabilities.csv")

    # agreement between the diagnostic re-run (rev_e, 160, end of epoch 1) and rev_g
    curve = pd.read_csv(RESULTS / "revision_distilbert_learning_curve.csv")
    end = curve[(curve["max_length"] == 160) & np.isclose(curve["epoch_progress"], 1.0)]
    stored = pd.read_csv(RESULTS / "revision_distilbert_stream_metrics.csv")
    worst = 0.0
    for condition, columns in ((LINKED, ("f1", "precision", "recall", "roc_auc", "pr_auc", "ece")),
                               (L2, ("f1", "precision", "recall", "roc_auc", "pr_auc", "ece", "fpr_at_0_5")),
                               (SPAM, ("fpr_at_0_5",))):
        a = end[end["eval_condition"] == condition].set_index(["seed", "training"])
        b = stored[stored["condition"] == condition].drop_duplicates(["seed", "training"]).set_index(["seed", "training"])
        for column in columns:
            diff = float((a[column] - b[column].reindex(a.index)).abs().max())
            worst = max(worst, diff)
            record("DistilBERT, rev_e (160, end of epoch 1) vs rev_g",
                   f"max |difference| {condition} {column}", source="learning curve vs stream metrics",
                   contrast=diff)
    record("DistilBERT, rev_e (160, end of epoch 1) vs rev_g", "max |difference| over all of the above",
           source="learning curve vs stream metrics", contrast=worst)


def reference_diagnostics(backbone: pd.DataFrame) -> None:
    block = "DistilBERT, reference partitions, diagnostic re-run (rev_e)"
    probability = pd.read_csv(RESULTS / "revision_distilbert_probabilities.csv")
    l2p = probability[(probability["epoch"] == 1) & (probability["condition"] == L2)]
    for (length, training), pooled in l2p.groupby(["max_length", "training"]):
        p, y = pooled["probability"].to_numpy(), pooled["label"].to_numpy()
        bottom, top = p < 0.1, p >= 0.9
        tag = f"{block}, {length} tokens, pooled over partitions, {training}"
        record(tag, "share < 0.05", source="revision_distilbert_probabilities.csv",
               contrast=float((p < 0.05).mean()))
        record(tag, "share > 0.95", source="revision_distilbert_probabilities.csv",
               contrast=float((p > 0.95).mean()))
        record(tag, "bottom decile [0, 0.1): mean probability / observed", p[bottom].mean(),
               y[bottom].mean(), "revision_distilbert_probabilities.csv")
        record(tag, "top decile [0.9, 1]: mean probability / observed", p[top].mean(),
               y[top].mean(), "revision_distilbert_probabilities.csv")
    for backbone_name in ("DistilBERT", "RoBERTa"):
        for length in (160, 320):
            part = backbone[(backbone["backbone"] == backbone_name)
                            & (backbone["max_length"] == length)
                            & backbone["seed"].isin(["29", "42", "71"])
                            & backbone["training"].isin([CLEAN, AUG])]
            if part.empty:
                continue
            sd = part.pivot(index="seed", columns="training", values="l2_f1_sd_epoch1")
            record(f"{backbone_name}, reference partitions, {length} tokens",
                   "L2 F1 SD within epoch 1, seed 42", sd.at["42", CLEAN], sd.at["42", AUG],
                   "backbone summary")
            record(f"{backbone_name}, reference partitions, {length} tokens",
                   "L2 F1 SD within epoch 1, mean of seeds 29 and 71",
                   sd.loc[["29", "71"], CLEAN].mean(), sd.loc[["29", "71"], AUG].mean(),
                   "backbone summary")
            temps = part.pivot(index="seed", columns="training", values="l2_temperature")
            record(f"{backbone_name}, reference partitions, {length} tokens",
                   "L2 temperature (fitted on L2)", temps[CLEAN].mean(), temps[AUG].mean(),
                   "backbone summary", n_positive=int((temps[AUG] > temps[CLEAN]).sum()),
                   k=len(temps))
    strata = pd.read_csv(RESULTS / "revision_r3_distilbert_truncation_strata.csv")
    narrow = strata[strata["max_length"] == 160]
    for quartile in (1, 4):
        q = narrow[narrow["length_quartile"] == quartile]
        piv = q.pivot(index="seed", columns="training", values="f1")
        record(block, f"F1 contrast within length quartile {quartile} (160 tokens)",
               source="revision_r3_distilbert_truncation_strata.csv",
               contrast=(piv[AUG] - piv[CLEAN]).mean(),
               note_median_tokens=float(q["median_tokens"].mean()))
    for name, model in (("revision_distilbert_truncation_audit.csv", "DistilBERT"),
                        ("revision_r4_backbone_truncation_audit.csv", "RoBERTa")):
        audit = pd.read_csv(RESULTS / name).set_index("corpus")
        for corpus in ("primary_linked", "external_openai"):
            for column in ("fraction_truncated_at_160", "fraction_truncated_at_320",
                           "mean_retained_fraction_at_160", "mean_retained_fraction_at_320"):
                record(f"truncation audit, {model} tokenizer", f"{corpus} {column}",
                       source=name, contrast=float(audit.at[corpus, column]))


def passthrough(metrics: pd.DataFrame) -> None:
    heterogeneity = json.loads((RESULTS / "revision_r4_heterogeneity.json").read_text())
    bert = next(f for f in heterogeneity["families"] if f["family"] == "DistilBERT")
    record("DistilBERT, 20 refitted partitions", "linked F1 random-effects I^2 (read)",
           source="revision_r4_heterogeneity.json", contrast=bert["re_I2"], k=bert["splits"])
    effects = pd.read_csv(RESULTS / "revision_r4_partition_effects.csv")
    l2 = effects[effects["stream"] == L2]
    word = l2[l2["family"] == "Word TF-IDF"]
    bert_l2 = l2[l2["family"] == "DistilBERT"]
    for metric in ("f1", "ece", "roc_auc"):
        w = word[word["metric"] == metric]
        record("Word TF-IDF, 100 partitions", f"L2 {metric} mean", w["clean"].mean(),
               w["augmented"].mean(), "revision_r4_partition_effects.csv", k=len(w),
               sd_augmented=float(w["augmented"].std(ddof=1)))
        b = bert_l2[bert_l2["metric"] == metric].set_index("seed")
        record("DistilBERT, 20 partitions (partition-effects file)", f"L2 {metric} mean",
               b["clean"].mean(), b["augmented"].mean(), "revision_r4_partition_effects.csv",
               k=len(b), sd_augmented=float(b["augmented"].std(ddof=1)))
        shared = w.set_index("seed").loc[b.index, "augmented"]
        better = shared < b["augmented"] if metric == "ece" else shared > b["augmented"]
        record("Word TF-IDF vs DistilBERT, 20 shared partitions (augmented models)",
               f"partitions on which the word model is better on L2 {metric}",
               source="revision_r4_partition_effects.csv", n_positive=int(better.sum()), k=len(b))
    # cross-check the tests above against rev_r's summary
    summary = pd.read_csv(RESULTS / "revision_r4_partition_effect_summary.csv")
    summary = summary[summary["family"] == "DistilBERT"].set_index(["stream", "metric"])
    mine = pd.DataFrame(records)
    mine = mine[mine["block"] == "DistilBERT, 20 refitted partitions"]
    worst = 0.0
    for stream, label in ((LINKED, "linked"), (L2, "L2")):
        for metric in ("f1", "roc_auc", "pr_auc", "ece"):
            row = mine[mine["quantity"] == f"{label} {metric}"].iloc[0]
            ref = summary.loc[(stream, metric)]
            worst = max(worst, abs(row["contrast"] - ref["mean"]),
                        abs(row["corrected_p"] - ref["corrected_p"]),
                        abs(row["n_positive"] / row["k"] - ref["share_positive"]))
    record("cross-check against revision_r4_partition_effect_summary.csv",
           "max |difference| (mean, corrected p, share positive)",
           source="rev_r", contrast=worst)
    if worst > 1e-9:
        print(f"WARNING: tests differ from rev_r's summary by up to {worst:.2e}", file=sys.stderr)


def roberta_counts(backbone: pd.DataFrame) -> None:
    for length in (160, 320):
        part = backbone[(backbone["backbone"] == "RoBERTa") & (backbone["max_length"] == length)
                        & backbone["seed"].isin(["29", "42", "71"])
                        & backbone["training"].isin([CLEAN, AUG])]
        if part["seed"].nunique() < 3:
            continue
        block = f"RoBERTa, reference partitions, {length} tokens"
        for key in ("l2_roc_auc", "l2_f1", "l2_ece", "spam_fpr_at_0_5", "oracle_threshold_f1",
                    "validation_temperature", "l2_ece_validation_scaled", "l2_ece_l2_scaled",
                    "share_below_0_05", "share_above_0_95"):
            piv = part.pivot(index="seed", columns="training", values=key)
            d = piv[AUG] - piv[CLEAN]
            record(block, key, piv[CLEAN].mean(), piv[AUG].mean(), "backbone summary",
                   contrast=d.mean(), n_positive=int((d > 0).sum()), k=3,
                   per_partition=" ".join(f"{s}:{v:+.3f}" for s, v in d.items()))


def main() -> None:
    backbone = backbone_rows()
    backbone.to_csv(RESULTS / "revision_r4_backbone_summary.csv", index=False, float_format="%.6g")
    metrics = refit_metrics()
    twenty_partitions(metrics, refit_probabilities())
    reference_stored_scores(metrics)
    reference_diagnostics(backbone)
    roberta_counts(backbone)
    passthrough(metrics)
    summary = pd.DataFrame(records)
    columns = ["block", "quantity", "clean", "augmented", "contrast", "n_positive", "k",
               "corrected_p", "corrected_low", "corrected_high", "source"]
    summary = summary[columns + [c for c in summary.columns if c not in columns]]
    summary.to_csv(RESULTS / "revision_r4_transformer_summary.csv", index=False, float_format="%.6g")
    assert "torch" not in sys.modules, "torch must not be imported"
    print(f"wrote {len(backbone)} rows to revision_r4_backbone_summary.csv and "
          f"{len(summary)} rows to revision_r4_transformer_summary.csv")


if __name__ == "__main__":
    main()
