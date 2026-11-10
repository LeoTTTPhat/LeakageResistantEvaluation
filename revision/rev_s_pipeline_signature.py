#!/usr/bin/env python3
"""Does the L2 augmentation gain survive removal of pipeline-specific signal?

Round-4 comments R1.1 and R2.1. The origin probe of round 3 separates the two
LLM corpora at ROC-AUC 1.000, so each corpus carries a strong pipeline
signature. Reviewer 1 points out that the L2 gain is then ambiguous: it may
reflect better coverage of phishing language, or better coverage of the second
pipeline's stylistic regularities. The reviewer suggests neutralizing corpus-
specific vocabulary, or domain-adversarial training. This script does both, and
adds linear concept erasure for the dense representation:

A. Vocabulary neutralization (word TF-IDF).  Every n-gram of the detector's
   vocabulary is scored for corpus specificity by a two-proportion z statistic
   on document frequency, PhishFuzzer training documents being reweighted to the
   L2 class prevalence so that phishing vocabulary is not mistaken for L2
   vocabulary. The k most corpus-specific n-grams are removed (k = 0 ... 40,000,
   up to the whole vocabulary but its 10,000 least specific terms), both
   detectors are refitted on the reduced vocabulary, and the residual origin
   separability is re-measured with a fresh origin probe.

B. Linear concept erasure (MiniLM).  LEACE (Belrose et al., 2023) removes the
   least-squares direction that predicts corpus origin; INLP (Ravfogel et al.,
   2020) iteratively projects out up to 64 origin-predictive directions. The
   phishing detectors are refitted on the erased embeddings.

C. Domain-adversarial training (DANN; Ganin et al., 2016) on MiniLM
   embeddings, with a gradient-reversal domain head that sees unlabeled L2
   messages, against the same network trained without the adversary.

Cross-fitting.  Everything that looks at L2 text -- the specificity scores, the
erasers, the adversary -- is fitted on one half of L2 and evaluated on the
other, and the halves are made of whole cosine-0.90 latent components, so no
near-duplicate of an evaluated message is ever used to decide what to remove.
Each half is used once for fitting and once for evaluation, so every L2 message
is scored exactly once per partition. No fitted component sees an L2 label: the
specificity scores, erasers and adversary use L2 text only, plus the corpus's
aggregate class prevalence for the reweighting in A and B. L2 labels enter in one
place only -- stratifying the assignment of whole components to the two halves,
so that both halves keep the corpus's class mix.

Partitions.  The twenty source-grouped partitions of rev_k_multisplit.py.

Outputs (results/):
  revision_r4_pipeline_vocab.csv       partition x k  (A)
  revision_r4_pipeline_erasure.csv     partition x method x directions (B)
  revision_r4_pipeline_dann.csv        partition x condition x adversary (C)
  revision_r4_pipeline_terms.csv       most corpus-specific n-grams, partition 1
  revision_r4_pipeline_summary.csv     means across partitions
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from rev_common import RESULTS, base, load_corpora, load_embeddings, split_frames
from rev_k_multisplit import SPLIT_SEEDS
from rev_m_provenance import components

VOCAB_K = [0, 250, 1000, 2500, 5000, 10000, 20000, 30000, 40000]
INLP_DIRECTIONS = [1, 2, 4, 8]
DANN_EPOCHS = 30
_STATE: dict[str, object] = {}


# --------------------------------------------------------------------------
# shared state
# --------------------------------------------------------------------------
def _init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    torch.set_num_threads(1)
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)
    labels = components(external_embeddings @ external_embeddings.T, 0.90)
    _STATE.update(
        binary=binary,
        external=external,
        spam=spam,
        primary_embeddings=primary_embeddings,
        external_embeddings=external_embeddings,
        external_components=labels,
    )


def halves(component_labels: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Assign whole components to half 0 or 1, stratified by component majority."""
    frame = pd.DataFrame({"c": component_labels, "y": y})
    majority = frame.groupby("c")["y"].mean().round().astype(int)
    rng = np.random.default_rng(seed + 17)
    assignment = {}
    sizes = frame.groupby("c").size()
    for value in (0, 1):
        ids = rng.permutation(majority.index[majority == value].to_numpy())
        # greedy balancing by row count
        totals = [0, 0]
        for c in ids:
            side = int(totals[1] < totals[0])
            assignment[c] = side
            totals[side] += int(sizes[c])
    return np.array([assignment[c] for c in component_labels])


def prevalence_weights(y_source: np.ndarray, target_prevalence: float) -> np.ndarray:
    """Reweight source rows so that their class mix equals the target prevalence."""
    p = y_source.mean()
    return np.where(y_source == 1, target_prevalence / p, (1 - target_prevalence) / (1 - p))


def scores(y, probability):
    return {
        "f1": float(f1_score(y, probability >= 0.5, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "recall": float((probability[y == 1] >= 0.5).mean()),
        "precision": float(
            (y[probability >= 0.5] == 1).mean() if (probability >= 0.5).any() else 0.0
        ),
    }


# --------------------------------------------------------------------------
# A. vocabulary neutralization
# --------------------------------------------------------------------------
def specificity(source_texts, source_weights, target_texts):
    """|z| of the weighted two-proportion test on document frequency, per n-gram."""
    counter = CountVectorizer(
        ngram_range=(1, 2), min_df=2, binary=True, strip_accents="unicode"
    )
    matrix = counter.fit_transform(list(source_texts) + list(target_texts))
    n_source = len(source_texts)
    source_matrix = matrix[:n_source]
    target_matrix = matrix[n_source:]
    w = source_weights / source_weights.mean()
    p_source = np.asarray(source_matrix.multiply(w[:, None]).sum(0)).ravel() / n_source
    p_target = np.asarray(target_matrix.sum(0)).ravel() / target_matrix.shape[0]
    n_eff = n_source  # weights are normalized to mean one
    pooled = (p_source * n_eff + p_target * target_matrix.shape[0]) / (
        n_eff + target_matrix.shape[0]
    )
    se = np.sqrt(np.maximum(pooled * (1 - pooled), 1e-12) * (1 / n_eff + 1 / target_matrix.shape[0]))
    z = (p_target - p_source) / se
    return dict(zip(counter.get_feature_names_out(), z))


def fit_word(texts, labels, vocabulary=None):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2 if vocabulary is None else 1,
        max_features=50000 if vocabulary is None else None,
        sublinear_tf=True,
        strip_accents="unicode",
        vocabulary=vocabulary,
    )
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(
        C=4.0, class_weight="balanced", max_iter=3000, solver="liblinear"
    ).fit(matrix, labels)
    return vectorizer, classifier


def origin_auc_text(vocabulary, fit_source, fit_target, eval_source, eval_target, weights):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
        vocabulary=vocabulary,
    )
    x_fit = vectorizer.fit_transform(list(fit_source) + list(fit_target))
    origin = np.r_[np.zeros(len(fit_source)), np.ones(len(fit_target))]
    sample_weight = np.r_[weights, np.ones(len(fit_target))]
    sample_weight = np.r_[
        sample_weight[: len(fit_source)] * len(fit_target) / sample_weight[: len(fit_source)].sum(),
        sample_weight[len(fit_source):],
    ]
    probe = LogisticRegression(C=4.0, max_iter=3000, solver="liblinear").fit(
        x_fit, origin, sample_weight=sample_weight
    )
    x_eval = vectorizer.transform(list(eval_source) + list(eval_target))
    truth = np.r_[np.zeros(len(eval_source)), np.ones(len(eval_target))]
    return float(roc_auc_score(truth, probe.predict_proba(x_eval)[:, 1]))


def vocabulary_neutralization(seed, frames, external, half, prevalence):
    rows, terms = [], []
    train = {
        "clean": frames["train_clean"],
        "augmented": frames["train_all"],
    }
    linked = frames["linked_rewrites"]
    source_fit_texts = frames["train_all"]["text"].tolist()
    source_weights = prevalence_weights(frames["train_all"]["label"].to_numpy(), prevalence)
    source_eval_texts = pd.concat([frames["clean"], linked])["text"].tolist()
    base_vocab = {
        name: fit_word(frame["text"].tolist(), frame["label"].to_numpy())[0].vocabulary_
        for name, frame in train.items()
    }
    probability = {(name, k): np.zeros(len(external)) for name in train for k in VOCAB_K}
    linked_f1 = {(name, k): [] for name in train for k in VOCAB_K}
    origin = {k: [] for k in VOCAB_K}
    for fit_half in (0, 1):
        fit_rows = np.flatnonzero(half == fit_half)
        eval_rows = np.flatnonzero(half != fit_half)
        target_fit = external["text"].to_numpy()[fit_rows]
        target_eval = external["text"].to_numpy()[eval_rows]
        z = specificity(source_fit_texts, source_weights, target_fit)
        if fit_half == 0 and seed == SPLIT_SEEDS[0]:
            ranked = sorted(z.items(), key=lambda item: -abs(item[1]))[:60]
            terms = [{"term": t, "z": float(v), "direction": "L2" if v > 0 else "PhishFuzzer"}
                     for t, v in ranked]
        union = sorted(set(base_vocab["clean"]) | set(base_vocab["augmented"]),
                       key=lambda t: -abs(z.get(t, 0.0)))
        for k in VOCAB_K:
            removed = set(union[:k])
            for name, frame in train.items():
                vocabulary = sorted(t for t in base_vocab[name] if t not in removed)
                if len(vocabulary) < 50:
                    raise ValueError(f"vocabulary exhausted at k={k} for {name}")
                vectorizer, classifier = fit_word(
                    frame["text"].tolist(), frame["label"].to_numpy(), vocabulary
                )
                probability[(name, k)][eval_rows] = classifier.predict_proba(
                    vectorizer.transform(target_eval)
                )[:, 1]
                if fit_half == 0:
                    linked_f1[(name, k)].append(
                        f1_score(
                            linked["label"].to_numpy(),
                            classifier.predict_proba(vectorizer.transform(linked["text"].tolist()))[:, 1]
                            >= 0.5,
                        )
                    )
            kept = sorted(t for t in union if t not in removed)
            origin[k].append(
                origin_auc_text(kept, source_fit_texts, target_fit, source_eval_texts,
                                target_eval, source_weights)
            )
    y = external["label"].to_numpy().astype(int)
    vocab_size = len(set(base_vocab["clean"]) | set(base_vocab["augmented"]))
    for k in VOCAB_K:
        clean = scores(y, probability[("clean", k)])
        augmented = scores(y, probability[("augmented", k)])
        rows.append(
            {
                "seed": seed,
                "removed_ngrams": k,
                "union_vocabulary": vocab_size,
                "origin_auc_after_removal": float(np.mean(origin[k])),
                **{f"clean_{m}": v for m, v in clean.items()},
                **{f"augmented_{m}": v for m, v in augmented.items()},
                **{f"delta_{m}": augmented[m] - clean[m] for m in clean},
                "linked_f1_clean": float(np.mean(linked_f1[("clean", k)])),
                "linked_f1_augmented": float(np.mean(linked_f1[("augmented", k)])),
            }
        )
    return rows, terms


# --------------------------------------------------------------------------
# B. linear concept erasure
# --------------------------------------------------------------------------
def leace(x, z, weights):
    """LEACE eraser for a binary concept, with sample weights (Belrose et al. 2023)."""
    w = weights / weights.sum()
    mu = (w[:, None] * x).sum(0)
    xc = x - mu
    zc = z - (w * z).sum()
    sigma = (w[:, None] * xc).T @ xc
    cross = (w[:, None] * xc).T @ zc  # d-vector
    values, vectors = np.linalg.eigh(sigma)
    keep = values > values.max() * 1e-8
    whiten = vectors[:, keep] @ np.diag(values[keep] ** -0.5) @ vectors[:, keep].T
    unwhiten = vectors[:, keep] @ np.diag(values[keep] ** 0.5) @ vectors[:, keep].T
    u = whiten @ cross
    u = u / np.linalg.norm(u)
    projection = unwhiten @ np.outer(u, u) @ whiten

    def erase(data):
        return data - (data - mu) @ projection.T

    return erase


def inlp(x, z, weights, directions, max_directions: int = 64):
    """Iterative nullspace projection (Ravfogel et al., 2020), run to convergence.

    Returns the projection after each requested count, and under the key
    ``"converged"`` the projection at the point where a fresh origin probe can
    no longer find a direction (zero coefficients or chance training accuracy),
    together with the number of directions that took.
    """
    d = x.shape[1]
    projection = np.eye(d)
    out = {}
    removed = []
    for i in range(1, max_directions + 1):
        probe = LogisticRegression(C=1.0, max_iter=2000).fit(
            x @ projection, z, sample_weight=weights
        )
        accuracy = probe.score(x @ projection, z, sample_weight=weights)
        direction = probe.coef_[0] @ projection
        for previous in removed:
            direction = direction - previous * (direction @ previous)
        norm = np.linalg.norm(direction)
        if norm < 1e-6 or accuracy <= 0.505:
            break
        direction = direction / norm
        removed.append(direction)
        projection = projection - np.outer(direction, direction)
        if i in directions:
            out[i] = projection.copy()
    out["converged"] = (projection.copy(), len(removed))
    return out


def cv_origin_auc(features, origin, groups, folds: int = 5) -> float:
    """Held-out origin decodability on already-erased data.

    A linear probe is cross-validated on the erased evaluation set with folds
    grouped by PhishFuzzer source group and by L2 latent component, so that no
    near-duplicate of a scored message is in the probe's training fold.
    """
    from sklearn.model_selection import GroupKFold

    scores_ = np.zeros(len(origin))
    for train_index, test_index in GroupKFold(folds).split(features, origin, groups):
        probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(
            features[train_index], origin[train_index]
        )
        scores_[test_index] = probe.predict_proba(features[test_index])[:, 1]
    return float(roc_auc_score(origin, scores_))


def fit_embed(x, y):
    return LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000,
                              solver="liblinear").fit(x, y)


def concept_erasure(seed, frames, primary_embeddings, external_embeddings, y_ext, half, prevalence,
                    component_labels):
    rows = []
    train = {"clean": frames["train_clean"], "augmented": frames["train_all"]}
    x_source_fit = primary_embeddings[frames["train_all"].index.to_numpy()]
    y_source_fit = frames["train_all"]["label"].to_numpy()
    source_eval = pd.concat([frames["clean"], frames["linked_rewrites"]])
    source_eval_index = source_eval.index.to_numpy()
    source_eval_groups = ("pf-" + source_eval["group_id"].astype(str)).to_numpy()
    x_source_eval = primary_embeddings[source_eval_index]
    linked = frames["linked_rewrites"]
    methods = ({("none", 0)} | {("leace", 1)} | {("inlp", d) for d in INLP_DIRECTIONS}
               | {("inlp_converged", -1)})
    probability = {(m, d, name): np.zeros(len(y_ext)) for (m, d) in methods for name in train}
    origin = {(m, d): [] for (m, d) in methods}
    origin_cv = {(m, d): [] for (m, d) in methods}
    converged_at = []
    linked_f1 = {(m, d, name): [] for (m, d) in methods for name in train}
    for fit_half in (0, 1):
        fit_rows = np.flatnonzero(half == fit_half)
        eval_rows = np.flatnonzero(half != fit_half)
        x_t_fit = external_embeddings[fit_rows]
        x_t_eval = external_embeddings[eval_rows]
        x = np.vstack([x_source_fit, x_t_fit])
        z = np.r_[np.zeros(len(x_source_fit)), np.ones(len(x_t_fit))]
        ws = prevalence_weights(y_source_fit, prevalence)
        ws = ws * len(x_t_fit) / ws.sum()
        weights = np.r_[ws, np.ones(len(x_t_fit))]
        transforms = {("none", 0): lambda a: a, ("leace", 1): leace(x, z, weights)}
        for d, projection in inlp(x, z, weights, INLP_DIRECTIONS).items():
            if d == "converged":
                projection, count = projection
                converged_at.append(count)
                transforms[("inlp_converged", -1)] = (lambda p: (lambda a: a @ p))(projection)
            else:
                transforms[("inlp", d)] = (lambda p: (lambda a: a @ p))(projection)
        truth = np.r_[np.zeros(len(x_source_eval)), np.ones(len(x_t_eval))]
        for key, transform in transforms.items():
            probe = LogisticRegression(C=1.0, max_iter=2000).fit(
                transform(x), z, sample_weight=weights
            )
            erased_eval = transform(np.vstack([x_source_eval, x_t_eval]))
            origin[key].append(float(roc_auc_score(truth, probe.predict_proba(erased_eval)[:, 1])))
            eval_groups = np.r_[source_eval_groups,
                                np.array([f"l2-{c}" for c in component_labels[eval_rows]])]
            origin_cv[key].append(cv_origin_auc(erased_eval, truth, eval_groups))
            for name, frame in train.items():
                model = fit_embed(transform(primary_embeddings[frame.index.to_numpy()]),
                                  frame["label"].to_numpy())
                probability[(*key, name)][eval_rows] = model.predict_proba(transform(x_t_eval))[:, 1]
                if fit_half == 0:
                    linked_f1[(*key, name)].append(f1_score(
                        linked["label"].to_numpy(),
                        model.predict_proba(transform(primary_embeddings[linked.index.to_numpy()]))[:, 1] >= 0.5,
                    ))
    for (method, d) in sorted(methods, key=lambda item: (item[0] != "none", item[0], item[1])):
        clean = scores(y_ext, probability[(method, d, "clean")])
        augmented = scores(y_ext, probability[(method, d, "augmented")])
        rows.append({
            "seed": seed, "method": method,
            "directions_removed": float(np.mean(converged_at)) if method == "inlp_converged" else d,
            "origin_auc_after_erasure": float(np.mean(origin[(method, d)])),
            "origin_auc_cv_on_erased_heldout": float(np.mean(origin_cv[(method, d)])),
            **{f"clean_{m}": v for m, v in clean.items()},
            **{f"augmented_{m}": v for m, v in augmented.items()},
            **{f"delta_{m}": augmented[m] - clean[m] for m in clean},
            "linked_f1_clean": float(np.mean(linked_f1[(method, d, "clean")])),
            "linked_f1_augmented": float(np.mean(linked_f1[(method, d, "augmented")])),
        })
    return rows


# --------------------------------------------------------------------------
# C. domain-adversarial training
# --------------------------------------------------------------------------
def dann(seed, frames, primary_embeddings, external_embeddings, y_ext, half):
    import torch
    from torch import nn

    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lam * grad, None

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.1),
                                          nn.Linear(256, 128), nn.ReLU())
            self.label = nn.Linear(128, 1)
            self.domain = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, lam=0.0):
            h = self.features(x)
            return self.label(h).squeeze(-1), self.domain(Reverse.apply(h, lam)).squeeze(-1), h

    rows = []
    linked = frames["linked_rewrites"]
    source_eval_index = pd.concat([frames["clean"], linked]).index.to_numpy()
    for name, frame in (("clean", frames["train_clean"]), ("augmented", frames["train_all"])):
        xs = torch.tensor(primary_embeddings[frame.index.to_numpy()], dtype=torch.float32)
        ys = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
        pos_weight = torch.tensor(float((ys == 0).sum() / (ys == 1).sum()))
        for adversarial in (False, True):
            probability = np.zeros(len(y_ext))
            domain_auc, head_auc, linked_scores = [], [], []
            for fit_half in (0, 1):
                torch.manual_seed(seed * 10 + fit_half)
                fit_rows = np.flatnonzero(half == fit_half)
                eval_rows = np.flatnonzero(half != fit_half)
                xt = torch.tensor(external_embeddings[fit_rows], dtype=torch.float32)
                net = Net()
                optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
                n = len(xs)
                batch = 128
                steps = DANN_EPOCHS * int(np.ceil(n / batch))
                generator = torch.Generator().manual_seed(seed)
                for step in range(steps):
                    progress = step / steps
                    lam = (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0) if adversarial else 0.0
                    i = torch.randint(0, n, (batch,), generator=generator)
                    j = torch.randint(0, len(xt), (batch,), generator=generator)
                    logits, domain_s, _ = net(xs[i], lam)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, ys[i], pos_weight=pos_weight)
                    if adversarial:
                        _, domain_t, _ = net(xt[j], lam)
                        loss = loss + nn.functional.binary_cross_entropy_with_logits(
                            torch.cat([domain_s, domain_t]),
                            torch.cat([torch.zeros(batch), torch.ones(batch)]))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                net.eval()
                with torch.no_grad():
                    xe = torch.tensor(external_embeddings[eval_rows], dtype=torch.float32)
                    probability[eval_rows] = torch.sigmoid(net(xe)[0]).numpy()
                    h_fit = net(torch.cat([xs, xt]))[2].numpy()
                    _, head_logits, h_eval = net(torch.cat([
                        torch.tensor(primary_embeddings[source_eval_index], dtype=torch.float32),
                        xe]))
                    h_eval = h_eval.numpy()
                    head_scores = torch.sigmoid(head_logits).numpy()
                    if fit_half == 0:
                        lp = torch.sigmoid(net(torch.tensor(
                            primary_embeddings[linked.index.to_numpy()], dtype=torch.float32))[0]).numpy()
                        linked_scores.append(f1_score(linked["label"].to_numpy(), lp >= 0.5))
                probe = LogisticRegression(C=1.0, max_iter=2000).fit(
                    h_fit, np.r_[np.zeros(len(xs)), np.ones(len(xt))])
                truth = np.r_[np.zeros(len(source_eval_index)), np.ones(len(eval_rows))]
                domain_auc.append(float(roc_auc_score(truth, probe.predict_proba(h_eval)[:, 1])))
                head_auc.append(float(roc_auc_score(truth, head_scores)) if adversarial else np.nan)
            rows.append({"seed": seed, "training": name, "adversarial": adversarial,
                         "domain_probe_auc": float(np.mean(domain_auc)),
                         "domain_head_auc": float(np.mean(head_auc)),
                         "linked_f1": float(np.mean(linked_scores)),
                         **scores(y_ext, probability)})
    return rows


def run_partition(seed):
    started = time.time()
    binary, external, spam = _STATE["binary"], _STATE["external"], _STATE["spam"]
    primary_embeddings = _STATE["primary_embeddings"]
    external_embeddings = _STATE["external_embeddings"]
    y_ext = external["label"].to_numpy().astype(int)
    prevalence = float(y_ext.mean())  # aggregate only; no row label is used
    half = halves(_STATE["external_components"], y_ext, seed)
    frames = split_frames(binary, external, spam, seed)
    vocab_rows, terms = vocabulary_neutralization(seed, frames, external, half, prevalence)
    erasure_rows = concept_erasure(seed, frames, primary_embeddings, external_embeddings,
                                   y_ext, half, prevalence, _STATE["external_components"])
    dann_rows = dann(seed, frames, primary_embeddings, external_embeddings, y_ext, half)
    return seed, vocab_rows, terms, erasure_rows, dann_rows, time.time() - started


def summarize(frame, keys):
    numeric = [c for c in frame.columns if c not in keys + ["seed"]]
    grouped = frame.groupby(keys)[numeric]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std().add_suffix("_sd")
    positive = frame.groupby(keys)["delta_f1"].apply(lambda s: (s > 0).mean()) if "delta_f1" in frame else None
    out = mean.join(sd)
    if positive is not None:
        out["share_partitions_delta_f1_positive"] = positive
    return out.reset_index()


def main() -> None:
    started = time.time()
    workers = int(os.environ.get("REV_S_WORKERS", "10"))
    vocab, erasure, adversarial, terms = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futures = [pool.submit(run_partition, seed) for seed in SPLIT_SEEDS]
        for done, future in enumerate(as_completed(futures), start=1):
            seed, v, t, e, d, seconds = future.result()
            vocab += v
            erasure += e
            adversarial += d
            terms += t
            print(f"{done}/{len(SPLIT_SEEDS)} seed {seed} ({seconds:.0f}s, "
                  f"elapsed {time.time() - started:.0f}s)", flush=True)
    vocab = pd.DataFrame(vocab).sort_values(["seed", "removed_ngrams"])
    erasure = pd.DataFrame(erasure).sort_values(["seed", "method", "directions_removed"])
    erasure.loc[erasure["method"] == "inlp_converged", "directions_label"] = "converged"
    adversarial = pd.DataFrame(adversarial).sort_values(["seed", "training", "adversarial"])
    vocab.to_csv(RESULTS / "revision_r4_pipeline_vocab.csv", index=False)
    erasure.to_csv(RESULTS / "revision_r4_pipeline_erasure.csv", index=False)
    adversarial.to_csv(RESULTS / "revision_r4_pipeline_dann.csv", index=False)
    pd.DataFrame(terms).to_csv(RESULTS / "revision_r4_pipeline_terms.csv", index=False)

    wide = adversarial.pivot_table(index=["seed", "adversarial"], columns="training",
                                   values=["f1", "roc_auc", "pr_auc", "domain_probe_auc",
                                           "domain_head_auc", "linked_f1"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for metric in ("f1", "roc_auc", "pr_auc", "linked_f1"):
        wide[f"delta_{metric}"] = wide[f"{metric}_augmented"] - wide[f"{metric}_clean"]
    summary = pd.concat(
        [
            summarize(vocab, ["removed_ngrams"]).assign(analysis="vocabulary"),
            summarize(erasure.assign(directions_removed=erasure["directions_removed"].where(
                erasure["method"] != "inlp_converged", -1)).drop(columns=["directions_label"]),
                ["method", "directions_removed"]).assign(analysis="erasure"),
            summarize(wide, ["adversarial"]).assign(analysis="dann"),
        ],
        ignore_index=True,
    )
    summary.to_csv(RESULTS / "revision_r4_pipeline_summary.csv", index=False)
    show = [c for c in summary.columns if c.endswith("_mean") and any(
        key in c for key in ("origin_auc", "domain_probe", "clean_f1", "augmented_f1",
                             "delta_f1", "delta_roc_auc", "linked_f1"))]
    print(summary[["analysis", "removed_ngrams", "method", "directions_removed", "adversarial"]
                  + show + ["share_partitions_delta_f1_positive"]].round(4).to_string(index=False))
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
