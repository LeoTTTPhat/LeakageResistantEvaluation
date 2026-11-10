#!/usr/bin/env python3
"""Paired inference across many refitted grouped partitions (round-3 comment 2).

Consumes the twenty lightweight partitions of ``rev_k_multisplit.py`` and the
eight DistilBERT partitions (three published, five added by
``rev_k_distilbert_multisplit.py``) and reports, for each representation family:

* per-split grouped bootstrap intervals (uncertainty *within* one partition);
* a two-stage hierarchical bootstrap that resamples partitions and then source
  groups within partitions;
* a paired split-level Student-t interval, which is the fixed-effect estimate of
  the one-way random-effects model with the partition as the random unit;
* an exact paired permutation (sign-flip) test over partitions, which assumes
  only symmetry of the split-level differences;
* a Student-t *prediction* interval for the effect on one further, as yet
  undrawn partition -- the quantity a practitioner refitting on new data faces;
* a variance decomposition with the between-partition share of total variance;
* a convergence curve giving the interval half-width obtainable with K
  partitions, K = 3, 5, 10, 20.

Outputs (results/):
  revision_r3_inference.csv
  revision_r3_variance_decomposition.json
  revision_r3_split_convergence.csv
"""

from __future__ import annotations

import json
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from rev_common import RESULTS

N_BOOTSTRAP = 5000
THRESHOLD = 0.5
RNG_SEED = 20260903


def group_counts(frame: pd.DataFrame, probability_column: str):
    codes, _ = pd.factorize(frame["group_id"].to_numpy())
    n_groups = codes.max() + 1
    y = frame["label"].to_numpy().astype(bool)
    predicted = frame[probability_column].to_numpy() >= THRESHOLD
    tp = np.bincount(codes[y & predicted], minlength=n_groups).astype(float)
    fp = np.bincount(codes[~y & predicted], minlength=n_groups).astype(float)
    fn = np.bincount(codes[y & ~predicted], minlength=n_groups).astype(float)
    return tp, fp, fn


def f1_from_counts(tp, fp, fn) -> float:
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def f1_total(counts) -> float:
    tp, fp, fn = counts
    return f1_from_counts(tp.sum(), fp.sum(), fn.sum())


def delta_replicates(counts_a, counts_b, draws: np.ndarray) -> np.ndarray:
    out = np.empty(len(draws), dtype=float)
    for index, sample in enumerate(draws):
        a = [array[sample].sum() for array in counts_a]
        b = [array[sample].sum() for array in counts_b]
        out[index] = f1_from_counts(*b) - f1_from_counts(*a)
    return out


def two_sided_p(replicates: np.ndarray) -> float:
    return float(
        min(1.0, 2.0 * min((replicates <= 0).mean(), (replicates >= 0).mean()))
    )


def sign_flip_p(values: np.ndarray) -> float:
    """Exact two-sided paired permutation test over split-level differences."""
    n = len(values)
    observed = abs(values.mean())
    if n <= 20:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        means = np.abs((signs * values).mean(axis=1))
        return float((means >= observed - 1e-15).mean())
    rng = np.random.default_rng(RNG_SEED)
    signs = rng.choice([-1.0, 1.0], size=(200_000, n))
    means = np.abs((signs * values).mean(axis=1))
    return float(((means >= observed - 1e-15).sum() + 1) / (len(means) + 1))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def hierarchical_replicates(per_split_counts, seeds, rng, n=N_BOOTSTRAP):
    out = np.empty(n, dtype=float)
    seed_array = np.array(seeds)
    for index in range(n):
        chosen = rng.choice(seed_array, size=len(seed_array), replace=True)
        values = []
        for seed in chosen:
            counts_a, counts_b = per_split_counts[seed]
            n_groups = len(counts_a[0])
            sample = rng.integers(0, n_groups, size=n_groups)
            a = [array[sample].sum() for array in counts_a]
            b = [array[sample].sum() for array in counts_b]
            values.append(f1_from_counts(*b) - f1_from_counts(*a))
        out[index] = float(np.mean(values))
    return out


def analyse(label: str, per_split_counts: dict, rng):
    seeds = sorted(per_split_counts)
    rows: list[dict] = []
    per_split_delta: dict[int, float] = {}
    within_variance: list[float] = []

    for seed in seeds:
        counts_a, counts_b = per_split_counts[seed]
        n_groups = len(counts_a[0])
        draws = rng.integers(0, n_groups, size=(N_BOOTSTRAP, n_groups))
        replicates = delta_replicates(counts_a, counts_b, draws)
        observed = f1_total(counts_b) - f1_total(counts_a)
        per_split_delta[seed] = observed
        within_variance.append(float(replicates.var(ddof=1)))
        low, high = np.quantile(replicates, [0.025, 0.975])
        rows.append(
            {
                "family": label,
                "scope": "single split (grouped bootstrap)",
                "splits": 1,
                "seed": seed,
                "f1_clean": f1_total(counts_a),
                "f1_augmented": f1_total(counts_b),
                "delta": observed,
                "ci_low": float(low),
                "ci_high": float(high),
                "p_two_sided": two_sided_p(replicates),
            }
        )

    values = np.array([per_split_delta[s] for s in seeds])
    n = len(values)
    mean_clean = float(np.mean([f1_total(per_split_counts[s][0]) for s in seeds]))
    mean_augmented = float(np.mean([f1_total(per_split_counts[s][1]) for s in seeds]))

    hierarchical = hierarchical_replicates(per_split_counts, seeds, rng)
    low, high = np.quantile(hierarchical, [0.025, 0.975])
    rows.append(
        {
            "family": label,
            "scope": "all splits (hierarchical bootstrap)",
            "splits": n,
            "seed": -1,
            "f1_clean": mean_clean,
            "f1_augmented": mean_augmented,
            "delta": float(values.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "p_two_sided": two_sided_p(hierarchical),
        }
    )

    sd = float(values.std(ddof=1))
    standard_error = sd / np.sqrt(n)
    t_critical = float(stats.t.ppf(0.975, n - 1))
    t_statistic = values.mean() / standard_error if standard_error > 0 else np.inf
    rows.append(
        {
            "family": label,
            "scope": "all splits (paired split-level t)",
            "splits": n,
            "seed": -1,
            "f1_clean": mean_clean,
            "f1_augmented": mean_augmented,
            "delta": float(values.mean()),
            "ci_low": float(values.mean() - t_critical * standard_error),
            "ci_high": float(values.mean() + t_critical * standard_error),
            "p_two_sided": float(
                2 * (1 - stats.t.cdf(abs(t_statistic), n - 1))
                if standard_error > 0
                else 0.0
            ),
        }
    )

    rows.append(
        {
            "family": label,
            "scope": "all splits (sign-flip permutation)",
            "splits": n,
            "seed": -1,
            "f1_clean": mean_clean,
            "f1_augmented": mean_augmented,
            "delta": float(values.mean()),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided": sign_flip_p(values),
        }
    )

    prediction_half_width = t_critical * sd * np.sqrt(1.0 + 1.0 / n)
    rows.append(
        {
            "family": label,
            "scope": "one further split (t prediction interval)",
            "splits": n,
            "seed": -1,
            "f1_clean": mean_clean,
            "f1_augmented": mean_augmented,
            "delta": float(values.mean()),
            "ci_low": float(values.mean() - prediction_half_width),
            "ci_high": float(values.mean() + prediction_half_width),
            "p_two_sided": np.nan,
        }
    )

    between_variance = float(values.var(ddof=1))
    within_mean = float(np.mean(within_variance))
    decomposition = {
        "family": label,
        "splits": n,
        "delta_mean": float(values.mean()),
        "between_split_variance": between_variance,
        "mean_within_split_bootstrap_variance": within_mean,
        "between_split_sd": float(np.sqrt(between_variance)),
        "mean_within_split_bootstrap_sd": float(np.sqrt(within_mean)),
        "between_split_share_of_total_variance": float(
            between_variance / (between_variance + within_mean)
        ),
        "hierarchical_sd": float(hierarchical.std(ddof=1)),
        "min_split_delta": float(values.min()),
        "max_split_delta": float(values.max()),
        "splits_with_negative_delta": int((values <= 0).sum()),
        "per_split_delta": {int(s): float(per_split_delta[s]) for s in seeds},
    }
    return rows, decomposition, two_sided_p(hierarchical), values, per_split_counts


def convergence(label, values, per_split_counts, rng):
    """Interval half-width as a function of the number of partitions used."""
    seeds = sorted(per_split_counts)
    rows = []
    for k in sorted({k for k in (3, 5, 8, 10, 20, len(seeds)) if 3 <= k <= len(seeds)}):
        subset = seeds[:k]
        subset_values = np.array([values[seeds.index(s)] for s in subset])
        sd = subset_values.std(ddof=1)
        t_critical = stats.t.ppf(0.975, k - 1)
        hierarchical = hierarchical_replicates(
            {s: per_split_counts[s] for s in subset}, subset, rng, n=2000
        )
        low, high = np.quantile(hierarchical, [0.025, 0.975])
        rows.append(
            {
                "family": label,
                "splits_used": k,
                "delta": float(subset_values.mean()),
                "t_half_width": float(t_critical * sd / np.sqrt(k)),
                "hierarchical_half_width": float((high - low) / 2.0),
                "hierarchical_ci_low": float(low),
                "hierarchical_ci_high": float(high),
            }
        )
    return rows


def lightweight_counts(column_clean, column_augmented):
    table = pd.read_csv(RESULTS / "revision_r3_multisplit_probabilities.csv")
    linked = table[table["condition"] == "linked_rewrites"]
    per_split = {}
    for seed in sorted(linked["seed"].unique()):
        frame = linked[linked["seed"] == seed].reset_index(drop=True)
        per_split[int(seed)] = (
            group_counts(frame, column_clean),
            group_counts(frame, column_augmented),
        )
    return per_split


def distilbert_counts():
    """Three published splits (group ids recovered from the lightweight table)
    plus the five refitted in this revision."""
    per_split: dict[int, tuple] = {}

    published = RESULTS / "revision_distilbert_probabilities.csv"
    reference = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    reference = reference[reference["condition"] == "linked_rewrites"]
    if published.exists():
        frame = pd.read_csv(published)
        frame = frame[
            (frame["max_length"] == 160)
            & (frame["epoch"] == 1)
            & (frame["condition"] == "linked_rewrites")
        ]
        for seed in (29, 42, 71):
            block = frame[frame["seed"] == seed]
            if block.empty:
                continue
            groups = reference[reference["seed"] == seed]["group_id"].to_numpy()
            clean = block[block["training"] == "clean-only"].reset_index(drop=True)
            augmented = block[block["training"] == "linked-augmented"].reset_index(
                drop=True
            )
            if len(clean) != len(groups):
                raise ValueError(f"DistilBERT seed {seed} rows do not align")
            per_split[seed] = (
                group_counts(clean.assign(group_id=groups), "probability"),
                group_counts(augmented.assign(group_id=groups), "probability"),
            )

    added = RESULTS / "revision_r3_distilbert_multisplit_probabilities.csv"
    if added.exists():
        frame = pd.read_csv(added)
        frame = frame[frame["condition"] == "linked_rewrites"]
        for seed in sorted(frame["seed"].unique()):
            block = frame[frame["seed"] == seed]
            clean = block[block["training"] == "clean-only"].reset_index(drop=True)
            augmented = block[block["training"] == "linked-augmented"].reset_index(
                drop=True
            )
            per_split[int(seed)] = (
                group_counts(clean, "probability"),
                group_counts(augmented, "probability"),
            )
    return per_split


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict] = []
    decompositions: list[dict] = []
    hierarchical_p: dict[str, float] = {}
    convergence_rows: list[dict] = []

    families = [
        (
            "Word TF-IDF",
            "probability_word_lr_clean_only",
            "probability_word_lr_linked_augmented",
        ),
        (
            "MiniLM",
            "probability_minilm_lr_clean_only",
            "probability_minilm_lr_linked_augmented",
        ),
    ]
    for label, clean_column, augmented_column in families:
        per_split = lightweight_counts(clean_column, augmented_column)
        new_rows, decomposition, p, values, counts = analyse(label, per_split, rng)
        rows.extend(new_rows)
        decompositions.append(decomposition)
        hierarchical_p[label] = p
        convergence_rows.extend(convergence(label, values, counts, rng))

    per_split = distilbert_counts()
    if per_split:
        new_rows, decomposition, p, values, counts = analyse(
            "DistilBERT", per_split, rng
        )
        rows.extend(new_rows)
        decompositions.append(decomposition)
        hierarchical_p["DistilBERT"] = p
        convergence_rows.extend(convergence("DistilBERT", values, counts, rng))

    adjusted = holm(hierarchical_p)
    output = pd.DataFrame(rows)
    output["holm_adjusted_p"] = [
        adjusted[row.family]
        if row.scope == "all splits (hierarchical bootstrap)"
        else np.nan
        for row in output.itertuples()
    ]
    output.to_csv(RESULTS / "revision_r3_inference.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(
        RESULTS / "revision_r3_split_convergence.csv", index=False
    )
    (RESULTS / "revision_r3_variance_decomposition.json").write_text(
        json.dumps(
            {
                "bootstrap_replicates": N_BOOTSTRAP,
                "families": decompositions,
                "hierarchical_p_two_sided": hierarchical_p,
                "holm_adjusted": adjusted,
            },
            indent=2,
        )
    )
    print(
        output[output["seed"] == -1][
            ["family", "scope", "splits", "delta", "ci_low", "ci_high", "p_two_sided"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
