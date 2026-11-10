#!/usr/bin/env python3
"""Multi-split inference for the paired augmentation effect.

The submitted manuscript reported a grouped bootstrap on one pre-designated
split (seed 42). That interval is conditional on a single training-group
composition. This script adds:

* per-split grouped bootstrap intervals for every split;
* a two-stage (hierarchical) bootstrap that resamples splits and then source
  groups within splits, so between-split variability enters the interval;
* a paired split-level summary with a Student-t interval on the split means;
* a variance decomposition separating between-split from within-split
  (group-resampling) variance.

F1 at a fixed threshold decomposes into sums of per-group true positives, false
positives, and false negatives, so each bootstrap replicate is an O(#groups)
sum rather than a full metric recomputation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rev_common import PAIRS, RESULTS, SEEDS, TRANSFORMER_SEEDS

N_BOOTSTRAP = 5000
THRESHOLD = 0.5


def group_counts(frame: pd.DataFrame, probability_column: str):
    """Return per-group (tp, fp, fn) arrays for a fixed-threshold F1."""
    codes, _ = pd.factorize(frame["group_id"].to_numpy())
    n_groups = codes.max() + 1
    y = frame["label"].to_numpy().astype(bool)
    predicted = frame[probability_column].to_numpy() >= THRESHOLD
    tp = np.bincount(codes[y & predicted], minlength=n_groups).astype(float)
    fp = np.bincount(codes[~y & predicted], minlength=n_groups).astype(float)
    fn = np.bincount(codes[y & ~predicted], minlength=n_groups).astype(float)
    return tp, fp, fn


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def f1_total(counts) -> float:
    tp, fp, fn = counts
    return f1_from_counts(tp.sum(), fp.sum(), fn.sum())


def delta_replicates(counts_a, counts_b, draws: np.ndarray) -> np.ndarray:
    """Delta for each bootstrap draw, given group index draws of shape (B, G)."""
    out = np.empty(len(draws), dtype=float)
    for index, sample in enumerate(draws):
        a = [array[sample].sum() for array in counts_a]
        b = [array[sample].sum() for array in counts_b]
        out[index] = f1_from_counts(*b) - f1_from_counts(*a)
    return out


def two_sided_p(replicates: np.ndarray) -> float:
    return float(
        min(
            1.0,
            2.0
            * min((replicates <= 0).mean(), (replicates >= 0).mean()),
        )
    )


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        value = (len(ordered) - rank) * p
        running = max(running, min(1.0, value))
        adjusted[name] = running
    return adjusted


def analyse(label, per_seed_counts, rng):
    """per_seed_counts maps seed -> (counts_clean, counts_augmented)."""
    seeds = sorted(per_seed_counts)
    rows = []
    per_seed_delta = {}
    within_variance = []
    for seed in seeds:
        counts_a, counts_b = per_seed_counts[seed]
        n_groups = len(counts_a[0])
        draws = rng.integers(0, n_groups, size=(N_BOOTSTRAP, n_groups))
        replicates = delta_replicates(counts_a, counts_b, draws)
        observed = f1_total(counts_b) - f1_total(counts_a)
        per_seed_delta[seed] = observed
        within_variance.append(replicates.var(ddof=1))
        low, high = np.quantile(replicates, [0.025, 0.975])
        rows.append(
            {
                "family": label,
                "scope": "single split (grouped bootstrap)",
                "seed": seed,
                "f1_clean": f1_total(counts_a),
                "f1_augmented": f1_total(counts_b),
                "delta": observed,
                "ci_low": float(low),
                "ci_high": float(high),
                "p_two_sided": two_sided_p(replicates),
            }
        )

    # Two-stage bootstrap: resample splits, then groups within each split.
    hierarchical = np.empty(N_BOOTSTRAP, dtype=float)
    seed_array = np.array(seeds)
    for index in range(N_BOOTSTRAP):
        chosen = rng.choice(seed_array, size=len(seed_array), replace=True)
        values = []
        for seed in chosen:
            counts_a, counts_b = per_seed_counts[seed]
            n_groups = len(counts_a[0])
            sample = rng.integers(0, n_groups, size=n_groups)
            a = [array[sample].sum() for array in counts_a]
            b = [array[sample].sum() for array in counts_b]
            values.append(f1_from_counts(*b) - f1_from_counts(*a))
        hierarchical[index] = float(np.mean(values))
    low, high = np.quantile(hierarchical, [0.025, 0.975])
    mean_delta = float(np.mean(list(per_seed_delta.values())))
    rows.append(
        {
            "family": label,
            "scope": "all splits (hierarchical bootstrap)",
            "seed": -1,
            "f1_clean": float(
                np.mean([f1_total(per_seed_counts[s][0]) for s in seeds])
            ),
            "f1_augmented": float(
                np.mean([f1_total(per_seed_counts[s][1]) for s in seeds])
            ),
            "delta": mean_delta,
            "ci_low": float(low),
            "ci_high": float(high),
            "p_two_sided": two_sided_p(hierarchical),
        }
    )

    # Paired split-level Student-t interval on the split means.
    values = np.array([per_seed_delta[s] for s in seeds])
    n = len(values)
    sd = values.std(ddof=1)
    from scipy import stats

    t_critical = stats.t.ppf(0.975, n - 1)
    half_width = t_critical * sd / np.sqrt(n)
    t_statistic = values.mean() / (sd / np.sqrt(n)) if sd > 0 else np.inf
    rows.append(
        {
            "family": label,
            "scope": "all splits (paired split-level t)",
            "seed": -1,
            "f1_clean": float(
                np.mean([f1_total(per_seed_counts[s][0]) for s in seeds])
            ),
            "f1_augmented": float(
                np.mean([f1_total(per_seed_counts[s][1]) for s in seeds])
            ),
            "delta": float(values.mean()),
            "ci_low": float(values.mean() - half_width),
            "ci_high": float(values.mean() + half_width),
            "p_two_sided": float(
                2 * (1 - stats.t.cdf(abs(t_statistic), n - 1)) if sd > 0 else 0.0
            ),
        }
    )

    decomposition = {
        "family": label,
        "splits": n,
        "between_split_variance": float(values.var(ddof=1)),
        "mean_within_split_bootstrap_variance": float(np.mean(within_variance)),
        "between_split_sd": float(values.std(ddof=1)),
        "mean_within_split_bootstrap_sd": float(np.mean(np.sqrt(within_variance))),
        "hierarchical_sd": float(hierarchical.std(ddof=1)),
        "per_split_delta": {int(s): float(per_seed_delta[s]) for s in seeds},
    }
    return rows, decomposition, two_sided_p(hierarchical)


def main() -> None:
    rng = np.random.default_rng(20260808)
    table = pd.read_csv(RESULTS / "revision_probabilities_all_seeds.csv")
    linked = table[table["condition"] == "linked_rewrites"]

    rows: list[dict] = []
    decompositions: list[dict] = []
    hierarchical_p: dict[str, float] = {}

    for label, clean_name, augmented_name in PAIRS:
        clean_column = "probability_" + clean_name.lower().replace("+", "plus").replace(
            ",", ""
        ).replace(" ", "_").replace("-", "_")
        augmented_column = "probability_" + augmented_name.lower().replace(
            "+", "plus"
        ).replace(",", "").replace(" ", "_").replace("-", "_")
        per_seed = {}
        for seed in SEEDS:
            frame = linked[linked["seed"] == seed].reset_index(drop=True)
            per_seed[seed] = (
                group_counts(frame, clean_column),
                group_counts(frame, augmented_column),
            )
        new_rows, decomposition, p = analyse(label, per_seed, rng)
        rows.extend(new_rows)
        decompositions.append(decomposition)
        hierarchical_p[label] = p

    distilbert_path = RESULTS / "revision_distilbert_probabilities.csv"
    if distilbert_path.exists():
        frame = pd.read_csv(distilbert_path)
        frame = frame[
            (frame["max_length"] == 160)
            & (frame["epoch"] == 1)
            & (frame["condition"] == "linked_rewrites")
        ]
        linked_groups = {
            seed: linked[linked["seed"] == seed]["group_id"].to_numpy()
            for seed in TRANSFORMER_SEEDS
        }
        per_seed = {}
        for seed in TRANSFORMER_SEEDS:
            block = frame[frame["seed"] == seed]
            if block.empty:
                continue
            clean = block[block["training"] == "clean-only"].reset_index(drop=True)
            augmented = block[block["training"] == "linked-augmented"].reset_index(
                drop=True
            )
            if len(clean) != len(linked_groups[seed]):
                raise ValueError("DistilBERT rows do not align with the linked frame")
            clean = clean.assign(group_id=linked_groups[seed])
            augmented = augmented.assign(group_id=linked_groups[seed])
            per_seed[seed] = (
                group_counts(clean, "probability"),
                group_counts(augmented, "probability"),
            )
        if per_seed:
            new_rows, decomposition, p = analyse("DistilBERT", per_seed, rng)
            rows.extend(new_rows)
            decompositions.append(decomposition)
            hierarchical_p["DistilBERT"] = p

    adjusted = holm(hierarchical_p)
    output = pd.DataFrame(rows)
    output["holm_adjusted_p"] = [
        adjusted[row.family]
        if row.scope == "all splits (hierarchical bootstrap)"
        else np.nan
        for row in output.itertuples()
    ]
    output.to_csv(RESULTS / "revision_multisplit_inference.csv", index=False)
    (RESULTS / "revision_variance_decomposition.json").write_text(
        json.dumps(
            {
                "bootstrap_replicates": N_BOOTSTRAP,
                "families": decompositions,
                "hierarchical_p_two_sided": hierarchical_p,
                "holm_adjusted": adjusted,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
