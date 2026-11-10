#!/usr/bin/env python3
"""Paired inference over 100 (lightweight) and 20 (DistilBERT) refitted partitions,
with an explicit test of whether the interval width has stabilized.

Round-4 comments R1.2 and R2.2. The reviewer observes that moving from five to
twenty partitions raised the between-partition SD of the word model's effect by
a factor of 3.6, asks whether twenty is enough, and asks for either more
partitions or a statistical argument that the interval width has stabilized.
Both reviewers also ask that DistilBERT's wide prediction interval be carried
into the claims. This script answers with data and with an argument:

1. All round-3 procedures, now over 100 lightweight and 20 DistilBERT partitions
   (hierarchical bootstrap, split-level t, exact or Monte-Carlo sign-flip test
   with its minimum attainable p, and a t prediction interval), plus a
   distribution-free prediction interval from order statistics.

2. The corrected resampled t-test of Nadeau and Bengio (2003). Partitions drawn
   from one corpus share most of their training and test groups, so they are not
   independent replicates and the split-level t interval is anti-conservative.
   The correction inflates the variance of the mean by (1/K + n_test/n_train);
   as K grows its half-width converges to a floor set by the corpus, not by the
   number of partitions. That floor is the formal statement of "stabilized":
   the ratio of the K-partition half-width to the floor is reported for every K.

3. A random-effects (DerSimonian-Laird) decomposition with a Q-profile
   confidence interval for the between-partition variance tau^2 and the I^2
   statistic. Unlike the round-3 "between-partition share", which counted
   test-set sampling noise as between-partition variability, tau^2 is the
   refit-to-refit heterogeneity left after test-set sampling is accounted for.

4. Stabilization diagnostics: for K = 3..100, the SD of the partition effects,
   the t and corrected half-widths and the prediction interval, both in the
   order the partitions were run and over 2,000 random orderings (median and
   5th-95th percentile bands); a chi-square interval for the SD at each K; the
   probability, under the 100-partition SD, of five partitions agreeing as
   closely as the five published ones did; and tests of whether partitions
   1-20 and 21-100 differ in mean, spread or distribution.

5. The same partition-level summaries for the generator-disjoint (L2) and, for
   DistilBERT, human-authored (L3) augmentation contrasts, which round 3 reported
   only as means over five seeds.

Outputs (results/):
  revision_r4_inference.csv
  revision_r4_heterogeneity.json
  revision_r4_stabilization.csv
  revision_r4_stabilization_tests.json
  revision_r4_partition_effects.csv          per partition x family x stream
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rev_common import RESULTS
from rev_l_r3_inference import distilbert_counts, group_counts
from rev_k_multisplit import SPLIT_SEEDS as R3_SEEDS
from rev_q_multisplit100 import NEW_SEEDS as R4_LIGHT_SEEDS
from rev_p_distilbert_multisplit20 import EXISTING_SEEDS, NEW_SEEDS as R4_BERT_SEEDS

N_BOOTSTRAP = 5000
N_ORDERINGS = 2000
RNG_SEED = 20261010
TRAIN_GROUPS = 1509
TEST_GROUPS = 324
RHO = TEST_GROUPS / (TRAIN_GROUPS + TEST_GROUPS)
NB_FACTOR = TEST_GROUPS / TRAIN_GROUPS  # n_test / n_train in Nadeau-Bengio
K_GRID = [3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

LIGHT_ORDER = list(R3_SEEDS) + list(R4_LIGHT_SEEDS)
BERT_ORDER = list(EXISTING_SEEDS) + list(R4_BERT_SEEDS)


# --------------------------------------------------------------------------
# per-partition counts
# --------------------------------------------------------------------------
def f1(tp, fp, fn):
    denominator = 2.0 * tp + fp + fn
    return np.where(denominator > 0, 2.0 * tp / np.maximum(denominator, 1e-12), 0.0)


def light_counts(clean_column: str, augmented_column: str) -> dict[int, tuple]:
    frames = [
        pd.read_csv(RESULTS / "revision_r3_multisplit_probabilities.csv"),
        pd.read_csv(RESULTS / "revision_r4_multisplit100_probabilities.csv"),
    ]
    per_split: dict[int, tuple] = {}
    for table in frames:
        linked = table[table["condition"] == "linked_rewrites"]
        for seed, frame in linked.groupby("seed", sort=False):
            frame = frame.reset_index(drop=True)
            per_split[int(seed)] = (
                group_counts(frame, clean_column),
                group_counts(frame, augmented_column),
            )
    return per_split


def bert_counts() -> dict[int, tuple]:
    per_split = distilbert_counts()
    table = pd.read_csv(RESULTS / "revision_r4_distilbert_multisplit_probabilities.csv")
    table = table[table["condition"] == "linked_rewrites"]
    for seed, block in table.groupby("seed"):
        clean = block[block["training"] == "clean-only"].reset_index(drop=True)
        augmented = block[block["training"] == "linked-augmented"].reset_index(drop=True)
        per_split[int(seed)] = (
            group_counts(clean, "probability"),
            group_counts(augmented, "probability"),
        )
    return per_split


def stack(per_split: dict[int, tuple], order: list[int]):
    """Arrays of shape (K, G) for the six count vectors, in the given order."""
    order = [s for s in order if s in per_split]
    sizes = {len(per_split[s][0][0]) for s in order}
    if len(sizes) != 1:
        raise ValueError(f"unequal numbers of test groups: {sizes}")
    arrays = [
        np.stack([per_split[s][side][kind] for s in order])
        for side in (0, 1)
        for kind in (0, 1, 2)
    ]
    return order, arrays


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------
def within_partition(arrays, rng, n=N_BOOTSTRAP):
    """Observed delta and within-partition grouped-bootstrap replicates."""
    tp_a, fp_a, fn_a, tp_b, fp_b, fn_b = arrays
    k, g = tp_a.shape
    observed = f1(tp_b.sum(1), fp_b.sum(1), fn_b.sum(1)) - f1(
        tp_a.sum(1), fp_a.sum(1), fn_a.sum(1)
    )
    replicates = np.empty((k, n))
    for i in range(k):
        draws = rng.integers(0, g, size=(n, g))
        replicates[i] = f1(
            tp_b[i][draws].sum(1), fp_b[i][draws].sum(1), fn_b[i][draws].sum(1)
        ) - f1(tp_a[i][draws].sum(1), fp_a[i][draws].sum(1), fn_a[i][draws].sum(1))
    return observed, replicates


def hierarchical(arrays, rng, n=N_BOOTSTRAP, chunk=200):
    tp_a, fp_a, fn_a, tp_b, fp_b, fn_b = arrays
    k, g = tp_a.shape
    out = np.empty(n)
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        chosen = rng.integers(0, k, size=(m, k))
        draws = rng.integers(0, g, size=(m, k, g))
        rows = chosen[:, :, None]

        def total(array):
            return array[rows, draws].sum(axis=2)

        delta = f1(total(tp_b), total(fp_b), total(fn_b)) - f1(
            total(tp_a), total(fp_a), total(fn_a)
        )
        out[start : start + m] = delta.mean(axis=1)
    return out


def sign_flip(values, rng):
    """Two-sided sign-flip test; exact up to 20 partitions, Monte Carlo above."""
    k = len(values)
    observed = abs(values.mean())
    if k <= 20:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=k)))
        means = np.abs(signs @ values / k)
        return float((means >= observed - 1e-15).mean()), 2.0 / 2**k, "exact"
    draws = 200_000
    signs = rng.choice([-1.0, 1.0], size=(draws, k))
    means = np.abs(signs @ values / k)
    p = float(((means >= observed - 1e-15).sum() + 1) / (draws + 1))
    return p, 1.0 / (draws + 1), "Monte Carlo (200,000 sign vectors)"


# --------------------------------------------------------------------------
# heterogeneity
# --------------------------------------------------------------------------
def dersimonian_laird(values, variances):
    k = len(values)
    w = 1.0 / variances
    mu_fixed = float((w * values).sum() / w.sum())
    q = float((w * (values - mu_fixed) ** 2).sum())
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c)
    w_star = 1.0 / (variances + tau2)
    mu = float((w_star * values).sum() / w_star.sum())
    se = float(np.sqrt(1.0 / w_star.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0

    def q_gen(t2):
        ws = 1.0 / (variances + t2)
        m = (ws * values).sum() / ws.sum()
        return float((ws * (values - m) ** 2).sum())

    upper_target = stats.chi2.ppf(0.025, k - 1)
    lower_target = stats.chi2.ppf(0.975, k - 1)
    span = 100.0 * float(np.var(values, ddof=1) + variances.mean())

    def solve(target):
        if q_gen(0.0) < target:
            return 0.0
        return float(optimize.brentq(lambda t2: q_gen(t2) - target, 0.0, span))

    tau2_low, tau2_high = solve(lower_target), solve(upper_target)
    t_crit = stats.t.ppf(0.975, k - 2)
    half = t_crit * np.sqrt(tau2 + se**2)
    return {
        "mu_random_effects": mu,
        "se_random_effects": se,
        "Q": q,
        "Q_df": k - 1,
        "Q_p": float(1 - stats.chi2.cdf(q, k - 1)),
        "tau2": tau2,
        "tau": float(np.sqrt(tau2)),
        "tau_ci_low": float(np.sqrt(tau2_low)),
        "tau_ci_high": float(np.sqrt(tau2_high)),
        "I2": i2,
        "prediction_low_random_effects": float(mu - half),
        "prediction_high_random_effects": float(mu + half),
        "mean_within_sd": float(np.sqrt(variances.mean())),
    }


# --------------------------------------------------------------------------
# interval summaries for a set of partition effects
# --------------------------------------------------------------------------
def summaries(values):
    k = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    t_crit = float(stats.t.ppf(0.975, k - 1))
    naive_se = sd / np.sqrt(k)
    corrected_se = sd * np.sqrt(1.0 / k + NB_FACTOR)
    floor_se = sd * np.sqrt(NB_FACTOR)
    corrected_t = mean / corrected_se if corrected_se > 0 else np.inf
    prediction_half = t_crit * sd * np.sqrt(1.0 + 1.0 / k)
    return {
        "k": k,
        "mean": mean,
        "sd": sd,
        "sd_chi2_low": float(sd * np.sqrt((k - 1) / stats.chi2.ppf(0.975, k - 1))),
        "sd_chi2_high": float(sd * np.sqrt((k - 1) / stats.chi2.ppf(0.025, k - 1))),
        "t_low": mean - t_crit * naive_se,
        "t_high": mean + t_crit * naive_se,
        "t_half_width": t_crit * naive_se,
        "t_p": float(2 * (1 - stats.t.cdf(abs(mean / naive_se), k - 1))) if naive_se > 0 else 0.0,
        "corrected_low": mean - t_crit * corrected_se,
        "corrected_high": mean + t_crit * corrected_se,
        "corrected_half_width": t_crit * corrected_se,
        "corrected_floor_half_width": float(stats.norm.ppf(0.975) * floor_se),
        "corrected_to_floor_ratio": float(t_crit * corrected_se / (stats.norm.ppf(0.975) * floor_se)),
        "corrected_p": float(2 * (1 - stats.t.cdf(abs(corrected_t), k - 1))),
        "prediction_low": mean - prediction_half,
        "prediction_high": mean + prediction_half,
        "share_positive": float((values > 0).mean()),
    }


def order_statistic_interval(values):
    """Distribution-free prediction interval [x_(j), x_(k+1-j)] closest to 95%."""
    k = len(values)
    ordered = np.sort(values)
    best = None
    for j in range(1, k // 2 + 1):
        coverage = (k + 1 - 2 * j) / (k + 1)
        if coverage >= 0.95 or best is None:
            best = (j, coverage)
        if coverage < 0.95:
            break
    j, coverage = best
    return float(ordered[j - 1]), float(ordered[k - j]), float(coverage)


def analyse_family(label, per_split, order, rng):
    order, arrays = stack(per_split, order)
    observed, replicates = within_partition(arrays, rng)
    within_var = replicates.var(axis=1, ddof=1)
    hier = hierarchical(arrays, rng)
    flip_p, flip_min, flip_kind = sign_flip(observed, rng)
    base = summaries(observed)
    np_low, np_high, np_cov = order_statistic_interval(observed)
    het = dersimonian_laird(observed, within_var)

    rows = []
    for i, seed in enumerate(order):
        low, high = np.quantile(replicates[i], [0.025, 0.975])
        rows.append(
            {
                "family": label,
                "scope": "single split (grouped bootstrap)",
                "splits": 1,
                "seed": seed,
                "run_order": i + 1,
                "delta": float(observed[i]),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_two_sided": float(
                    min(1.0, 2 * min((replicates[i] <= 0).mean(), (replicates[i] >= 0).mean()))
                ),
            }
        )
    k = len(order)
    h_low, h_high = np.quantile(hier, [0.025, 0.975])
    common = {"family": label, "splits": k, "seed": -1, "run_order": -1, "delta": base["mean"]}
    rows += [
        {**common, "scope": "all splits (hierarchical bootstrap)", "ci_low": float(h_low),
         "ci_high": float(h_high),
         "p_two_sided": float(min(1.0, 2 * min((hier <= 0).mean(), (hier >= 0).mean())))},
        {**common, "scope": "all splits (paired split-level t)", "ci_low": base["t_low"],
         "ci_high": base["t_high"], "p_two_sided": base["t_p"]},
        {**common, "scope": "all splits (corrected resampled t, Nadeau-Bengio)",
         "ci_low": base["corrected_low"], "ci_high": base["corrected_high"],
         "p_two_sided": base["corrected_p"]},
        {**common, "scope": f"all splits (sign-flip permutation, {flip_kind})",
         "ci_low": np.nan, "ci_high": np.nan, "p_two_sided": flip_p,
         "p_minimum_attainable": flip_min},
        {**common, "scope": "one further split (t prediction interval)",
         "ci_low": base["prediction_low"], "ci_high": base["prediction_high"],
         "p_two_sided": np.nan},
        {**common, "scope": f"one further split (order-statistic interval, coverage {np_cov:.3f})",
         "ci_low": np_low, "ci_high": np_high, "p_two_sided": np.nan},
        {**common, "scope": "one further split (random-effects prediction interval)",
         "ci_low": het["prediction_low_random_effects"],
         "ci_high": het["prediction_high_random_effects"], "p_two_sided": np.nan},
    ]
    between_var = float(observed.var(ddof=1))
    decomposition = {
        "family": label,
        "splits": k,
        "delta_mean": base["mean"],
        "between_split_sd": float(np.sqrt(between_var)),
        "mean_within_split_bootstrap_sd": float(np.sqrt(within_var.mean())),
        "round3_between_split_share": float(between_var / (between_var + within_var.mean())),
        "min_split_delta": float(observed.min()),
        "max_split_delta": float(observed.max()),
        "splits_with_nonpositive_delta": int((observed <= 0).sum()),
        "shapiro_wilk_p": float(stats.shapiro(observed).pvalue) if k >= 3 else np.nan,
        "sign_flip_minimum_attainable_p": flip_min,
        **{f"re_{key}": value for key, value in het.items()},
        **{f"summary_{key}": value for key, value in base.items()},
    }
    return rows, decomposition, observed, order, arrays


def stabilization(label, observed, rng):
    k_max = len(observed)
    rows = []
    for k in [k for k in K_GRID if k <= k_max] + ([k_max] if k_max not in K_GRID else []):
        in_order = summaries(observed[:k])
        rows.append({"family": label, "ordering": "as run", "k": k, "quantile": "observed",
                     **{key: in_order[key] for key in
                        ("mean", "sd", "t_half_width", "corrected_half_width",
                         "corrected_to_floor_ratio", "prediction_low", "prediction_high",
                         "sd_chi2_low", "sd_chi2_high")}})
        if k == k_max:
            continue
        draws = [summaries(rng.choice(observed, size=k, replace=False))
                 for _ in range(N_ORDERINGS)]
        for quantile, name in ((0.05, "p05"), (0.5, "median"), (0.95, "p95")):
            rows.append({"family": label, "ordering": "random", "k": k, "quantile": name,
                         **{key: float(np.quantile([d[key] for d in draws], quantile))
                            for key in ("mean", "sd", "t_half_width", "corrected_half_width",
                                        "corrected_to_floor_ratio", "prediction_low",
                                        "prediction_high")}})
        rows.append({"family": label, "ordering": "random", "k": k,
                     "quantile": "share_sd_within_10pct_of_full",
                     "sd": float(np.mean([abs(d["sd"] / observed.std(ddof=1) - 1) <= 0.10
                                          for d in draws]))})
    return rows


def stabilization_tests(label, observed, first):
    head, tail = observed[:first], observed[first:]
    full_sd = observed.std(ddof=1)
    published = observed[:5]
    s5 = published.std(ddof=1)
    statistic = 4 * s5**2 / full_sd**2
    out = {
        "family": label,
        "first_block": first,
        "first_block_mean": float(head.mean()),
        "second_block_mean": float(tail.mean()),
        "first_block_sd": float(head.std(ddof=1)),
        "second_block_sd": float(tail.std(ddof=1)),
        "welch_t_p": float(stats.ttest_ind(head, tail, equal_var=False).pvalue),
        "brown_forsythe_p": float(stats.levene(head, tail, center="median").pvalue),
        "ks_p": float(stats.ks_2samp(head, tail).pvalue),
        "full_sd": float(full_sd),
        "published_five_sd": float(s5),
        "p_five_as_tight_as_published": float(stats.chi2.cdf(statistic, 4)),
    }
    return out


# --------------------------------------------------------------------------
# partition-level contrasts on the other streams
# --------------------------------------------------------------------------
def stream_effects():
    rows = []
    light = pd.concat(
        [
            pd.read_csv(RESULTS / "revision_r3_multisplit_metrics.csv"),
            pd.read_csv(RESULTS / "revision_r4_multisplit100_metrics.csv"),
        ],
        ignore_index=True,
    )
    pairs = [
        ("Word TF-IDF", "Word LR, clean-only", "Word LR, linked-augmented"),
        ("MiniLM", "MiniLM LR, clean-only", "MiniLM LR, linked-augmented"),
    ]
    for family, clean, augmented in pairs:
        for stream in ("linked_rewrites", "external_openai", "clean"):
            block = light[light["condition"] == stream]
            a = block[block["model"] == clean].set_index("seed")
            b = block[block["model"] == augmented].set_index("seed")
            for metric in ("f1", "roc_auc", "pr_auc", "ece"):
                for seed in LIGHT_ORDER:
                    if seed in a.index and seed in b.index:
                        rows.append({"family": family, "stream": stream, "metric": metric,
                                     "seed": seed, "clean": a.at[seed, metric],
                                     "augmented": b.at[seed, metric],
                                     "delta": b.at[seed, metric] - a.at[seed, metric]})
    # The three published partitions are taken from rev_g, which scored the
    # pre-registered DistilBERT configuration on every stream (including L3);
    # the other seventeen come from the round-3 and round-4 refits.
    bert = pd.concat(
        [
            pd.read_csv(RESULTS / "revision_distilbert_stream_metrics.csv"),
            pd.read_csv(RESULTS / "revision_r3_distilbert_multisplit_metrics.csv"),
            pd.read_csv(RESULTS / "revision_r4_distilbert_multisplit_metrics.csv"),
        ],
        ignore_index=True,
    )
    streams = {
        "linked_rewrites": "linked_rewrites",
        "external_openai": "external_openai",
        "clean": "clean",
        "human_phishing": "human_phishing",
    }
    for stream in streams:
        block = bert[bert["condition"] == stream]
        a = block[block["training"] == "clean-only"].drop_duplicates("seed").set_index("seed")
        b = block[block["training"] == "linked-augmented"].drop_duplicates("seed").set_index("seed")
        for metric in ("f1", "roc_auc", "pr_auc", "ece"):
            for seed in BERT_ORDER:
                if seed in a.index and seed in b.index:
                    rows.append({"family": "DistilBERT", "stream": stream, "metric": metric,
                                 "seed": seed, "clean": a.at[seed, metric],
                                 "augmented": b.at[seed, metric],
                                 "delta": b.at[seed, metric] - a.at[seed, metric]})
    return pd.DataFrame(rows)


def stream_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, stream, metric), block in effects.groupby(["family", "stream", "metric"]):
        values = block["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            continue
        s = summaries(values)
        np_low, np_high, np_cov = order_statistic_interval(values)
        rows.append({"family": family, "stream": stream, "metric": metric,
                     "clean_mean": float(block["clean"].mean()),
                     "augmented_mean": float(block["augmented"].mean()),
                     **s, "order_low": np_low, "order_high": np_high,
                     "order_coverage": np_cov})
    return pd.DataFrame(rows)


def holm(p_values):
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows, decompositions, stab_rows, tests = [], [], [], []
    families = [
        ("Word TF-IDF", light_counts("probability_word_lr_clean_only",
                                     "probability_word_lr_linked_augmented"), LIGHT_ORDER, 20),
        ("MiniLM", light_counts("probability_minilm_lr_clean_only",
                                "probability_minilm_lr_linked_augmented"), LIGHT_ORDER, 20),
        ("DistilBERT", bert_counts(), BERT_ORDER, 8),
    ]
    corrected_p = {}
    for label, per_split, order, first in families:
        new_rows, decomposition, observed, used_order, _ = analyse_family(
            label, per_split, order, rng
        )
        rows += new_rows
        decompositions.append(decomposition)
        corrected_p[label] = decomposition["summary_corrected_p"]
        stab_rows += stabilization(label, observed, rng)
        tests.append(stabilization_tests(label, observed, first))
        print(f"{label}: {len(used_order)} partitions, mean {observed.mean():.4f}, "
              f"sd {observed.std(ddof=1):.4f}")

    adjusted = holm(corrected_p)
    for decomposition in decompositions:
        decomposition["holm_corrected_resampled_p"] = adjusted[decomposition["family"]]
    pd.DataFrame(rows).to_csv(RESULTS / "revision_r4_inference.csv", index=False)
    (RESULTS / "revision_r4_heterogeneity.json").write_text(
        json.dumps({"rho_between_partitions": RHO, "nadeau_bengio_factor": NB_FACTOR,
                    "families": decompositions}, indent=2) + "\n")
    pd.DataFrame(stab_rows).to_csv(RESULTS / "revision_r4_stabilization.csv", index=False)
    (RESULTS / "revision_r4_stabilization_tests.json").write_text(
        json.dumps(tests, indent=2) + "\n")

    effects = stream_effects()
    effects.to_csv(RESULTS / "revision_r4_partition_effects.csv", index=False)
    summary = stream_summary(effects)
    summary.to_csv(RESULTS / "revision_r4_partition_effect_summary.csv", index=False)

    table = pd.DataFrame(rows)
    print(table[table["seed"] == -1][["family", "scope", "splits", "delta", "ci_low",
                                      "ci_high", "p_two_sided"]].to_string(index=False))
    print(json.dumps(tests, indent=2))
    for decomposition in decompositions:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in decomposition.items()
               if k.startswith(("re_", "round3", "between", "mean_within", "shapiro"))})
    print(summary[summary["metric"] == "f1"][["family", "stream", "k", "clean_mean",
                                              "augmented_mean", "mean", "sd", "prediction_low",
                                              "prediction_high", "share_positive"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
