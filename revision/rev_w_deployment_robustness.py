#!/usr/bin/env python3
"""How robust is the operating-point procedure? (round-4 comment R1.5)

Round 3 reduced the threshold decision to one number, pi * r, and a three-band
rule: below pi*r ~ 0.1 use the validation cost-minimizing threshold, between
about 0.2 and 1 the 95%-recall rule, above that threshold 0.5. Reviewer 1 asks
how a practitioner would estimate pi and r, whether the rule survives an
independent stream and other score distributions, and for a sensitivity
analysis. Four analyses:

1. Estimating prevalence without labels. On each stream, at true prevalence
   0.1%, 1% and 5%, prevalence is estimated by classify-and-count (CC), by
   adjusted classify-and-count (ACC; Forman, 2008) at two thresholds, and by
   the EM prior adjustment of Saerens et al. (2002); the excess cost of
   choosing the threshold from the estimate rather than the truth is reported.
   Expected values are used, so the numbers carry no sampling noise.

2. Misestimating the cost ratio. The threshold chosen under a believed r is
   deployed at a different true r.

3. Held-out validation of the rule. The stated three-band rule, and a
   decile-to-rule map learned on one set of partitions, are applied to (a)
   held-out partitions, (b) every other detector, (c) held-out streams, and (d)
   a stream that played no part in deriving the rule: era-matched real
   phishing (Nazario 2015-2025) mixed with real legitimate mail (the 2000s L3
   legitimate class and SpamAssassin ham, without the L3 rows that duplicate a
   SpamAssassin message; see rev_t3_l3_sa_overlap.py). Agreement with
   the best implementable rule and excess cost over it are reported.

4. Score-distribution sensitivity. Probabilities are perturbed by temperature,
   logit shift and logit noise -- applied to validation and target together
   (a detector with a differently shaped score distribution) or to the target
   alone (deployment shift) -- and the pi*r purity and the rule's excess cost
   are recomputed.

Outputs (results/): revision_r4_deploy_*.csv (the always-cost-min rule under the
perturbations: revision_r4_deploy_score_sensitivity_costmin.csv)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from rev_c_thresholds import GRID, flagged_fraction_threshold, rates, threshold_for_recall
from rev_common import RESULTS, base, encode_minilm, load_corpora, load_embeddings, split_frames
from rev_era_data import load_era_matched
from rev_o_deployment import COST_RATIOS, PREVALENCES
from rev_realworld_data import load_human_phishing, load_spamassassin

RULES = ["fixed_0.5", "fixed_95_recall", "fixed_cost_min", "dynamic_rate_match"]
TRUE_PREVALENCE = [0.001, 0.01, 0.05]


# --------------------------------------------------------------------------
# stream store: {(model, seed): {stream: (y, p)}} including "validation"
# --------------------------------------------------------------------------
def sa_duplicates() -> set:
    """L3 rows that duplicate a SpamAssassin message (rev_t3_l3_sa_overlap.py)."""
    table = pd.read_csv(RESULTS / "revision_r4_l3_sa_duplicates.csv")
    return set(table.loc[table["duplicate"], "row_index"].astype(int))


def build_store():
    """Every stream; the L3 aggregate is deduplicated against SpamAssassin, so no
    message enters the human-authored stream twice with contradictory labels."""
    store: dict = {}
    drop = sa_duplicates()
    light = pd.read_csv(RESULTS / "revision_r3_multisplit_probabilities.csv")
    columns = [c for c in light.columns if c.startswith("probability_")]
    realworld = pd.read_csv(RESULTS / "revision_realworld_probabilities.csv")
    realworld = realworld[~((realworld["condition"] == "human_phishing") & realworld["row_index"].isin(drop))]
    for seed, block in light.groupby("seed"):
        for column in columns:
            model = column.replace("probability_", "")
            get = lambda c: block[block["condition"] == c]  # noqa: E731
            entry = {
                "validation": (get("validation")["label"].to_numpy(), get("validation")[column].to_numpy()),
                "L1 primary": (
                    np.r_[get("linked_rewrites")["label"].to_numpy(), get("spam_hard_negative")["label"].to_numpy()],
                    np.r_[get("linked_rewrites")[column].to_numpy(), get("spam_hard_negative")[column].to_numpy()],
                ),
                "L2 generator-disjoint": (get("external_openai")["label"].to_numpy(), get("external_openai")[column].to_numpy()),
            }
            rw = realworld[realworld["seed"] == seed]
            if len(rw) and column in rw:
                parts = [rw[rw["condition"] == c] for c in ("human_phishing", "spamassassin_ham", "spamassassin_spam")]
                entry["L3 human-authored"] = (np.concatenate([p["label"].to_numpy() for p in parts]),
                                              np.concatenate([p[column].to_numpy() for p in parts]))
            store[(model, int(seed))] = entry

    bert_streams = pd.read_csv(RESULTS / "revision_distilbert_stream_probabilities.csv")
    r4 = pd.read_csv(RESULTS / "revision_r4_distilbert_multisplit_probabilities.csv")
    for table in (bert_streams, r4):
        table = table[~((table["condition"] == "human_phishing") & table["row_index"].isin(drop))]
        for (seed, training), block in table.groupby(["seed", "training"]):
            get = lambda c: block[block["condition"] == c]  # noqa: E731
            model = f"distilbert_{training.replace('-', '_')}"
            entry = {
                "validation": (get("validation")["label"].to_numpy(), get("validation")["probability"].to_numpy()),
                "L1 primary": (np.r_[get("linked_rewrites")["label"].to_numpy(), get("spam_hard_negative")["label"].to_numpy()],
                               np.r_[get("linked_rewrites")["probability"].to_numpy(), get("spam_hard_negative")["probability"].to_numpy()]),
                "L2 generator-disjoint": (get("external_openai")["label"].to_numpy(), get("external_openai")["probability"].to_numpy()),
            }
            parts = [get(c) for c in ("human_phishing", "spamassassin_ham", "spamassassin_spam")]
            if all(len(p) for p in parts):
                entry["L3 human-authored"] = (np.concatenate([p["label"].to_numpy() for p in parts]),
                                              np.concatenate([p["probability"].to_numpy() for p in parts]))
            store[(model, int(seed))] = entry
    return store


def add_contemporary_stream(store):
    """Nazario 2015-2025 positives + real legitimate negatives, five partitions."""
    primary, external, binary, spam = load_corpora()
    primary_embeddings, _ = load_embeddings(primary, external)
    nazario = load_era_matched()
    human = load_human_phishing()
    legit = human[(human["label"] == 0) & ~human.index.isin(sa_duplicates())]
    ham = load_spamassassin()
    ham = ham[ham["stream"] == "ham"]
    texts = nazario["text_full"].tolist() + legit["text"].tolist() + ham["text"].tolist()
    y = np.r_[np.ones(len(nazario)), np.zeros(len(legit) + len(ham))].astype(int)
    embedded = np.vstack([encode_minilm(nazario["text_full"]), encode_minilm(legit["text"]), encode_minilm(ham["text"])])
    for seed in (17, 29, 42, 71, 113):
        frames = split_frames(binary, external, spam, seed)
        for training, frame in (("clean_only", frames["train_clean"]), ("linked_augmented", frames["train_all"])):
            word = base.WordModel.fit(frame["text"].tolist(), frame["label"].to_numpy())
            embed = base.EmbeddingModel.fit(primary_embeddings[frame.index.to_numpy()], frame["label"].to_numpy())
            for model, p in ((f"word_lr_{training}", word.predict_proba(texts)),
                             (f"minilm_lr_{training}", embed.predict_proba(embedded))):
                if (model, seed) in store:
                    store[(model, seed)]["Contemporary (Nazario + real legitimate)"] = (y, p)
    return store


# --------------------------------------------------------------------------
# rule machinery
# --------------------------------------------------------------------------
def rule_thresholds(validation, prevalence, ratio):
    vy, vp = validation
    tpr, fpr = rates(vy, vp, GRID)
    cost = (1 - prevalence) * fpr + prevalence * ratio * (1 - tpr)
    t_cost = float(GRID[int(np.argmin(cost))])
    rate = float((vp >= t_cost).mean())
    return {"fixed_0.5": 0.5, "fixed_95_recall": threshold_for_recall(vy, vp),
            "fixed_cost_min": t_cost, "dynamic_rate_match": ("rate", rate)}


def realized(y, p, threshold, prevalence, ratio):
    if isinstance(threshold, tuple):
        threshold = flagged_fraction_threshold(p, threshold[1])
    flagged = p >= threshold
    tpr = flagged[y == 1].mean()
    fpr = flagged[y == 0].mean()
    return float((1 - prevalence) * fpr + prevalence * ratio * (1 - tpr))


def cell_costs(validation, target, prevalence, ratio):
    thresholds = rule_thresholds(validation, prevalence, ratio)
    y, p = target
    return {rule: realized(y, p, thresholds[rule], prevalence, ratio) for rule in RULES}


def three_band(miss_cost):
    if miss_cost < 0.15:
        return "fixed_cost_min"
    if miss_cost <= 1.5:
        return "fixed_95_recall"
    return "fixed_0.5"


def grid_table(store, streams=None, perturb=None):
    rows = []
    for (model, seed), entry in store.items():
        validation = entry["validation"]
        for stream, target in entry.items():
            if stream == "validation" or (streams and stream not in streams):
                continue
            v, t = validation, target
            if perturb is not None:
                v, t = perturb(v, t)
            for prevalence in PREVALENCES:
                for ratio in COST_RATIOS:
                    costs = cell_costs(v, t, prevalence, ratio)
                    best = min(costs, key=costs.get)
                    rows.append({"model": model, "seed": seed, "stream": stream, "prevalence": prevalence,
                                 "cost_ratio": ratio, "miss_cost": prevalence * ratio, "best_rule": best,
                                 "best_cost": costs[best], **{f"cost_{r}": c for r, c in costs.items()}})
    return pd.DataFrame(rows)


def purity(table):
    out = []
    for (model, stream), block in table.groupby(["model", "stream"]):
        winner = (block.groupby(["prevalence", "cost_ratio", "miss_cost"])["best_rule"]
                  .agg(lambda s: s.value_counts().idxmax()).reset_index().sort_values("miss_cost"))
        winner["decile"] = pd.qcut(winner["miss_cost"].rank(method="first"), 10, labels=False)
        majority = winner.groupby("decile")["best_rule"].transform(lambda s: s.value_counts().idxmax())
        out.append({"model": model, "stream": stream, "pi_r_purity": float((winner["best_rule"] == majority).mean())})
    return pd.DataFrame(out)


def rule_performance(table, chooser, label):
    table = table.copy()
    table["recommended"] = table["miss_cost"].map(chooser)
    table["recommended_cost"] = np.select(
        [table["recommended"] == rule for rule in RULES], [table[f"cost_{rule}"] for rule in RULES])
    table["excess_over_best"] = table["recommended_cost"] / table["best_cost"] - 1.0
    table["agrees"] = table["recommended"] == table["best_rule"]
    return (table.groupby(["model", "stream"])
            .agg(cells=("agrees", "size"), agreement=("agrees", "mean"),
                 mean_excess=("excess_over_best", "mean"),
                 median_excess=("excess_over_best", "median"),
                 p95_excess=("excess_over_best", lambda s: s.quantile(0.95)),
                 max_excess=("excess_over_best", "max"))
            .reset_index().assign(rule=label))


def learned_map(table):
    """Decile-of-pi*r -> majority winning rule, learned on a derivation table."""
    cells = (table.groupby(["miss_cost"])["best_rule"].agg(lambda s: s.value_counts().idxmax())
             .reset_index().sort_values("miss_cost"))
    edges = np.quantile(cells["miss_cost"], np.linspace(0, 1, 11))
    cells["bin"] = np.clip(np.searchsorted(edges, cells["miss_cost"], side="right") - 1, 0, 9)
    mapping = cells.groupby("bin")["best_rule"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    return lambda value: mapping.get(int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, 9)), "fixed_cost_min")


# --------------------------------------------------------------------------
# prevalence and cost-ratio estimation
# --------------------------------------------------------------------------
def em_prevalence(p, train_prior=0.5, iterations=200):
    pi = 0.5
    p = np.clip(p, 1e-6, 1 - 1e-6)
    for _ in range(iterations):
        a = (pi / train_prior) * p
        b = ((1 - pi) / (1 - train_prior)) * (1 - p)
        new = float((a / (a + b)).mean())
        if abs(new - pi) < 1e-9:
            break
        pi = new
    return pi


def prevalence_estimation(store):
    rows = []
    for (model, seed), entry in store.items():
        vy, vp = entry["validation"]
        for stream, (y, p) in entry.items():
            if stream == "validation" or not ((y == 1).any() and (y == 0).any()):
                continue
            pos, neg = p[y == 1], p[y == 0]
            for prevalence in TRUE_PREVALENCE:
                estimates = {}
                for name, t in (("t0.5", 0.5), ("t95R", threshold_for_recall(vy, vp))):
                    q = prevalence * (pos >= t).mean() + (1 - prevalence) * (neg >= t).mean()
                    tpr_v, fpr_v = (vp[vy == 1] >= t).mean(), (vp[vy == 0] >= t).mean()
                    if name == "t0.5":
                        estimates["CC"] = q
                    estimates[f"ACC_{name}"] = float(np.clip((q - fpr_v) / max(1e-9, tpr_v - fpr_v), 0, 1))
                # EM on the expected mixture, using a weighted pseudo-sample
                n_pos = max(1, int(round(prevalence * 20000)))
                rng = np.random.default_rng(seed)
                mixture = np.r_[rng.choice(pos, n_pos), rng.choice(neg, 20000 - n_pos)]
                estimates["EM"] = em_prevalence(mixture)
                correct = cell_costs((vy, vp), (y, p), prevalence, 10.0)["fixed_cost_min"]
                for name, estimate in estimates.items():
                    believed = float(np.clip(estimate, 1e-4, 0.5))
                    thresholds = rule_thresholds((vy, vp), believed, 10.0)
                    cost = realized(y, p, thresholds["fixed_cost_min"], prevalence, 10.0)
                    rows.append({"model": model, "seed": seed, "stream": stream, "true_prevalence": prevalence,
                                 "estimator": name, "estimate": float(estimate),
                                 "ratio_to_truth": float(estimate / prevalence),
                                 "excess_cost_ratio": float(cost / correct) if correct > 0 else np.nan})
    return pd.DataFrame(rows)


def cost_ratio_misspecification(store):
    rows = []
    for (model, seed), entry in store.items():
        if not model.endswith("linked_augmented"):
            continue
        validation = entry["validation"]
        for stream, (y, p) in entry.items():
            if stream == "validation" or not ((y == 1).any() and (y == 0).any()):
                continue
            for prevalence in (0.001, 0.01, 0.05):
                for true_r in (2.0, 5.0, 10.0, 20.0, 50.0):
                    reference = realized(y, p, rule_thresholds(validation, prevalence, true_r)["fixed_cost_min"], prevalence, true_r)
                    for factor in (0.2, 0.5, 2.0, 5.0):
                        believed = true_r * factor
                        cost = realized(y, p, rule_thresholds(validation, prevalence, believed)["fixed_cost_min"], prevalence, true_r)
                        rows.append({"model": model, "seed": seed, "stream": stream, "prevalence": prevalence,
                                     "true_r": true_r, "believed_factor": factor,
                                     "excess_cost_ratio": cost / reference if reference > 0 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def perturbations():
    def make(kind, value, target_only):
        def transform(v, t):
            def f(pair, rng_seed):
                y, p = pair
                z = logit(np.clip(p, 1e-6, 1 - 1e-6))
                if kind == "temperature":
                    z = z / value
                elif kind == "shift":
                    z = z + value
                elif kind == "noise":
                    z = z + np.random.default_rng(rng_seed).normal(0.0, value, size=len(z))
                return y, expit(z)
            return (v if target_only else f(v, 1)), f(t, 2)
        return transform
    specs = []
    for target_only in (False, True):
        for kind, values in (("temperature", (0.5, 2.0, 3.0)), ("shift", (-2.0, -1.0, 1.0, 2.0)), ("noise", (0.5, 1.0, 2.0))):
            if kind == "noise" and not target_only:
                pass
            for value in values:
                specs.append((("target only" if target_only else "validation and target"), kind, value,
                              make(kind, value, target_only)))
    return specs


def main() -> None:
    started = time.time()
    store = build_store()
    store = add_contemporary_stream(store)
    print(f"store: {len(store)} model-partition entries ({time.time() - started:.0f}s)", flush=True)

    # 1-2. estimation
    prevalence = prevalence_estimation(store)
    prevalence.to_csv(RESULTS / "revision_r4_deploy_prevalence_estimation.csv", index=False)
    ratio = cost_ratio_misspecification(store)
    ratio.to_csv(RESULTS / "revision_r4_deploy_cost_ratio_misspecification.csv", index=False)
    print("estimation done", flush=True)

    # 3. held-out validation of the rule
    table = grid_table(store)
    table.to_csv(RESULTS / "revision_r4_deploy_grid.csv", index=False)
    purity(table).to_csv(RESULTS / "revision_r4_deploy_purity.csv", index=False)
    word = table[table["model"] == "word_lr_linked_augmented"]
    seeds = sorted(word["seed"].unique())
    derive, holdout = seeds[: len(seeds) // 2], seeds[len(seeds) // 2:]
    results = [rule_performance(table, three_band, "three-band rule (as stated)")]
    for stream in ("L1 primary", "L2 generator-disjoint"):
        mapping = learned_map(word[(word["seed"].isin(derive)) & (word["stream"] == stream)])
        held = table[table["seed"].isin(holdout) | ~table["model"].eq("word_lr_linked_augmented")]
        results.append(rule_performance(held, mapping,
                                        f"decile map learned on word aug., {stream}, partitions {derive[0]}..{derive[-1]}"))
    for fixed in RULES:
        results.append(rule_performance(table, lambda value, f=fixed: f, f"always {fixed}"))
    pd.concat(results, ignore_index=True).to_csv(RESULTS / "revision_r4_deploy_rule_validation.csv", index=False)
    print("rule validation done", flush=True)

    # 4. score-distribution sensitivity (augmented word model, all partitions)
    word_store = {k: v for k, v in store.items() if k[0] in ("word_lr_linked_augmented", "minilm_lr_linked_augmented")}
    sens, sens_costmin = [], []
    for scope, kind, value, transform in perturbations():
        perturbed = grid_table(word_store, perturb=transform)
        p = purity(perturbed)
        r = rule_performance(perturbed, three_band, "three-band rule (as stated)")
        merged = r.merge(p, on=["model", "stream"])
        merged["scope"], merged["perturbation"], merged["value"] = scope, kind, value
        sens.append(merged)
        c = rule_performance(perturbed, lambda value_, f="fixed_cost_min": f, "always fixed_cost_min")
        c["scope"], c["perturbation"], c["value"] = scope, kind, value
        sens_costmin.append(c)
    baseline = rule_performance(grid_table(word_store), three_band, "three-band rule (as stated)").merge(purity(grid_table(word_store)), on=["model", "stream"])
    baseline["scope"], baseline["perturbation"], baseline["value"] = "none", "none", 0.0
    pd.concat([baseline] + sens, ignore_index=True).to_csv(RESULTS / "revision_r4_deploy_score_sensitivity.csv", index=False)
    baseline_costmin = rule_performance(grid_table(word_store), lambda value_: "fixed_cost_min",
                                        "always fixed_cost_min")
    baseline_costmin["scope"], baseline_costmin["perturbation"], baseline_costmin["value"] = "none", "none", 0.0
    pd.concat([baseline_costmin] + sens_costmin, ignore_index=True).to_csv(
        RESULTS / "revision_r4_deploy_score_sensitivity_costmin.csv", index=False)

    pd.set_option("display.width", 250)
    print(prevalence.groupby(["stream", "estimator", "true_prevalence"])[["ratio_to_truth", "excess_cost_ratio"]]
          .median().round(3).unstack("true_prevalence").to_string())
    print(ratio.groupby(["stream", "believed_factor"])["excess_cost_ratio"].median().round(3).unstack().to_string())
    validation = pd.concat(results, ignore_index=True)
    print(validation.groupby(["rule", "stream"])[["agreement", "mean_excess", "p95_excess"]].mean().round(3).to_string())
    print(pd.concat([baseline] + sens).groupby(["scope", "perturbation", "value", "stream"])[["pi_r_purity", "agreement", "mean_excess"]]
          .mean().round(3).to_string())
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
