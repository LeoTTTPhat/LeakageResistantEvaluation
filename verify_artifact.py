#!/usr/bin/env python3
"""Fail fast when the reproduction artifact is incomplete or internally invalid.

The main-experiment checks (corpus checksums, expected outputs and figures,
group isolation, metric ranges, the leakage control, no message text in the
prediction files, and the result manifest) run first and are unchanged.

Round 4 adds two checks, skipped with ``--main-only``:

* every result file written by a finished round-4 script (``R4_RESULTS``)
  exists and contains data;
* every ``revision_r4_*probabilit*.csv`` carries no text-like column and only
  probabilities in [0, 1] -- a release condition, like the text check on the
  main prediction files.

``--results-only`` checks the shipped outputs without downloading corpora.
``--main-only`` verifies the main experiment before running the revision scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math

from collections import Counter, defaultdict
from pathlib import Path

from download_data import DATA, FILES, sha256
from make_manifest import RESULTS


EXPECTED_RESULTS = [
    "dataset_stats.json",
    "metrics_by_seed.csv",
    "summary.csv",
    "variant_metrics_by_seed.csv",
    "variant_summary.csv",
    "split_assignments.csv",
    "predictions_seed42.csv",
    "error_analysis_seed42.csv",
    "feature_audit_seed42.csv",
    "runtime_costs.csv",
    "paired_group_bootstrap.json",
    "transformer_metrics_by_seed.csv",
    "transformer_summary.csv",
    "transformer_variant_metrics_by_seed.csv",
    "transformer_variant_summary.csv",
    "transformer_predictions_seed42.csv",
    "transformer_runtime_costs.csv",
    "transformer_paired_group_bootstrap.json",
    "rigor_dose_metrics_by_seed.csv",
    "rigor_dose_summary.csv",
    "rigor_leave_one_out_metrics_by_seed.csv",
    "rigor_leave_one_out_summary.csv",
    "rigor_leakage_metrics_by_seed.csv",
    "rigor_leakage_summary.csv",
    "rigor_cue_mask_metrics_by_seed.csv",
    "rigor_cue_mask_summary.csv",
    "rigor_cue_features.csv",
    "rigor_runtime_costs.csv",
    "rigor_run_info.json",
    "manifest.json",
]

EXPECTED_FIGURES = [
    "augmentation_dose.pdf",
    "augmentation_dose.png",
    "external_spam_tradeoff.pdf",
    "external_spam_tradeoff.png",
    "leakage_and_masking.pdf",
    "leakage_and_masking.png",
]

# --------------------------------------------------------------------------
# Round 4. One entry per script in revision/, listing the files it writes to
# results/. Only finished scripts belong in R4_RESULTS; a script that is still
# running goes in R4_PENDING, which is reported but not enforced. When it
# finishes, move its entry up.
# --------------------------------------------------------------------------
R4_RESULTS: dict[str, list[str]] = {
    "rev_p_distilbert_multisplit20.py": [
        "revision_r4_distilbert_multisplit_metrics.csv",
        "revision_r4_distilbert_multisplit_probabilities.csv",
        "revision_r4_distilbert_multisplit_run_info.json",
    ],
    "rev_q_multisplit100.py": [
        "revision_r4_multisplit100_metrics.csv",
        "revision_r4_multisplit100_probabilities.csv",
    ],
    "rev_r_r4_inference.py": [
        "revision_r4_inference.csv",
        "revision_r4_heterogeneity.json",
        "revision_r4_stabilization.csv",
        "revision_r4_stabilization_tests.json",
        "revision_r4_partition_effects.csv",
        "revision_r4_partition_effect_summary.csv",
    ],
    "rev_s_pipeline_signature.py": [
        "revision_r4_pipeline_vocab.csv",
        "revision_r4_pipeline_erasure.csv",
        "revision_r4_pipeline_dann.csv",
        "revision_r4_pipeline_terms.csv",
        "revision_r4_pipeline_summary.csv",
    ],
    "rev_t_human_error_analysis.py": [
        "revision_r4_l3_corpus_profile.csv",
        "revision_r4_l3_error_profile.csv",
        "revision_r4_l3_vocab_human.csv",
        "revision_r4_l3_vocab_synthetic.csv",
        "revision_r4_l3_topics.csv",
        "revision_r4_l3_coverage.csv",
        "revision_r4_l3_partition_sensitivity.csv",
        "revision_r4_l3_kshot_draws.csv",
        "revision_r4_l3_kshot_summary.csv",
        "revision_r4_l3_summary.json",
    ],
    "rev_t2_l3_internal_topics.py": [
        "revision_r4_l3_internal_topics.csv",
        "revision_r4_l3_cue_families.csv",
    ],
    "rev_t3_l3_sa_overlap.py": [
        "revision_r4_l3_sa_overlap.json",
        "revision_r4_l3_sa_duplicates.csv",
        "revision_r4_l3_dedup_metrics.csv",
        "revision_r4_l3_dedup_ceiling.csv",
    ],
    "rev_c2_dedup_costs.py": [
        "revision_r4_threshold_cost_dedup_summary.csv",
        "revision_r4_regions_dedup.csv",
        "revision_r4_miss_cost_boundaries_dedup.csv",
        "revision_r4_misspec_dedup.csv",
        "revision_r4_operational_dedup.csv",
        "revision_r4_break_even_dedup.csv",
        "revision_r4_stream_ceiling_dedup.csv",
        "revision_r4_threshold_cost_curves_dedup.csv",
    ],
    "rev_u_era_format.py": [
        "revision_r4_era_format_metrics.csv",
        "revision_r4_era_by_year.csv",
        "revision_r4_era_format_summary.csv",
        "revision_r4_era_corpus.json",
    ],
    "rev_u3_proximity.py": [
        "revision_r4_proximity_recall.csv",
        "revision_r4_proximity_summary.csv",
    ],
    "rev_w_deployment_robustness.py": [
        "revision_r4_deploy_prevalence_estimation.csv",
        "revision_r4_deploy_cost_ratio_misspecification.csv",
        "revision_r4_deploy_grid.csv",
        "revision_r4_deploy_purity.csv",
        "revision_r4_deploy_rule_validation.csv",
        "revision_r4_deploy_score_sensitivity.csv",
        "revision_r4_deploy_score_sensitivity_costmin.csv",
    ],
    "rev_x_operational_partitions.py": [
        "revision_r4_operational_partitions.csv",
        "revision_r4_break_even_partitions.csv",
    ],
    "rev_z_corpus_facts.py": [
        "revision_r4_corpus_facts.json",
    ],
    "rev_u2_era_format_distilbert.py": [
        "revision_r4_era_format_distilbert_metrics.csv",
        "revision_r4_era_format_distilbert_by_year.csv",
        "revision_r4_era_format_distilbert_run_info.json",
    ],
    "rev_v_backbone_window.py": [
        "revision_r4_backbone_learning_curve.csv",
        "revision_r4_backbone_probabilities.csv",
        "revision_r4_backbone_truncation_audit.csv",
        "revision_r4_backbone_run_info.json",
    ],
    "rev_y_timing.py": [
        "revision_r4_timing.csv",
        "revision_r4_timing_run_info.json",
    ],
    "rev_v2_transformer_summary.py": [
        "revision_r4_transformer_summary.csv",
        "revision_r4_backbone_summary.csv",
    ],
}

R4_PENDING: dict[str, list[str]] = {}

R4_PROBABILITY_GLOB = "revision_r4_*probabilit*.csv"
TEXT_COLUMNS = {
    "text", "text_full", "text_body", "raw_text", "body", "subject",
    "message", "email", "content",
}


def has_data(path: Path) -> bool:
    """A CSV with at least one data row, or a JSON document that is not empty."""
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8", errors="replace") as stream:
            reader = csv.reader(stream)
            return next(reader, None) is not None and next(reader, None) is not None
    if path.suffix == ".json":
        return bool(json.loads(path.read_text()))
    return path.stat().st_size > 0


def check_probability_file(path: Path, *, include_unqualified: bool = False) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        header = rows.fieldnames or []
        text = sorted(column for column in header if column.strip().lower() in TEXT_COLUMNS)
        assert not text, f"{path.name}: text column(s) {text}; probability files carry no message text"
        columns = [column for column in header if column.startswith("probability_")]
        if include_unqualified and "probability" in header:
            columns.append("probability")
        assert columns, f"{path.name}: no probability column"
        for row in rows:
            assert all(0.0 <= float(row[column]) <= 1.0 for column in columns), (
                f"{path.name}:{rows.line_num}: probability outside [0, 1]"
            )


def check_round4() -> None:
    for script, filenames in R4_RESULTS.items():
        for filename in filenames:
            path = RESULTS / filename
            assert path.exists(), (
                f"missing round-4 result: {filename} (written by revision/{script}; "
                "run it, or pass --main-only to verify the main experiment alone)"
            )
            assert path.stat().st_size > 0, f"empty round-4 result: {filename}"
            assert has_data(path), f"round-4 result has no data: {filename}"

    expected = {name for names in R4_RESULTS.values() for name in names if "probabilit" in name}
    probability_files = sorted(RESULTS.glob(R4_PROBABILITY_GLOB))
    assert expected <= {path.name for path in probability_files}
    for path in probability_files:
        check_probability_file(path, include_unqualified=True)

    present = sum(len(names) for names in R4_RESULTS.values())
    print(
        f"Round 4: {present} result files from {len(R4_RESULTS)} scripts present; "
        f"{len(probability_files)} probability files carry no text column."
    )
    for script, filenames in R4_PENDING.items():
        found = sum((RESULTS / name).exists() for name in filenames)
        print(f"  pending, not enforced: {script} ({found}/{len(filenames)} files present)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="skip the round-4 checks (for a from-scratch rerun before revision/)",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="verify shipped results and figures without requiring the downloaded corpora",
    )
    args = parser.parse_args(argv)

    if not args.results_only:
        for filename, (_, expected_checksum) in FILES.items():
            path = DATA / filename
            assert path.exists(), f"missing dataset: {path}; run download_data.py first"
            assert sha256(path) == expected_checksum, f"checksum mismatch: {path}"

    for filename in EXPECTED_RESULTS:
        assert (RESULTS / filename).exists(), f"missing result: {filename}"

    figure_dir = RESULTS.parent / "figures"
    for filename in EXPECTED_FIGURES:
        path = figure_dir / filename
        assert path.exists() and path.stat().st_size > 1000, (
            f"missing or empty figure: {filename}"
        )

    groups_by_seed = defaultdict(set)
    partitions = set()
    with (RESULTS / "split_assignments.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups = groups_by_seed[row["seed"]]
            group = row["group_id"]
            assert group and group not in groups, f"duplicate or empty group for seed {row['seed']}: {group}"
            groups.add(group)
            partitions.add(row["partition"])
    assert partitions == {"train", "validation", "test"}
    counts = {seed: len(groups) for seed, groups in groups_by_seed.items()}
    assert all(count == 2156 for count in counts.values()), counts

    for filename in [
        "metrics_by_seed.csv",
        "transformer_metrics_by_seed.csv",
        "rigor_dose_metrics_by_seed.csv",
        "rigor_leave_one_out_metrics_by_seed.csv",
        "rigor_leakage_metrics_by_seed.csv",
        "rigor_cue_mask_metrics_by_seed.csv",
    ]:
        with (RESULTS / filename).open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            for row in rows:
                for column in ("precision", "recall", "f1", "brier", "ece"):
                    cell = row[column].strip()
                    if cell:
                        value = float(cell)
                        assert math.isnan(value) or 0.0 <= value <= 1.0, (
                            f"{filename}:{rows.line_num}: invalid {column}"
                        )

    with (RESULTS / "rigor_leakage_metrics_by_seed.csv").open(newline="", encoding="utf-8") as stream:
        strategies = set()
        for row in csv.DictReader(stream):
            strategy = row["split_strategy"]
            strategies.add(strategy)
            overlap = float(row["test_group_overlap_rate"])
            if strategy == "source_group":
                assert overlap == 0.0, "source groups overlap under grouped splitting"
            elif strategy == "row_stratified":
                assert overlap > 0.95, "expected source group overlap in the row-wise control"
    assert {"source_group", "row_stratified"} <= strategies

    with (RESULTS / "rigor_cue_features.csv").open(newline="", encoding="utf-8") as stream:
        cue_counts = Counter(row["seed"] for row in csv.DictReader(stream))
    assert cue_counts and all(count == 20 for count in cue_counts.values()), cue_counts

    for filename in ["predictions_seed42.csv", "transformer_predictions_seed42.csv"]:
        check_probability_file(RESULTS / filename)

    stats = json.loads((RESULTS / "dataset_stats.json").read_text())
    assert stats["binary_groups"] == 2156
    assert stats["external_duplicate_texts"] == 0

    manifest = json.loads((RESULTS / "manifest.json").read_text())
    manifested = {entry["file"] for entry in manifest["files"]}
    assert set(EXPECTED_RESULTS) - {"manifest.json"} <= manifested
    for entry in manifest["files"]:
        path = RESULTS / entry["file"]
        assert path.stat().st_size == entry["bytes"]
        assert sha256(path) == entry["sha256"], f"result checksum mismatch: {path}"

    if args.main_only:
        print("Round-4 checks skipped (--main-only).")
    else:
        check_round4()
    if args.results_only:
        print("Corpus checksum checks skipped (--results-only).")
    print("Artifact verification passed.")


if __name__ == "__main__":
    main()
