# ESWA robustness study

This artifact replaces the earlier design-stage corpus claims with an executed,
leakage-resistant study on public LLM-generated email corpora, evaluated
against human-authored and era-matched real mail.

## Research questions

These are the research questions of the manuscript's Study Design section. They
are posed on the evaluation ladder: L1 linked-rewrite robustness (within the
training pipeline), L2 cross-corpus performance under partially audited
provenance (on the generator-disjoint corpus), L3 human-authored performance
(2000s-era human-authored phishing and real legitimate mail), extended in round
4 by era-matched real phishing, and L4 adaptive-adversary robustness.

- **RQ1.** Does linked-rewrite augmentation improve F1 on source-group-held-out
  canonical emails and their linked rewrites (L1)?
- **RQ2.** Are gains consistent across sparse lexical, frozen semantic, and
  fine-tuned transformer representations?
- **RQ3a.** How does augmentation change cross-corpus performance under
  partially audited provenance (L2), on a corpus published by a different LLM
  pipeline that discloses no source, campaign, or prompt identifier? What can
  an audit of its origin, latent redundancy, proximity, and pipeline-specific
  signal establish, and how much uncertainty does the unverifiable remainder
  add?
- **RQ3b.** How do the detectors perform on human-authored phishing with real
  legitimate mail, and on era-matched real phishing from 2015--2025 (L3:
  human-authored and era-matched performance)? Is any shortfall a property of
  the target corpus or of the detector, and how much target-domain data would
  be needed instead?
- **RQ4.** How does augmentation affect spam false positives, calibration, and
  deployment operating points selected under asymmetric error costs? How
  accurately must a deployment estimate prevalence and cost ratio, and which
  threshold rule is preferable in which region of the prevalence/cost plane?
- **RQ5.** Are gains explained by repeated exposure, or does rewrite diversity
  carry over to a rewrite type held out from training?
- **RQ6.** How much does row-wise source leakage inflate performance, does a
  stricter template-disjoint partition change the estimate, and how sensitive is
  the sparse detector to its strongest phishing-directed words?
- **RQ7.** Do the modest F1 gains justify their computational and calibration
  costs, and under what resource constraints?

RQ1, RQ2, RQ4, RQ5, and the leakage half of RQ6 are answered by the three
top-level run scripts, and their uncertainty across partitions is answered by
`revision/`. RQ3a, RQ3b, RQ7, the template-disjoint half of RQ6, and the
bounded L4 result are answered by `revision/`.

## Data

Training and in-pipeline evaluation (downloaded by `download_data.py`):

- `simlab-vs/PhishFuzzer_emails_entity_rephrased_v1`, revision
  `839d0d927d3e6a39e2b007a3c71b8fc414213f2e`, 19,800 rows and 3,300
  `Original_ID` groups. The upstream repository does not currently declare a
  software/data license, so this artifact downloads but does not redistribute it.
- `Dizzzy0x00/LLMGen-Phishing-Email-Dataset`, revision
  `d717e5457b71cb44dd038dafe6a81cea5e0badab`, English OpenAI-generated split,
  Apache-2.0. This is the generator-disjoint (L2) corpus.

Evaluation only (downloaded by the scripts in `revision/`; never used for
training, feature selection, calibration, or threshold selection):

- `zefang-liu/phishing-email-dataset` (LGPL-3.0): the human-authored L3
  corpus, via `revision/fetch_realworld.sh`.
- Apache SpamAssassin public corpus, 2002--2004: real ham and spam, via
  `revision/fetch_realworld.sh`.
- J. Nazario, *The Online Phishing Corpus*, yearly files 2015--2025
  (CC-BY-4.0, attribution to J. Nazario): era-matched real phishing, added in
  round 4, via `revision/fetch_era_matched.sh`, which pins a SHA-256 per file.
  Years 2015--2022 are treated as human-authored. The authorship of 2023--2025
  cannot be verified, so those years are reported separately.

None of the five corpora is redistributed. The data directories ship with a
README each and nothing else. All links, email addresses, and IP addresses are
replaced with inert placeholders before modeling. No file in `results/`
carries message text, raw or neutralized; the vocabulary and topic tables of
round 4 list individual n-grams only.

## Verify and run

Verify the bundled results and figures without installing dependencies or
downloading corpora:

```bash
python3.11 verify_artifact.py --results-only
```

Run the following from this study directory to reproduce the main experiment:

```bash
python3.11 -m pip install -r requirements.txt
python3.11 download_data.py
python3.11 run_experiment.py
python3.11 run_rigor_experiments.py
python3.11 run_transformer.py
python3.11 make_figures.py
python3.11 make_manifest.py
python3.11 verify_artifact.py --main-only
```

After dependencies are installed, `./run_all.sh` performs the complete workflow
for the main experiment.

The main experiment uses five group-stratified train/validation/test splits for
the lightweight models and three for DistilBERT. Every variant of an
`Original_ID` remains in one partition. The principal uncertainty analysis
resamples source groups, not individual variants.

The revision analyses refit the same protocol on further, independently drawn
source-grouped partitions:

- **100** for the paired lightweight families (word TF-IDF and MiniLM): 20
  from round 3 and 80 from round 4.
- **20** for the augmented character+word model.
- **20** for DistilBERT: the 3 published, 5 from round 3 and 12 from round 4.

The 20 DistilBERT partitions are the same 20 partitions as the first 20
lightweight ones.

Outputs are written to `results/`:

- `dataset_stats.json`
- `metrics_by_seed.csv`
- `summary.csv`
- `paired_group_bootstrap.json`
- `variant_metrics_by_seed.csv` and `variant_summary.csv`
- `split_assignments.csv`
- `predictions_seed42.csv`
- `error_analysis_seed42.csv`
- `feature_audit_seed42.csv`
- `runtime_costs.csv`
- `run_info.json`
- `transformer_metrics_by_seed.csv`
- `transformer_summary.csv`
- `transformer_variant_metrics_by_seed.csv`
- `transformer_variant_summary.csv`
- `transformer_predictions_seed42.csv`
- `transformer_paired_group_bootstrap.json`
- `transformer_runtime_costs.csv`
- `transformer_run_info.json`
- `rigor_dose_metrics_by_seed.csv` and `rigor_dose_summary.csv`
- `rigor_leave_one_out_metrics_by_seed.csv` and
  `rigor_leave_one_out_summary.csv`
- `rigor_leakage_metrics_by_seed.csv` and `rigor_leakage_summary.csv`
- `rigor_cue_mask_metrics_by_seed.csv` and `rigor_cue_mask_summary.csv`
- `rigor_cue_features.csv`
- `rigor_runtime_costs.csv`
- `rigor_run_info.json`
- `manifest.json`

Prediction and probability files contain labels, source-group IDs, variant
metadata, and probabilities only. They deliberately exclude raw or neutralized
message text. `verify_artifact.py` checks this for the main prediction files
and for every round-4 probability file.

## Revision analyses (review rounds 2 to 4)

`revision/` contains one script per reviewer request, together with its own
README. That README maps every script to the comment it answers and gives the
run order, the dependencies between scripts, and runtime notes. Outputs are
written to `results/` with a `revision_`, `revision_r3_` or `revision_r4_`
prefix and never overwrite a published result. Run
`revision/fetch_realworld.sh` and `revision/fetch_era_matched.sh` first, then
follow `revision/README.md`.

**Round 2** added the following:

- a human-authored (non-generated) evaluation corpus;
- a latent-provenance audit for corpora that publish no source identifiers;
- a template-disjoint splitting ablation;
- hierarchical bootstrap inference across training splits;
- DistilBERT learning-curve and sequence-length diagnostics;
- a cost-sensitive threshold framework.

Round 2 also adds domain adaptation and aggregate L4 robustness results:

- `rev_i_domain_adaptation.py` decomposes the human-authored shortfall into a
  corpus ceiling, a cross-domain gap, and a directional asymmetry, and measures
  how much target-domain data would be needed instead.
- The L4 robustness analysis is available as aggregate tables and figures.

**Round 3** added the following:

- 20 refitted lightweight partitions and 5 further DistilBERT partitions, with
  multi-partition paired inference (`rev_k_*`, `rev_l_r3_inference.py`);
- an origin-separability probe, proximity strata, redundancy sensitivity, and
  the provenance ledger (`rev_m_provenance.py`);
- DistilBERT temperature, ranking, and truncation tests
  (`rev_n_distilbert_mechanism.py`);
- deployment operating regions and break-even analyses
  (`rev_o_deployment.py`).

**Round 4** added the following:

- 100 lightweight and 20 DistilBERT partitions, with corrected resampled
  t-tests, random-effects heterogeneity, and a stabilization analysis of the
  interval width (`rev_p_distilbert_multisplit20.py`, `rev_q_multisplit100.py`,
  `rev_r_r4_inference.py`);
- a pipeline-signature test: vocabulary neutralization, linear concept
  erasure, and domain-adversarial training (`rev_s_pipeline_signature.py`);
- an error analysis of human-authored misses with topics, coverage, partition
  sensitivity, and the k-shot comparison with uncertainty
  (`rev_t_human_error_analysis.py`), and a topic model of the L3 phishing
  class on its own (`rev_t2_l3_internal_topics.py`);
- the finding that the public L3 aggregate contains the SpamAssassin corpus.
  `rev_t3_l3_sa_overlap.py` measures the overlap and re-scores L3 without the
  duplicates. `rev_c2_dedup_costs.py` recomputes the round-3 human-authored
  cost tables without them, beside the round-3 files rather than over them;
- era-matched real phishing and a header-format control (`rev_era_data.py`,
  `rev_u_era_format.py`, `rev_u2_era_format_distilbert.py`);
- recall as a function of proximity to the training data
  (`rev_u3_proximity.py`);
- a second transformer backbone for the 160- versus 320-token window
  (`rev_v_backbone_window.py`);
- deployment robustness: label-free prevalence estimation, cost-ratio
  misestimation, held-out rule validation, and score-distribution sensitivity
  (`rev_w_deployment_robustness.py`), and the deployment-unit analysis over
  the larger partition sets (`rev_x_operational_partitions.py`);
- the corpus facts that the provenance ledger states
  (`rev_z_corpus_facts.py`);
- fit and inference timing for every family, measured in one session
  (`rev_y_timing.py`).

## Figures

Publication-ready PDF and PNG visualizations are written to `figures/`. Each
name below exists in both formats; the manuscript uses the PNGs. Two file names
predate the round-4 terminology: `transfer_ladder` and the hand-drawn
`rewriteArgumentationTransferFramework.png`. They are kept so that earlier
references still resolve.

Generated by `make_figures.py`:

- `augmentation_dose`
- `external_spam_tradeoff`
- `leakage_and_masking`

Generated by `revision/make_revision_figures.py`:

- `threshold_cost`
- `distilbert_calibration`
- `distilbert_learning_curve`

The round-2 single-panel ladder is retired. Its function remains as a record
and writes `transfer_ladder_r2` if called.

Generated by `revision/make_limitation_figures.py`:

- `domain_adaptation`
- `adaptive_evasion`

Generated by `revision/make_r3_figures.py`:

- `multisplit_inference_r3`, the round-3 version, kept for the record

Generated by `revision/make_r4_revised_figures.py`, from the full sets of
refitted partitions:

- `multisplit_inference`
- `tradeoff`
- `transfer_ladder`

Generated by `revision/make_r4_new_figures.py`:

- `pipeline_signature`
- `l3_explained`
- `era_matched`
- `deployment_robustness`

The older figure scripts no longer write any name that a round-4 script
writes, so re-running them cannot overwrite a round-4 figure.

The framework overview (`rewriteArgumentationTransferFramework.png`) is drawn
by hand rather than generated, and ships with the manuscript sources.
