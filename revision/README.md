# Revision analyses (review rounds 2, 3 and 4)

Each script answers one reviewer request, or closes one limitation that a
revision raised on itself and had acknowledged without measuring. Each writes
machine-readable output to `../results/`. Nothing here changes the original
experiment; the published tables are reproduced unchanged by
`../run_experiment.py`, `../run_rigor_experiments.py`, and
`../run_transformer.py`. Outputs carry the prefix `revision_` (round 2),
`revision_r3_` (round 3) or `revision_r4_` (round 4), and no script overwrites
an earlier round's file.

## Requirements

The scripts reuse the loaders, splitters, and estimators of
`../run_experiment.py`. `sentence-transformers` is optional. If it is absent,
the pinned MiniLM cache in `../cache/` is used for the study corpora, and the
revision corpora are embedded with an equivalent mean-pooling implementation
(`rev_common.encode_minilm`, verified against the cache to 3e-7).

Six scripts fine-tune transformers: `rev_e_distilbert.py`,
`rev_k_distilbert_multisplit.py`, `rev_p_distilbert_multisplit20.py`,
`rev_u2_era_format_distilbert.py`, `rev_v_backbone_window.py` and
`rev_y_timing.py`. They use an
Apple-silicon MPS device when one is available and fall back to the CPU
otherwise. There is no CUDA code path, so on a CUDA machine they run on the CPU
unless the `torch.device(...)` line is changed. We ran them with
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Data

```bash
python3 ../download_data.py     # PhishFuzzer + generator-disjoint corpus (pinned)
./fetch_realworld.sh            # human-authored corpora for the L3 evaluation (pinned, SHA-256 verified)
./fetch_era_matched.sh          # era-matched real phishing, Nazario 2015-2025 (round 4)
```

The human-authored corpora are the aggregated public phishing/legitimate corpus
`zefang-liu/phishing-email-dataset` (LGPL-3.0) and the Apache SpamAssassin
public corpus (2002--2004). Neither involves a language model, so their
provenance is verifiable in the sense the L3 evaluation requires. The aggregate
is pinned to dataset commit `34085a03` (2024-01-17), and `fetch_realworld.sh`
checks the SHA-256 of the aggregate and of every SpamAssassin folder against
the copies the study used. The aggregate itself contains the SpamAssassin
corpus; `rev_t3_l3_sa_overlap.py` identifies the duplicated rows.

Round 4 adds J. Nazario's *Online Phishing Corpus* (CC-BY-4.0), which is
hand-classified real phishing from 2015 to 2025. Years 2015--2022 are treated
as human-authored; the authorship of 2023--2025 cannot be verified, so those
years are reported separately. `fetch_era_matched.sh` pins a SHA-256 for each
yearly file and fails on any mismatch.

None of these corpora is redistributed here. None is used for training,
feature selection, calibration, or threshold selection. See `data/README.md`
and `data/nazario_README.md`.

## Run order

```bash
python3 rev_a_probabilities.py        # per-split probabilities, all conditions
python3 rev_f_realworld.py            # L3 human-authored evaluation      (R1.1)
python3 rev_e_distilbert.py           # curves, calibration, seq. length  (R1.4)
python3 rev_g_distilbert_streams.py   # DistilBERT on every stream
python3 rev_b_inference.py            # hierarchical bootstrap            (R2.2)
python3 rev_c_thresholds.py           # cost-sensitive thresholds         (R1.5)
python3 rev_d_similarity.py           # similarity + provenance audit (R1.2/R2.1)
python3 rev_h_clustersplit.py         # template-disjoint splitting       (R1.2)
python3 rev_i_domain_adaptation.py    # L3 ceiling, asymmetry, exchange rate (L-1)
python3 make_revision_figures.py
python3 make_limitation_figures.py
```

Round 3 adds five scripts. They depend on the round-2 outputs above, so run them
afterwards, in this order:

```bash
python3 rev_k_multisplit.py             # 20 refitted grouped partitions      (R3.2)
python3 rev_k_distilbert_multisplit.py  # 5 further DistilBERT partitions     (R3.2)
python3 rev_l_r3_inference.py           # multi-partition paired inference    (R3.2)
python3 rev_m_provenance.py             # origin probe, proximity, redundancy (R3.1)
python3 rev_n_distilbert_mechanism.py   # temperature, ranking, truncation    (R3.4)
python3 rev_o_deployment.py             # operating regions, break-even     (R3.3/5)
python3 make_r3_figures.py
```

`rev_l_r3_inference.py` consumes both `rev_k` scripts and, for the three
published DistilBERT partitions, `rev_e_distilbert.py`'s probability table.
`rev_m_provenance.py` and `rev_o_deployment.py` consume the `rev_a` and
`rev_f`/`rev_g` outputs. `rev_n_distilbert_mechanism.py` consumes `rev_e` and
`rev_g`.

`rev_k_distilbert_multisplit.py` fine-tunes ten models and is the only
expensive round-3 script: roughly 1,000 s per partition on an Apple-silicon MPS
device, about 85 minutes in total. Everything else in the round-3 set runs in
minutes.

`rev_a` must run first; `rev_b`, `rev_c`, and `rev_d` consume its probability
table. `rev_d` uses `rev_f`'s output for the human-authored audit, and `rev_c`
uses `rev_g`'s output to include the transformer.

## Which script answers which comment (rounds 2 and 3)

| Comment | Script | Outputs |
| --- | --- | --- |
| R1.1 real-attack evidence | `rev_f_realworld.py` | `revision_realworld_{probabilities,metrics_by_seed,summary}.csv` |
| R1.2 variant similarity | `rev_d_similarity.py` | `revision_variant_{similarity,exchangeability}.csv`, `revision_variant_similarity_summary.json` |
| R1.2 partition strictness | `rev_h_clustersplit.py` | `revision_template_split_{metrics,summary}.csv`, `revision_template_cluster_structure.json` |
| R1.3 cost--benefit | `rev_c_thresholds.py` + runtime files | `revision_threshold_cost_summary.csv` |
| R1.4 DistilBERT diagnostics | `rev_e_distilbert.py` | `revision_distilbert_{learning_curve,probabilities,truncation_audit}.csv` |
| R1.5 threshold framework | `rev_c_thresholds.py` | `revision_threshold_cost_{by_seed,summary,curves}.csv`, `revision_stream_ceiling_summary.csv` |
| R2.1 provenance audit | `rev_d_similarity.py` | `revision_corpus_provenance_audit.csv`, `revision_cluster_aware_intervals.csv` |
| R2.2 multi-split inference | `rev_b_inference.py` | `revision_multisplit_inference.csv`, `revision_variance_decomposition.json` |
| R3.1 provenance evidence and claim scope | `rev_m_provenance.py` | `revision_r3_{pipeline_separability,proximity_strata,redundancy_sensitivity,provenance_ledger}.csv` |
| R3.2 partitions and refits | `rev_k_multisplit.py`, `rev_k_distilbert_multisplit.py` | `revision_r3_multisplit_{metrics,probabilities}.csv`, `revision_r3_distilbert_multisplit_{metrics,probabilities}.csv`, `..._run_info.json` |
| R3.2 inference across partitions | `rev_l_r3_inference.py` | `revision_r3_inference.csv`, `revision_r3_variance_decomposition.json`, `revision_r3_split_convergence.csv` |
| R3.3 practical interpretation | `rev_o_deployment.py` | `revision_r3_operational_value.csv`, `revision_r3_break_even.csv` |
| R3.4 DistilBERT explanations | `rev_n_distilbert_mechanism.py` | `revision_r3_distilbert_{temperature,truncation_strata,hypotheses}.csv` |
| R3.5 deployment thresholds | `rev_o_deployment.py` | `revision_r3_operating_regions.csv`, `revision_r3_miss_cost_boundaries.csv`, `revision_r3_prevalence_misspecification.csv` |

## Round 4

Round 4 adds the following:

- seventeen analysis scripts: `rev_p` to `rev_z`, plus `rev_t2`, `rev_t3`,
  `rev_c2` and `rev_v2` (which recomputes the transformer section's derived
  numbers from stored outputs);
- a corpus loader, `rev_era_data.py`;
- a downloader, `fetch_era_matched.sh`;
- two figure scripts, `make_r4_revised_figures.py` and
  `make_r4_new_figures.py`;
- five TikZ generators, `tikz_inference.py`, `tikz_ladder_era.py`, `tikz_new.py`,
  `tikz_limitations.py` and `tikz_revision.py`. The manuscript's figures are the
  pgfplots files they write to `figures/tikz/` (each a single `tikzpicture`
  with its data inline). They reuse the matplotlib scripts' data code, so the
  plotted numbers are the same as in the PNG/PDF figures, which remain for
  reference; `--copy-to DIR` also writes the files into the submission bundle.

The comment labels in this section refer to the round-4 review: R1.1 to R1.11
are Reviewer 1's numbered comments, and R2 is Reviewer 2's report. Several of
these scripts consume round-2 and round-3 outputs (see Dependencies), so run
the round-4 set after the two sets above.

### Run order

```bash
./fetch_era_matched.sh                    # Nazario 2015-2025, SHA-256 pinned           (R1.9)

python3 rev_p_distilbert_multisplit20.py  # 12 more DistilBERT partitions, 20 in all    (R1.2, R2)  GPU
python3 rev_q_multisplit100.py            # 80 more lightweight partitions, 100 in all  (R1.2)      parallel
python3 rev_r_r4_inference.py             # inference and stabilization, 100 / 20       (R1.2, R2)
python3 rev_s_pipeline_signature.py       # vocabulary neutralization, erasure, DANN    (R1.1, R2)  parallel
python3 rev_u3_proximity.py               # recall by proximity to the training data    (R1.3, R1.9, R2)
python3 rev_t_human_error_analysis.py     # L3 error analysis, topics, coverage, k-shot (R1.3, R1.9, R1.10)
python3 rev_t2_l3_internal_topics.py      # what the L3 phishing class itself contains  (R1.3, R1.9)
python3 rev_t3_l3_sa_overlap.py           # L3 aggregate contains SpamAssassin; dedup   (R1.3, R1.9)
python3 rev_c2_dedup_costs.py             # round-3 L3 cost tables without duplicates   (R1.5)
python3 rev_u_era_format.py               # era-matched recall, header-format control   (R1.9, R2)  parallel
python3 rev_u2_era_format_distilbert.py   # the same for DistilBERT                     (R1.9, R2)  GPU
python3 rev_v_backbone_window.py          # roberta-base, 160 vs 320 tokens             (R1.4)      GPU
python3 rev_v2_transformer_summary.py     # derived transformer numbers, stored outputs (R1.4, R2)
python3 rev_w_deployment_robustness.py    # prevalence estimation, rule validation      (R1.5)
python3 rev_x_operational_partitions.py   # deployment units over 15 / 20 partitions    (R1.5)
python3 rev_z_corpus_facts.py             # corpus facts the provenance ledger quotes   (R1.9, R1.11)
python3 make_r4_revised_figures.py        # multisplit_inference, tradeoff, evaluation_ladder (R1.7)
python3 make_r4_new_figures.py            # pipeline_signature, l3_explained, era_matched,
                                          # deployment_robustness
python3 tikz_inference.py                # TikZ/pgfplots figures for the manuscript:  (R1.7)
python3 tikz_ladder_era.py               #   multisplit, tradeoff, evaluation ladder, era-matched,
python3 tikz_new.py                      #   pipeline signature, L3 explained, deployment robustness,
python3 tikz_limitations.py              #   domain adaptation, adaptive evasion, dose, leakage,
python3 tikz_revision.py                 #   threshold cost, DistilBERT calibration and learning curve
python3 rev_y_timing.py                   # every family timed in one session           GPU
                                          # run it last, alone, on an otherwise idle machine
```

`fetch_era_matched.sh` runs first because `rev_u3`, `rev_t`, `rev_w` and
`rev_z` read the Nazario corpus too, not only `rev_u` and `rev_u2`.
`rev_era_data.py` is the loader those six scripts share. It writes nothing; `python3 rev_era_data.py`
prints the cleaned corpus composition, and so does the last step of
`fetch_era_matched.sh`. The loader skips an absent yearly file instead of
failing, so let the fetch script finish (it verifies all eleven files) before
running anything that reads the corpus.

### Dependencies

* **Common inputs.**
  - `rev_r`, `rev_c2` and `rev_x` work from saved tables only. `rev_r`
    imports `rev_p`'s seed list, so `torch` must still be importable.
  - Every other round-4 script reads the two primary corpora
    (`../download_data.py`), and most of them draw partitions with the
    study's own splitter.
  - `rev_q`, `rev_s`, `rev_t`, `rev_u`, `rev_u2`, `rev_u3`, `rev_w` and `rev_y`
    also load the pinned MiniLM cache that `../run_experiment.py` writes to
    `../cache/`. New texts (the human-authored and Nazario corpora, body-only
    views) are embedded by `rev_common.encode_minilm` and cached on first use;
    `rev_t3` uses only that route.
  - `rev_p`, `rev_t`, `rev_t2`, `rev_t3`, `rev_u`, `rev_u2`, `rev_u3`, `rev_w`
    and `rev_z` also need the L3 corpora from `fetch_realworld.sh`.
* **`rev_p` and `rev_q` feed `rev_r`.** `rev_r_r4_inference.py` combines
  `rev_q`'s 80 partitions with the 20 of `rev_k_multisplit.py`. It combines
  `rev_p`'s 12 DistilBERT partitions with the 5 of
  `rev_k_distilbert_multisplit.py` and the 3 published ones, which it takes
  from the round-2 tables of `rev_a`, `rev_e` and `rev_g`. It also reuses
  helpers from `rev_l_r3_inference.py`. The 20 DistilBERT partitions are
  exactly the 20 partitions of `rev_k_multisplit.py`, so all three families
  are compared on identical splits.
* **Self-contained refits.** `rev_s` and `rev_u` refit their own models on the
  20 partitions of `rev_k_multisplit.py`, importing only its seed list. `rev_s`
  also reuses the component routine of `rev_m_provenance.py`. `rev_t`,
  `rev_t2`, `rev_t3`, `rev_u3` and `rev_u2` use the five published partitions
  (seeds 17, 29, 42, 71, 113). `rev_v` uses the three published DistilBERT
  partitions (29, 42, 71) and reuses the encoding and evaluation helpers of
  `rev_e_distilbert.py`.
* **`rev_u2` imports `rev_u`'s input builder.** It depends on the module, not
  on `rev_u`'s outputs, but running `rev_u` first fills the MiniLM cache for
  the body-only views.
* **`rev_t3` runs after `rev_p`.** It reads the saved L3 probabilities of every
  detector: `revision_realworld_probabilities.csv` (`rev_f`),
  `revision_distilbert_stream_probabilities.csv` (`rev_g`) and
  `revision_r4_distilbert_multisplit_probabilities.csv` (`rev_p`). It reuses
  the component split of `rev_i_domain_adaptation.py` and `rev_u`'s
  SpamAssassin body-only loader.
* **`rev_c2` and `rev_w` run after `rev_t3`.** Both read its duplicate list,
  `revision_r4_l3_sa_duplicates.csv`, and drop those L3 rows from the
  human-authored stream. `rev_c2` also reads the round-2 probability tables of
  `rev_a`, `rev_f` and `rev_g`, and reuses `rev_o_deployment.py`. `rev_w` also
  reads `revision_r4_distilbert_multisplit_probabilities.csv` (`rev_p`),
  `revision_r3_multisplit_probabilities.csv` (`rev_k_multisplit`),
  `revision_realworld_probabilities.csv` (`rev_f`) and
  `revision_distilbert_stream_probabilities.csv` (`rev_g`). It refits the five
  published lightweight partitions for the contemporary stream, and imports
  constants from `rev_c_thresholds.py` and `rev_o_deployment.py`.
* **`rev_x` reuses `rev_w`'s stream store.** It imports
  `rev_w_deployment_robustness.build_store` (which reads the tables listed
  above, including `rev_t3`'s duplicate list) and `rev_o_deployment.py`. It
  writes nothing that `rev_w` reads.
* **`rev_z` reads the corpora only.** That means PhishFuzzer's publisher
  metadata, L2, L3 and Nazario. It writes counts, never text.
* **`rev_y` measures time, so it runs alone.** It fits every family on one
  partition (seed 42), sequentially, and records the system load. Run it after
  the GPU jobs have finished, with nothing else running, so that its ratios
  compare like with like.
* **The two `make_r4_*` figure scripts** read the round-4 outputs, so they run
  last. `make_r4_revised_figures.py` writes `multisplit_inference`, `tradeoff`
  and `evaluation_ladder` (the round-3 `transfer_ladder`, renamed) from the full
  partition sets. The older figure scripts
  no longer write those names: `make_r3_figures.py` writes
  `multisplit_inference_r3`, and `make_revision_figures.py` has retired its
  round-2 ladder. Re-running an older figure script therefore cannot overwrite
  a round-4 figure. `make_r4_revised_figures.py` adds DistilBERT to the
  ladder's recall panel only once `rev_u2` has written all of its partitions;
  run it again after `rev_u2` finishes.

### Runtime

These are wall-clock times from our run on one Apple-silicon machine, rounded,
often with other jobs running. They are indicative only and support no cost
claim; fit-time comparisons are meant to use `rev_y_timing.py`.

* `rev_p_distilbert_multisplit20.py` (GPU) fine-tunes 24 DistilBERT models (12
  partitions, clean and augmented) in about 70 minutes on MPS. It rewrites its
  tables after every partition, so an interrupted run keeps what it finished.
  A restart begins again at the first new seed and overwrites those tables.
* `rev_u2_era_format_distilbert.py` (GPU) fine-tunes 20 DistilBERT models
  (5 partitions × 2 text formats × 2 training sets). `rev_v_backbone_window.py`
  (GPU) fine-tunes 12 `roberta-base` models (3 partitions × 2 window lengths ×
  2 training sets) and evaluates ten checkpoints per epoch. It is the longest
  round-4 script, and it downloads `FacebookAI/roberta-base` from the Hugging
  Face Hub on first use. Both rewrite their tables as they go.
* `rev_q_multisplit100.py` fits its partitions in parallel worker processes
  (`REV_Q_WORKERS`, default 12) and takes about 2 minutes with 12 workers. Its
  recorded fit times are inflated by contention and are not used for any cost
  claim.
* `rev_s_pipeline_signature.py` (`REV_S_WORKERS`, default 10) and
  `rev_u_era_format.py` (`REV_U_WORKERS`, default 10; `REV_U_MPS=1` embeds on
  MPS) each take about 6 minutes.
* `rev_t_human_error_analysis.py` takes about 12 minutes (200 random draws per
  partition and k), and `rev_w_deployment_robustness.py` about 3 minutes.
  `rev_r_r4_inference.py` (2,000 random orderings per K), `rev_u3_proximity.py`,
  `rev_t2_l3_internal_topics.py`, `rev_t3_l3_sa_overlap.py` and
  `rev_c2_dedup_costs.py` each take a few minutes or less.

### Which script answers which comment (round 4)

| Comment | Script | Outputs |
| --- | --- | --- |
| R1.1 pipeline signature: does the L2 gain survive removal of corpus-specific signal? | `rev_s_pipeline_signature.py` | `revision_r4_pipeline_{vocab,erasure,dann,terms,summary}.csv` |
| R1.2 more partitions, or an argument that the interval has stabilized | `rev_q_multisplit100.py` | `revision_r4_multisplit100_{metrics,probabilities}.csv` |
| | `rev_p_distilbert_multisplit20.py` | `revision_r4_distilbert_multisplit_{metrics,probabilities}.csv`, `revision_r4_distilbert_multisplit_run_info.json` |
| | `rev_r_r4_inference.py` | `revision_r4_inference.csv`, `revision_r4_heterogeneity.json`, `revision_r4_stabilization.csv`, `revision_r4_stabilization_tests.json`, `revision_r4_partition_effects.csv`, `revision_r4_partition_effect_summary.csv` |
| R1.3 error analysis of human-authored misses; partition sensitivity of the in-domain ceiling | `rev_t_human_error_analysis.py` | `revision_r4_l3_{corpus_profile,error_profile,vocab_human,vocab_synthetic,topics,coverage,partition_sensitivity}.csv`, `revision_r4_l3_summary.json` |
| | `rev_u3_proximity.py` | `revision_r4_proximity_{recall,summary}.csv` |
| | `rev_t2_l3_internal_topics.py` (what the L3 phishing class contains) | `revision_r4_l3_{internal_topics,cue_families}.csv` |
| | `rev_t3_l3_sa_overlap.py` (the L3 aggregate contains SpamAssassin; found in round 4) | `revision_r4_l3_sa_overlap.json`, `revision_r4_l3_{sa_duplicates,dedup_metrics,dedup_ceiling}.csv` |
| R1.4 second transformer backbone for the window effect | `rev_v_backbone_window.py` | `revision_r4_backbone_{learning_curve,probabilities,truncation_audit}.csv`, `revision_r4_backbone_run_info.json` |
| R1.5 deployment: estimating π and r, rule validation, score-distribution sensitivity | `rev_w_deployment_robustness.py` | `revision_r4_deploy_{prevalence_estimation,cost_ratio_misspecification,grid,purity,rule_validation,score_sensitivity}.csv` |
| | `rev_c2_dedup_costs.py` (the round-3 human-authored cost tables, recomputed without `rev_t3`'s duplicates) | `revision_r4_{threshold_cost_dedup_summary,regions_dedup,miss_cost_boundaries_dedup,misspec_dedup,operational_dedup,break_even_dedup,stream_ceiling_dedup}.csv` |
| | `rev_x_operational_partitions.py` (missed and flagged messages per million over 15 DistilBERT and 20 lightweight partitions) | `revision_r4_{operational,break_even}_partitions.csv` |
| R1.9 / R1.11 corpus facts that the provenance ledger states | `rev_z_corpus_facts.py` | `revision_r4_corpus_facts.json` |
| RQ7 cost claims: every family timed in one session (not a reviewer request) | `rev_y_timing.py` | `revision_r4_timing.csv`, `revision_r4_timing_run_info.json` |
| R1.9 era-matched human corpus; format differences | `fetch_era_matched.sh`, `rev_era_data.py` | the downloaded corpus (no results file) |
| | `rev_u_era_format.py` | `revision_r4_era_format_{metrics,summary}.csv`, `revision_r4_era_by_year.csv`, `revision_r4_era_corpus.json` |
| | `rev_u2_era_format_distilbert.py` | `revision_r4_era_format_distilbert_{metrics,by_year}.csv`, `revision_r4_era_format_distilbert_run_info.json` |
| | `rev_t_human_error_analysis.py` (length, header, topic profile) | `revision_r4_l3_{corpus_profile,topics}.csv` |
| R1.10 the k-shot comparison, with uncertainty and a length control | `rev_t_human_error_analysis.py` | `revision_r4_l3_kshot_{draws,summary}.csv` |
| R2 provenance: is the L2 gain pipeline coverage or phishing coverage? | `rev_s_pipeline_signature.py` | as R1.1 |
| R2 DistilBERT interval and between-partition variance | `rev_p_distilbert_multisplit20.py`, `rev_r_r4_inference.py` | as R1.2 |
| R2 compound shift of the human-authored result (authorship, era, format) | `rev_u_era_format.py`, `rev_u3_proximity.py` | as R1.9 and R1.3 |
| R1.7 figure requests | `make_r4_revised_figures.py`, `make_r4_new_figures.py` | `../figures/` |

R1.6 (editing artifacts) and R1.8 (overstated conclusions) concern the
manuscript text only. R1.11 (the ledger as an organizing principle) is also a
change to the text; `rev_z` supplies the corpus facts the ledger quotes.

## Limitations closed rather than restated

Two limitations arose in the new material the second revision added, and were
at first acknowledged rather than measured. Neither was raised by a reviewer;
neither appears in the originally submitted manuscript. The artifact includes the following analyses and results.

| Limitation as first drafted | Script | Outputs |
| --- | --- | --- |
| L-1 "the design cannot apportion [the L3 collapse] between authorship and distribution shift"; augmentation "is not a substitute for target-domain data", with no quantity attached | `rev_i_domain_adaptation.py` | `revision_domain_adaptation_{by_seed,summary}.csv`, `revision_fewshot_curve.csv`, `revision_human_cluster_structure.json` |
| L-2 "the study makes no L4 claim: nothing here speaks to an adversary who optimizes against the deployed detector" | Aggregate results | `revision_adaptive_{by_seed,summary,transfer,semantics}.csv` |

## Notes on interpretation

* `rev_e_distilbert.py` trains for two epochs to expose the trajectory beyond
  the pre-registered stopping point. The end-of-epoch-1, length-160 checkpoint
  is the pre-registered configuration reported in the main tables.
  `rev_v_backbone_window.py` stops at the pre-registered one epoch.
* Cluster assignments are single-linkage connected components over a cosine
  graph. Single linkage is deliberately permissive: it merges aggressively and
  therefore gives a conservative (small) count of independent units.
* `rev_i_domain_adaptation.py` partitions the human-authored corpus over cosine
  0.90 components, not rows. A row-wise split inflates the in-domain ceiling,
  and so overstates the cross-domain gap that the ceiling is used to bound.
  `rev_t_human_error_analysis.py` reports the same ceiling under row-wise and
  component-disjoint splits at four thresholds, with component counts and
  train/test sizes. The few-shot arms sample at the corpus prevalence. `mixed`
  adds the synthetic set in its collected proportion, and `mixed_balanced`
  oversamples the target to the mass of the synthetic set, so the reported
  penalty is not an artifact of proportion alone.
* The L4 tables summarize detector robustness and cross-model transfer.
* The cost framework normalizes `C_FP` to one, so only the ratio
  `r = C_FN / C_FP` matters. Expected cost depends on a stream only through TPR
  and FPR, so no reweighting of the evaluation corpora is required.
* `rev_r_r4_inference.py` uses an exact sign-flip test for 20 partitions, whose
  minimum attainable two-sided p is 2/2^20. With 100 partitions it uses 200,000
  random sign vectors, whose resolution floor is 1/200,001; the exact minimum
  2/2^100 cannot be reached by Monte Carlo. Partitions drawn from one corpus
  share most of their groups, so the script also reports the corrected
  resampled t-test of Nadeau and Bengio, whose half-width has a floor that more
  partitions cannot cross.
* `rev_s_pipeline_signature.py` never uses L2 labels; only the aggregate L2
  class prevalence enters, through reweighting. Everything that looks at L2 text
  is cross-fitted on two halves of L2 built from whole cosine-0.90 components.
  Residual origin separability is always re-measured with a fresh probe. The
  adversary's own domain head does not count as evidence of invariance, so the
  DANN arm is reported but does not certify that the signature was removed.
* Nazario messages are all phishing, so `rev_u` and `rev_u2` report recall at
  the deployed operating points. Their ROC-AUC uses the only real legitimate
  mail available, which is 2000s-era, so it is era-mismatched on the negative
  side.
* `rev_t3_l3_sa_overlap.py` found that the public L3 aggregate includes the
  SpamAssassin corpus. SpamAssassin spam appears in it labelled as phishing,
  and SpamAssassin ham labelled as safe. The round-3 human-authored cost
  stream, which appended every SpamAssassin message as a negative, therefore
  held some messages twice with contradictory labels. `rev_t3` records the
  duplicate L3 rows by row index only, with no text. `rev_c2` recomputes the
  round-3 human-authored cost tables without them, and `rev_w` drops them from
  its human-authored and contemporary streams. The round-3 files are left
  unchanged, and the `_dedup` files sit beside them.
