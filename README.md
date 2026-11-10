# Leakage-resistant evaluation of rewrite augmentation for phishing detection

Reproduction artifact for the Expert Systems with Applications manuscript
*Within-Pipeline Gains, Proximity-Bound Real-World Detection: A Leakage-Resistant,
Provenance-Audited Evaluation of Rewrite Augmentation for Phishing Detection*.

The study keeps linked rewrites of each source email together during splitting.
It evaluates lexical, MiniLM, and transformer detectors on held-out source
groups and measures transfer to external and human-authored email corpora.

## Verify the bundled results

From this directory, run the verifier with Python 3.11. This check uses the
standard library and requires no corpus downloads.

```bash
python3.11 verify_artifact.py --results-only
```

The verifier checks result hashes, expected outputs and figures, source-group
isolation, the row-wise leakage control, and probability ranges. It also
checks that prediction and round-4 probability tables exclude message-text
columns.

## Reproduce the experiments

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_all.sh
```

The runner downloads the two pinned primary corpora, runs the main experiments,
generates their figures and manifest, and verifies the outputs and corpus
checksums. For the follow-up analyses, see the
[revision run order](revision/README.md).

## Artifact contents

| Path | Contents |
| --- | --- |
| [Study guide](STUDY.md) | Study design, main experiment scripts, and output guide |
| [`revision/`](revision/README.md) | Revision analyses, dependencies, and runtimes |
| [`results/`](results/) | Derived results, probabilities, and checksum manifest |
| [`figures/`](figures/) | Publication figures in PDF, PNG, and TikZ formats |
| [`data/`](data/README.md) and [`revision/data/`](revision/data/README.md) | Corpus sources, pinned revisions, and licenses |
| [`cache/`](cache/README.md) | Instructions for regenerating MiniLM embeddings |

The artifact ships derived results and corpus documentation. The downloaders
retrieve the corpora from their upstream sources. The study regenerates
embedding caches locally.

The L4 robustness analysis is represented by aggregate tables and figures.
