# `revision/data/nazario/`: era-matched real phishing (downloaded, never redistributed)

This README sits next to the corpus directory, not inside it. Everything inside
`nazario/` is the downloaded corpus, and none of it ships with the artifact.

## Source

J. Nazario, *The Online Phishing Corpus*, <https://monkey.org/~jose/phishing/>.
This is real phishing collected from one personal inbox and classified by hand.
It is published as one mbox file per year (upstream covers 2005 to 2025), together
with the maintainer's `README.txt` and `LICENSE.txt`.

The maintainer's own notes (in the downloaded `README.txt`) say the corpus is
meant to be representative rather than exhaustive, since it is a single
inbox. They also say hand classification may contain errors, and that only the
earlier mailboxes were anonymized.

## Licence and attribution

CC-BY-4.0 (Creative Commons Attribution 4.0 International). Any use must
attribute the corpus to J. Nazario, for example:

> J. Nazario, The Online Phishing Corpus, https://monkey.org/~jose/phishing/,
> licensed under CC BY 4.0.

The manuscript cites it this way. The licence would allow redistribution with
attribution, but this artifact does not redistribute the corpus. This matches
how the other corpora are handled, and the later yearly files are not
anonymized upstream. The files are downloaded and verified instead, and
`results/` contains no message text from them.

## Years used and how to fetch them

Eleven yearly files, 2015 to 2025, about 102 MB in total:

```bash
./fetch_era_matched.sh    # run from revision/
```

The script skips files that are already present. On every run it checks each
yearly file against the SHA-256 pinned in the script, and it stops with an
error on any mismatch. It also fetches the maintainer's `README.txt` and
`LICENSE.txt` for attribution; those two are not checksummed, because upstream
updates them and no analysis reads them.

| File | Bytes | Messages after cleaning |
| --- | ---: | ---: |
| `phishing-2015` | 8,815,568 | 262 |
| `phishing-2016` | 12,578,274 | 390 |
| `phishing-2017` | 11,352,450 | 271 |
| `phishing-2018` | 8,775,701 | 251 |
| `phishing-2019` | 5,211,476 | 210 |
| `phishing-2020` | 5,826,445 | 145 |
| `phishing-2021` | 2,719,946 | 94 |
| `phishing-2022` | 4,775,954 | 213 |
| `phishing-2023` | 7,763,704 | 381 |
| `phishing-2024` | 15,457,816 | 356 |
| `phishing-2025` | 18,769,863 | 444 |

The message counts come from `results/revision_r4_era_corpus.json`: 3,017
messages in total.

## Authorship: two eras, never pooled as "human"

* **2015 to 2022** (1,836 messages) are treated as human-authored. The
  cut-off is the public release of instruction-tuned chat models in November
  2022.
* **2023 to 2025** (1,181 messages) are contemporary real phishing whose
  authorship **cannot be verified**: any of them may have been written or
  rewritten with a language model. They are always reported separately and are
  never counted as human-authored evidence.

## How the analyses use it

Evaluation only. The corpus is never used for training, feature selection,
calibration, or threshold selection. `rev_era_data.py` loads it:

* It decodes headers and prefers the `text/plain` part, falling back to
  tag-stripped `text/html`.
* Attachments are never decoded and links are never followed.
* URLs, e-mail addresses and IP addresses are neutralized by the study's own
  `neutralize` before any model sees the text.
* Bodies are capped at 6,000 characters, bodies under 20 characters are
  dropped, and exact duplicates are removed.

Each message has two views. `text_full` uses the training corpus's
`Subject / Sender / body` layout, and `text_body` drops the header block.

The loader skips a yearly file that is absent rather than failing. Run
`fetch_era_matched.sh` to completion, so that all eleven files are verified,
before any script that reads the corpus: `rev_t_human_error_analysis.py`,
`rev_u_era_format.py`, `rev_u2_era_format_distilbert.py`,
`rev_u3_proximity.py`, `rev_w_deployment_robustness.py` and
`rev_z_corpus_facts.py`.

Every message is phishing, so recall is the era-matched quantity. The only real
legitimate mail available for ranking metrics is 2000s-era: the L3 legitimate
messages and SpamAssassin ham. Any ROC-AUC against it is therefore
era-mismatched on the negative side, and is labelled that way wherever it is
reported.
