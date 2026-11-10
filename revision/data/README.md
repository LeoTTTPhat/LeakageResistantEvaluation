# `revision/data/`: downloaded, never redistributed

The real-mail evaluation corpora are fetched here, not shipped. None of them is
used for training, feature selection, calibration, or threshold selection. In
the released archive this directory holds only this file and
`nazario_README.md`.

```bash
# run from revision/
./fetch_realworld.sh      # human-authored corpus (L3) and SpamAssassin
./fetch_era_matched.sh    # era-matched real phishing, Nazario 2015-2025 (round 4)
```

These scripts populate:

| Path | Source | Licence | Fetched by |
| --- | --- | --- | --- |
| `human_phishing_email.csv` | `zefang-liu/phishing-email-dataset` | LGPL-3.0 | `fetch_realworld.sh` |
| `spamassassin/{easy_ham,easy_ham_2,hard_ham,spam,spam_2}/` | Apache SpamAssassin public corpus, 2002--2004 | publicly distributed for research | `fetch_realworld.sh` |
| `nazario/phishing-2015` ... `nazario/phishing-2025`, plus the corpus's `README.txt` and `LICENSE.txt` | J. Nazario, The Online Phishing Corpus | CC-BY-4.0 (attribution to J. Nazario) | `fetch_era_matched.sh` |

The first two corpora are 2000s-era mail and involve no language model. The
Nazario files are era-matched real phishing. Years 2015--2022 are treated as
human-authored. The authorship of 2023--2025 cannot be verified, so those years
are always reported separately. Source, licence, checksums and caveats are in
`nazario_README.md`.

After cleaning (empty and near-empty bodies removed, bodies capped at 6,000
characters, exact duplicates dropped), these corpora yield:

* the 17,415-message human-authored corpus: 6,454 phishing, 10,961 legitimate;
* the 5,888-message real ham/spam stress test: 4,110 ham, 1,778 spam;
* 3,017 era-matched real phishing messages: 1,836 from 2015--2022 and 1,181
  from 2023--2025 (`results/revision_r4_era_corpus.json`).

All three are reported in the manuscript.
