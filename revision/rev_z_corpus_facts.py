#!/usr/bin/env python3
"""Corpus facts quoted in the round-4 ledger and corpus descriptions.

Collects, from the downloaded corpora, the structural and metadata facts that
the provenance ledger states (round 4, R1.9/R1.11): the publisher metadata of
PhishFuzzer (source, creator, year fields), the share of each corpus whose text
opens with a "Subject:" header block, the share containing a neutralized link
token, and median length by class. Text is read only to compute these counts;
nothing but the counts is written.

Output (results/): revision_r4_corpus_facts.json
"""

from __future__ import annotations

import json
import re

import pandas as pd

from rev_common import RESULTS, base, load_corpora
from rev_era_data import load_era_matched
from rev_realworld_data import load_human_phishing

HEADER = re.compile(r"(?i)^\s*\**\s*subject\s*:")


def describe(texts: pd.Series) -> dict:
    return {"n": int(len(texts)),
            "share_opening_with_subject_block": float(texts.str.match(HEADER).mean()),
            "share_with_url_token": float(texts.str.contains("<URL>", regex=False).mean()),
            "median_characters": float(texts.str.len().median())}


def main() -> None:
    raw = pd.read_parquet(base.DATA / "phishfuzzer.parquet")
    raw = raw[raw["Language"].eq("en")]
    primary, external, binary, spam = load_corpora()
    human = load_human_phishing()
    nazario = load_era_matched()
    facts = {
        "phishfuzzer_metadata": {column: raw[column].value_counts().to_dict()
                                 for column in ("Source", "Created by", "Year")},
        "corpora": {
            "PhishFuzzer phishing": describe(binary.loc[binary["label"] == 1, "text"]),
            "PhishFuzzer valid": describe(binary.loc[binary["label"] == 0, "text"]),
            "PhishFuzzer spam (canonical)": describe(spam["text"]),
            "L2 all": describe(external["text"]),
            "L2 phishing": describe(external.loc[external["label"] == 1, "text"]),
            "L2 legitimate": describe(external.loc[external["label"] == 0, "text"]),
            "L3 human phishing": describe(human.loc[human["label"] == 1, "text"]),
            "L3 human legitimate": describe(human.loc[human["label"] == 0, "text"]),
            "Nazario (training layout)": describe(nazario["text_full"]),
            "Nazario (body only)": describe(nazario["text_body"]),
        },
    }
    (RESULTS / "revision_r4_corpus_facts.json").write_text(json.dumps(facts, indent=2, default=str) + "\n")
    print(json.dumps(facts, indent=2, default=str))


if __name__ == "__main__":
    main()
