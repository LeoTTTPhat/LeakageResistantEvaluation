#!/usr/bin/env python3
"""What does the 2000s human-authored "phishing" class contain? (round 4, R1.3/R1.9)

``rev_t_human_error_analysis.py`` fits one topic model to all phishing corpora,
and 95% of the 2000s L3 phishing class loads on a single generic topic: the
corpus shares almost no thematic vocabulary with the synthetic phishing it is
compared against. This script characterizes that class on its own terms: an NMF
topic model fitted to L3 phishing alone, the share of each topic, and the
augmented word model's recall within each topic (majority over the five
published partitions), together with the share of L3 phishing that matches
simple cue families for commercial spam (pharmacy, stock promotion, advertising
and unsubscribe language) and advance-fee fraud. If much of the class is
commercial spam, the L3 collapse partly reflects a difference in what the two
corpora call phishing: PhishFuzzer keeps spam as a separate, negative class.

Outputs (results/):
  revision_r4_l3_internal_topics.csv
  revision_r4_l3_cue_families.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from rev_common import RESULTS, base, load_corpora, split_frames
from rev_realworld_data import load_human_phishing

SEEDS = [17, 29, 42, 71, 113]
N_TOPICS = 10
CUES = {
    "pharmacy spam": r"\b(viagra|cialis|pharmacy|meds|pills|prescription|vicodin|xanax|valium|levitra)\b",
    "stock promotion": r"\b(stock|stocks|shares|investor|forward[- ]looking|otc|symbol|ticker)\b",
    "advertising / unsubscribe": r"\b(unsubscribe|remove me|to be removed|opt[- ]out|mailing list|sponsored|advertis\w*)\b",
    "advance-fee fraud": r"\b(beneficiary|next of kin|barrister|inheritance|consignment|million (?:us )?dollars|transfer of (?:the )?fund)\b",
    "account / credential": r"\b(verify your|confirm your|account (?:has been|will be)|password|login|log in|sign in|suspended|unusual activity)\b",
}


def main() -> None:
    primary, external, binary, spam = load_corpora()
    human = load_human_phishing()
    phish = human[human["label"] == 1].reset_index(drop=True)
    caught = np.zeros((len(SEEDS), len(phish)), dtype=bool)
    for i, seed in enumerate(SEEDS):
        frames = split_frames(binary, external, spam, seed)
        model = base.WordModel.fit(frames["train_all"]["text"].tolist(), frames["train_all"]["label"].to_numpy())
        caught[i] = model.predict_proba(phish["text"].tolist()) >= 0.5
    majority = caught.mean(axis=0) >= 0.5

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.4, stop_words="english",
                            strip_accents="unicode", sublinear_tf=True,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    x = tfidf.fit_transform(phish["text"])
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=0, max_iter=800).fit(x)
    weights = nmf.transform(x)
    dominant = np.where(weights.max(1) > 0, weights.argmax(1), -1)
    names = tfidf.get_feature_names_out()
    rows = []
    for t in range(N_TOPICS):
        mask = dominant == t
        rows.append({"topic": t, "top_terms": ", ".join(names[np.argsort(-nmf.components_[t])[:12]]),
                     "share": float(mask.mean()), "n": int(mask.sum()),
                     "recall_augmented_word_0.5": float(majority[mask].mean()) if mask.any() else np.nan})
    rows.append({"topic": -1, "top_terms": "(no topic weight)", "share": float((dominant == -1).mean()),
                 "n": int((dominant == -1).sum()),
                 "recall_augmented_word_0.5": float(majority[dominant == -1].mean()) if (dominant == -1).any() else np.nan})
    topics = pd.DataFrame(rows).sort_values("share", ascending=False)
    topics.to_csv(RESULTS / "revision_r4_l3_internal_topics.csv", index=False)

    lower = phish["text"].str.lower()
    synthetic = binary.loc[binary["label"] == 1, "text"].str.lower()
    nazario = None
    try:
        from rev_era_data import load_era_matched
        nazario = load_era_matched()["text_full"].str.lower()
    except Exception:  # pragma: no cover - corpus not downloaded
        pass
    cue_rows = []
    for name, pattern in CUES.items():
        hit = lower.str.contains(pattern, regex=True).to_numpy()
        record = {"cue_family": name, "share_L3_phishing": float(hit.mean()),
                  "recall_within": float(majority[hit].mean()) if hit.any() else np.nan,
                  "share_synthetic_phishing": float(synthetic.str.contains(pattern, regex=True).mean())}
        if nazario is not None:
            record["share_nazario"] = float(nazario.str.contains(pattern, regex=True).mean())
        cue_rows.append(record)
    any_spam = lower.str.contains("|".join(CUES[k] for k in ("pharmacy spam", "stock promotion",
                                                             "advertising / unsubscribe")), regex=True).to_numpy()
    cue_rows.append({"cue_family": "any commercial-spam cue", "share_L3_phishing": float(any_spam.mean()),
                     "recall_within": float(majority[any_spam].mean()),
                     "share_synthetic_phishing": float(synthetic.str.contains("|".join(
                         CUES[k] for k in ("pharmacy spam", "stock promotion", "advertising / unsubscribe")),
                         regex=True).mean())})
    cues = pd.DataFrame(cue_rows)
    cues.to_csv(RESULTS / "revision_r4_l3_cue_families.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(topics.round(3).to_string(index=False))
    print(cues.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
