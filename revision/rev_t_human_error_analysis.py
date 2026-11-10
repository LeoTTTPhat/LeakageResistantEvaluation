#!/usr/bin/env python3
"""Why does synthetic training fail on human-authored phishing? (round 4)

Comments R1.3, R1.9 and R1.10. Round 3 decomposed the L3 shortfall into a
threshold component and a discrimination deficit and stated -- without
measuring it -- that the synthetic corpus occupies a narrow region of the space
human phishing spans. Reviewer 1 asks for an error analysis comparing what the
detector misses on human phishing with what it catches on synthetic phishing,
for the features present in human phishing but absent from synthetic data, for
a check that the component-disjoint partition does not distort the in-domain
ceiling, and for a more careful treatment of the claim that 25 human messages
outperform the 9,054-message synthetic set. This script provides:

1. Corpus profile: length, header block, link/address tokens, HTML origin,
   mentioned years, and lexicon rates for urgency, credential, payment,
   advance-fee, commercial-spam and brand cues, per corpus and class.
2. Error profile: the same features for human phishing the augmented word model
   misses (FN) and catches (TP), against synthetic phishing it catches, at the
   deployed threshold 0.5 and at the target-oracle threshold, as standardized
   mean differences.
3. Vocabulary: log-odds with an informative Dirichlet prior (Monroe et al.,
   2008) between human and synthetic phishing; n-grams frequent in human
   phishing but absent from the synthetic training data; out-of-vocabulary
   rates under the detector's own vocabulary.
4. Topics: NMF topics fitted on all phishing corpora, topic shares per corpus,
   and the synthetic-trained detector's recall on human and era-matched
   phishing per topic.
5. Coverage: k-NN manifold coverage (Kynkaanniemi et al., 2019) between
   synthetic and human phishing in MiniLM and TF-IDF space, which measures the
   "narrow region" claim directly, and detector recall inside versus outside
   the synthetic manifold.
6. Partition sensitivity of the in-domain ceiling: row-wise versus component-
   disjoint splits at cosine 0.95, 0.90, 0.85 and 0.80, with component counts
   and train/test sizes.
7. The k-shot comparison with uncertainty: 200 random draws per partition and
   k, intervals over draws, the probability that k human messages beat the
   synthetic set under three metrics, a length-matched draw, a length-stratified
   evaluation, and the k-shot models scored on L1 and L2.

Outputs (results/): revision_r4_l3_*.csv / .json (see main()).
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from rev_common import RESULTS, base, encode_minilm, load_corpora, load_embeddings, split_frames
from rev_era_data import load_era_matched
from rev_realworld_data import load_human_phishing

SEEDS = [17, 29, 42, 71, 113]
K_SHOTS = [10, 25, 50, 100, 250]
DRAWS = 200
HOLDOUT = 0.30
N_TOPICS = 12

LEXICON = {
    "urgency": r"\b(urgent|urgently|immediately|immediate|asap|today|within \d+ (?:hours|days)|"
               r"expire[sd]?|expiring|expiration|deadline|final (?:notice|warning)|act now|"
               r"limited time|right away|as soon as possible|suspend(?:ed|ion)?)\b",
    "credential": r"\b(account|password|passwords|login|log in|sign in|verify|verification|"
                  r"confirm|credentials|username|security|unusual activity|locked|"
                  r"unauthori[sz]ed|update your|validate|authenticate)\b",
    "payment": r"\b(payment|invoice|bank|banking|card|credit|refund|transaction|billing|"
               r"wire|paypal|tax|irs)\b",
    "advance_fee": r"\b(million|millions|usd|us\$|beneficiary|inheritance|next of kin|barrister|"
                   r"compensation|lottery|winner|prize|consignment|diplomat|deceased|"
                   r"foreign|fund|funds|nigeria|abroad|partnership|confidential)\b",
    "commercial_spam": r"\b(offer|free|discount|buy|order now|cheap|viagra|cialis|pills|pharmacy|"
                       r"meds|prescription|stock|investment|mortgage|loan|casino|dating|"
                       r"unsubscribe|save \d+)\b",
    "brand": r"\b(paypal|microsoft|office 365|outlook|apple|icloud|amazon|netflix|dhl|fedex|ups|"
             r"usps|wells fargo|chase|bank of america|ebay|google|facebook|instagram|linkedin|"
             r"docusign|dropbox|adobe|coinbase|binance|metamask)\b",
    "call_to_action": r"\b(click|click here|follow the link|link below|visit|download|open the|"
                      r"reply|respond|call us|contact us)\b",
}
YEAR = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sparse_components(embeddings: np.ndarray, threshold: float, block: int = 1024) -> np.ndarray:
    n = len(embeddings)
    rows, cols = [], []
    for start in range(0, n, block):
        sim = embeddings[start:start + block] @ embeddings.T
        r, c = np.nonzero(sim >= threshold)
        rows.append(r + start)
        cols.append(c)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    graph = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def component_split(labels_of_component, y, seed):
    frame = pd.DataFrame({"c": labels_of_component, "y": y})
    majority = frame.groupby("c")["y"].mean().round().astype(int)
    rng = np.random.default_rng(seed)
    test_ids = []
    for value in (0, 1):
        ids = rng.permutation(majority.index[majority == value].to_numpy())
        cut = int(round(len(ids) * (1 - HOLDOUT)))
        test_ids.append(ids[cut:])
    test = np.isin(labels_of_component, np.concatenate(test_ids))
    return ~test, test


def profile(texts: pd.Series) -> pd.DataFrame:
    lower = texts.str.lower()
    tokens = texts.str.split().str.len().clip(lower=1)
    out = pd.DataFrame(
        {
            "chars": texts.str.len(),
            "tokens": tokens,
            "header_block": texts.str.match(r"(?i)^\s*\**\s*subject\s*:").astype(float),
            "url_tokens": texts.str.count("<URL>"),
            "email_tokens": texts.str.count("<EMAIL>"),
            "has_url": texts.str.contains("<URL>").astype(float),
            "uppercase_share": texts.map(lambda t: sum(ch.isupper() for ch in t) / max(1, sum(ch.isalpha() for ch in t))),
            "exclamations_per_100": 100 * texts.str.count("!") / tokens,
            "currency_symbols": texts.str.count(r"[$£€]"),
            "years_mentioned": texts.map(lambda t: [int(y) for y in YEAR.findall(t)]),
        }
    )
    out["mentions_year"] = out["years_mentioned"].map(bool).astype(float)
    out["max_year_mentioned"] = out["years_mentioned"].map(lambda ys: max(ys) if ys else np.nan)
    out["mentions_2015_or_later"] = out["years_mentioned"].map(lambda ys: float(any(y >= 2015 for y in ys)))
    for name, pattern in LEXICON.items():
        out[f"{name}_per_100"] = 100 * lower.str.count(pattern) / tokens
    return out.drop(columns=["years_mentioned"])


def smd(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Standardized mean difference (a - b) / pooled SD, per column."""
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2).replace(0, np.nan)
    return (a.mean() - b.mean()) / pooled


def log_odds_dirichlet(counts_a, counts_b, prior, alpha0=1000.0):
    """Monroe, Colaresi & Quinn (2008) z-scores, a versus b."""
    na, nb = counts_a.sum(), counts_b.sum()
    alpha = alpha0 * prior / prior.sum()
    a0 = alpha.sum()
    delta = (np.log((counts_a + alpha) / (na + a0 - counts_a - alpha))
             - np.log((counts_b + alpha) / (nb + a0 - counts_b - alpha)))
    variance = 1.0 / (counts_a + alpha) + 1.0 / (counts_b + alpha)
    return delta / np.sqrt(variance)


def oracle_f1(y, p):
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    f1 = 2 * tp / (np.arange(1, len(y) + 1) + y.sum())
    i = int(np.argmax(f1))
    return float(f1[i]), float(p[order][i])


def manifold_coverage(reference, query, k: int = 5) -> np.ndarray:
    """Boolean per query row: inside the union of k-NN balls of the reference set.

    Rows are unit vectors (dense MiniLM, or sparse l2-normalized TF-IDF), so the
    Euclidean distance is sqrt(2 - 2 cos)."""
    def distances(a, b):
        product = a @ b.T
        if hasattr(product, "toarray"):
            product = product.toarray()
        return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.asarray(product)))
    d_ref = distances(reference, reference)
    np.fill_diagonal(d_ref, np.inf)
    radius = np.partition(d_ref, k - 1, axis=1)[:, k - 1]
    inside = np.zeros(query.shape[0], dtype=bool)
    for start in range(0, query.shape[0], 1024):
        d = distances(query[start:start + 1024], reference)
        inside[start:start + 1024] = (d <= radius[None, :]).any(axis=1)
    return inside


def unit_rows(matrix):
    """TfidfVectorizer rows are already l2-normalized; keep them sparse."""
    return matrix.tocsr()


class FastWord:
    """Exact, tokenize-once reimplementation of ``base.WordModel`` for many refits.

    ``base.WordModel`` re-tokenizes its evaluation texts on every call, which is
    prohibitive for thousands of k-shot refits. Here every text is tokenized
    once with the same analyzer (word 1-2 grams, unicode accent stripping,
    lowercasing); each refit then selects its vocabulary (min_df = 2,
    max_features = 50,000 by corpus frequency), smoothed idf, sublinear tf and
    l2 normalization from the cached counts, exactly as TfidfVectorizer does.
    """

    def __init__(self, texts):
        from sklearn.feature_extraction.text import CountVectorizer as _CV
        self.counts = _CV(ngram_range=(1, 2), strip_accents="unicode").fit_transform(texts).tocsr()

    def _transform(self, rows, keep, idf):
        from sklearn.preprocessing import normalize
        m = self.counts[rows][:, keep].astype(float).tocsr()
        m.data = 1.0 + np.log(m.data)
        m = m.multiply(idf[None, :]).tocsr()
        return normalize(m, norm="l2")

    def fit_predict(self, train_rows, y, eval_rows):
        from sklearn.linear_model import LogisticRegression as _LR
        x = self.counts[train_rows]
        df = np.asarray((x > 0).sum(0)).ravel()
        keep = np.flatnonzero(df >= 2)
        if len(keep) > 50000:
            total = np.asarray(x[:, keep].sum(0)).ravel()
            keep = np.sort(keep[np.argsort(-total, kind="stable")[:50000]])
        idf = np.log((1 + len(train_rows)) / (1 + df[keep])) + 1.0
        clf = _LR(C=4.0, class_weight="balanced", max_iter=3000, solver="liblinear").fit(
            self._transform(train_rows, keep, idf), y)
        return [clf.predict_proba(self._transform(rows, keep, idf))[:, 1] for rows in eval_rows]


# --------------------------------------------------------------------------
def main() -> None:
    started = time.time()
    primary, external, binary, spam = load_corpora()
    primary_embeddings, external_embeddings = load_embeddings(primary, external)
    human = load_human_phishing()
    human_embeddings = encode_minilm(human["text"])
    nazario = load_era_matched()
    nazario_embeddings = encode_minilm(nazario["text_full"])
    y_h = human["label"].to_numpy()
    out = {}

    # ---------------- 1. corpus profile ----------------------------------
    corpora = {
        "PhishFuzzer phishing": binary.loc[binary["label"] == 1, "text"],
        "PhishFuzzer valid": binary.loc[binary["label"] == 0, "text"],
        "L2 phishing": external.loc[external["label"] == 1, "text"],
        "L2 legitimate": external.loc[external["label"] == 0, "text"],
        "L3 human phishing": human.loc[human["label"] == 1, "text"],
        "L3 human legitimate": human.loc[human["label"] == 0, "text"],
        "Nazario 2015-2022": nazario.loc[nazario["year"] <= 2022, "text_full"],
        "Nazario 2023-2025": nazario.loc[nazario["year"] >= 2023, "text_full"],
    }
    profiles = {name: profile(texts.reset_index(drop=True)) for name, texts in corpora.items()}
    rows = []
    for name, frame in profiles.items():
        record = {"corpus": name, "n": len(frame)}
        for column in frame.columns:
            record[f"{column}_median"] = float(frame[column].median())
            record[f"{column}_mean"] = float(frame[column].mean())
        rows.append(record)
    pd.DataFrame(rows).to_csv(RESULTS / "revision_r4_l3_corpus_profile.csv", index=False)
    print("profile done", flush=True)

    # ---------------- models on the five published partitions -------------
    human_scores = np.zeros((len(SEEDS), len(human)))
    nazario_scores = np.zeros((len(SEEDS), len(nazario)))
    minilm_human = np.zeros((len(SEEDS), len(human)))
    linked_tp_profiles, vocab_sets = [], []
    for i, seed in enumerate(SEEDS):
        frames = split_frames(binary, external, spam, seed)
        train = frames["train_all"]
        model = base.WordModel.fit(train["text"].tolist(), train["label"].to_numpy())
        human_scores[i] = model.predict_proba(human["text"].tolist())
        nazario_scores[i] = model.predict_proba(nazario["text_full"].tolist())
        embed = base.EmbeddingModel.fit(primary_embeddings[train.index.to_numpy()], train["label"].to_numpy())
        minilm_human[i] = embed.predict_proba(human_embeddings)
        linked = frames["linked_rewrites"]
        p_linked = model.predict_proba(linked["text"].tolist())
        tp = linked[(linked["label"] == 1).to_numpy() & (p_linked >= 0.5)]
        linked_tp_profiles.append(profile(tp["text"].reset_index(drop=True)))
        vocab_sets.append(set(model.vectorizer.vocabulary_))
    synthetic_tp = pd.concat(linked_tp_profiles, ignore_index=True)

    # ---------------- 2. error profile ----------------------------------
    h_pos = np.flatnonzero(y_h == 1)
    human_profile = profile(human["text"]).iloc[h_pos].reset_index(drop=True)
    p_pos = human_scores[:, h_pos]
    error_rows = []
    for rule in ("threshold_0.5", "oracle_threshold"):
        tp_mask = np.zeros_like(p_pos, dtype=bool)
        for i in range(len(SEEDS)):
            if rule == "threshold_0.5":
                t = 0.5
            else:
                t = oracle_f1(y_h, human_scores[i])[1]
            tp_mask[i] = p_pos[i] >= t
        caught = tp_mask.mean(axis=0) >= 0.5  # caught in the majority of partitions
        fn, tp = human_profile[~caught], human_profile[caught]
        d_fn_tp = smd(fn, tp)
        d_fn_syn = smd(fn, synthetic_tp)
        for column in human_profile.columns:
            error_rows.append({
                "rule": rule, "feature": column,
                "human_fn_mean": float(fn[column].mean()), "human_tp_mean": float(tp[column].mean()),
                "synthetic_tp_mean": float(synthetic_tp[column].mean()),
                "smd_fn_vs_human_tp": float(d_fn_tp[column]),
                "smd_fn_vs_synthetic_tp": float(d_fn_syn[column]),
                "n_fn": int(len(fn)), "n_tp": int(len(tp)), "n_synthetic_tp": int(len(synthetic_tp)),
            })
    pd.DataFrame(error_rows).to_csv(RESULTS / "revision_r4_l3_error_profile.csv", index=False)
    print("error profile done", flush=True)

    # ---------------- 3. vocabulary --------------------------------------
    synthetic_phish = binary.loc[binary["label"] == 1, "text"].tolist()
    human_phish = human.loc[human["label"] == 1, "text"].tolist()
    counter = CountVectorizer(ngram_range=(1, 2), min_df=5, strip_accents="unicode", binary=True)
    matrix = counter.fit_transform(synthetic_phish + human_phish)
    terms = counter.get_feature_names_out()
    df_syn = np.asarray(matrix[: len(synthetic_phish)].sum(0)).ravel()
    df_hum = np.asarray(matrix[len(synthetic_phish):].sum(0)).ravel()
    z = log_odds_dirichlet(df_hum.astype(float), df_syn.astype(float), (df_hum + df_syn).astype(float))
    vocab = pd.DataFrame({"term": terms, "df_human": df_hum, "df_synthetic": df_syn,
                          "share_human": df_hum / len(human_phish),
                          "share_synthetic": df_syn / len(synthetic_phish), "z_human_vs_synthetic": z})
    vocab.sort_values("z_human_vs_synthetic", ascending=False).head(40).assign(direction="human").to_csv(
        RESULTS / "revision_r4_l3_vocab_human.csv", index=False)
    vocab.sort_values("z_human_vs_synthetic").head(40).assign(direction="synthetic").to_csv(
        RESULTS / "revision_r4_l3_vocab_synthetic.csv", index=False)
    absent = vocab[(vocab["share_human"] >= 0.01) & (vocab["df_synthetic"] == 0)]
    absent_terms = set(absent["term"])
    analyzer = counter.build_analyzer()
    has_absent = np.array([bool(absent_terms & set(analyzer(t))) for t in human_phish])
    union_vocab = set.intersection(*vocab_sets)

    def oov(texts):
        unigram = CountVectorizer(ngram_range=(1, 1), strip_accents="unicode").build_analyzer()
        rates = []
        for t in texts:
            toks = unigram(t)
            if toks:
                rates.append(np.mean([tok not in union_vocab for tok in toks]))
        return float(np.mean(rates))

    caught05 = (p_pos >= 0.5).mean(axis=0) >= 0.5
    vocab_summary = {
        "frequent_human_ngrams_absent_from_synthetic": int(len(absent)),
        "human_phishing_with_an_absent_frequent_ngram": float(has_absent.mean()),
        "examples_absent": absent.sort_values("share_human", ascending=False)["term"].head(30).tolist(),
        "oov_rate_human_phishing": oov(human_phish),
        "oov_rate_human_fn": oov([human_phish[i] for i in np.flatnonzero(~caught05)]),
        "oov_rate_human_tp": oov([human_phish[i] for i in np.flatnonzero(caught05)]),
        "oov_rate_l2_phishing": oov(external.loc[external["label"] == 1, "text"].tolist()),
        "oov_rate_nazario": oov(nazario["text_full"].tolist()),
        "recall_human_with_absent_ngram": float(caught05[has_absent].mean()),
        "recall_human_without_absent_ngram": float(caught05[~has_absent].mean()),
    }
    print("vocabulary done", flush=True)

    # ---------------- 4. topics ------------------------------------------
    canonical_phish = base.canonical(binary[binary["label"] == 1])["text"].tolist()
    topic_corpora = {
        "PhishFuzzer phishing": canonical_phish,
        "L2 phishing": external.loc[external["label"] == 1, "text"].tolist(),
        "L3 human phishing": human_phish,
        "Nazario 2015-2022": nazario.loc[nazario["year"] <= 2022, "text_full"].tolist(),
        "Nazario 2023-2025": nazario.loc[nazario["year"] >= 2023, "text_full"].tolist(),
    }
    docs, owner = [], []
    for name, texts in topic_corpora.items():
        docs += texts
        owner += [name] * len(texts)
    owner = np.array(owner)
    # equal weight per corpus in the factorization
    weights = pd.Series(owner).map(1.0 / pd.Series(owner).value_counts()).to_numpy()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.4, stop_words="english",
                            strip_accents="unicode", sublinear_tf=True,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    x = tfidf.fit_transform(docs)
    xw = x.multiply(np.sqrt(weights * len(weights) / len(topic_corpora))[:, None]).tocsr()
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=0, max_iter=600)
    nmf.fit(xw)
    doc_topics = nmf.transform(x)
    dominant = np.where(doc_topics.max(1) > 0, doc_topics.argmax(1), -1)
    names = tfidf.get_feature_names_out()
    topic_rows = []
    for t in range(N_TOPICS):
        top = names[np.argsort(-nmf.components_[t])[:12]].tolist()
        record = {"topic": t, "top_terms": ", ".join(top)}
        for name in topic_corpora:
            record[f"share_{name}"] = float((dominant[owner == name] == t).mean())
        topic_rows.append(record)
    human_topic = dominant[owner == "L3 human phishing"]
    naz_mask = np.isin(owner, ["Nazario 2015-2022", "Nazario 2023-2025"])
    naz_topic = dominant[naz_mask]
    naz_caught = (nazario_scores >= 0.5).mean(axis=0) >= 0.5
    for record in topic_rows:
        t = record["topic"]
        record["recall_human_at_0.5"] = float(caught05[human_topic == t].mean()) if (human_topic == t).any() else np.nan
        record["n_human"] = int((human_topic == t).sum())
        record["recall_nazario_at_0.5"] = float(naz_caught[naz_topic == t].mean()) if (naz_topic == t).any() else np.nan
        record["n_nazario"] = int((naz_topic == t).sum())
    topics = pd.DataFrame(topic_rows)
    topics.to_csv(RESULTS / "revision_r4_l3_topics.csv", index=False)
    print("topics done", flush=True)

    # ---------------- 5. coverage ----------------------------------------
    rng = np.random.default_rng(0)
    syn_idx = binary.index[(binary["label"] == 1).to_numpy()].to_numpy()
    spaces = {
        "MiniLM": {
            "synthetic": primary_embeddings[syn_idx],
            "human": human_embeddings[h_pos],
            "L2": external_embeddings[(external["label"] == 1).to_numpy()],
            "nazario": nazario_embeddings,
        }
    }
    tf = TfidfVectorizer(ngram_range=(1, 1), min_df=5, sublinear_tf=True, strip_accents="unicode")
    tf.fit(synthetic_phish + human_phish + external.loc[external["label"] == 1, "text"].tolist()
           + nazario["text_full"].tolist())
    spaces["TF-IDF"] = {
        "synthetic": unit_rows(tf.transform(synthetic_phish)),
        "human": unit_rows(tf.transform(human_phish)),
        "L2": unit_rows(tf.transform(external.loc[external["label"] == 1, "text"].tolist())),
        "nazario": unit_rows(tf.transform(nazario["text_full"].tolist())),
    }
    cover_rows = []
    n_sub = 1800
    for space, sets in spaces.items():
        for repeat in range(5):
            sub = {k: v[rng.choice(v.shape[0], size=min(n_sub, v.shape[0]), replace=False)] for k, v in sets.items()}
            for reference, query in (("synthetic", "human"), ("human", "synthetic"),
                                     ("synthetic", "L2"), ("L2", "synthetic"),
                                     ("synthetic", "nazario"), ("nazario", "synthetic"),
                                     ("human", "human"), ("synthetic", "synthetic")):
                if reference == query:
                    half = sub[reference].shape[0] // 2
                    inside = manifold_coverage(sub[reference][:half], sub[reference][half:])
                else:
                    inside = manifold_coverage(sub[reference], sub[query])
                cover_rows.append({"space": space, "repeat": repeat, "reference": reference,
                                   "query": query, "share_inside": float(inside.mean())})
    coverage = pd.DataFrame(cover_rows)
    coverage.groupby(["space", "reference", "query"])["share_inside"].agg(["mean", "std"]).reset_index().to_csv(
        RESULTS / "revision_r4_l3_coverage.csv", index=False)
    # recall inside versus outside the synthetic manifold (MiniLM, full sets)
    inside_full = manifold_coverage(spaces["MiniLM"]["synthetic"][rng.choice(len(syn_idx), 3000, replace=False)],
                                    spaces["MiniLM"]["human"])
    coverage_detect = {
        "share_human_phishing_inside_synthetic_manifold": float(inside_full.mean()),
        "word_recall_inside": float(caught05[inside_full].mean()),
        "word_recall_outside": float(caught05[~inside_full].mean()),
        "minilm_recall_inside": float(((minilm_human[:, h_pos] >= 0.5).mean(0) >= 0.5)[inside_full].mean()),
        "minilm_recall_outside": float(((minilm_human[:, h_pos] >= 0.5).mean(0) >= 0.5)[~inside_full].mean()),
    }
    print("coverage done", flush=True)

    # ---------------- 6. partition sensitivity ---------------------------
    sensitivity_rows, structure = [], {}
    comp = {th: sparse_components(human_embeddings, th) for th in (0.95, 0.90, 0.85, 0.80)}
    for th, labels in comp.items():
        sizes = np.bincount(labels)
        structure[str(th)] = {"components": int(len(sizes)), "largest": int(sizes.max()),
                              "rows_in_multi_member": int(sizes[sizes > 1].sum())}
    for scheme in ["row-wise", 0.95, 0.90, 0.85, 0.80]:
        for seed in SEEDS:
            if scheme == "row-wise":
                rng_s = np.random.default_rng(seed)
                test = np.zeros(len(human), dtype=bool)
                for value in (0, 1):
                    idx = rng_s.permutation(np.flatnonzero(y_h == value))
                    test[idx[int(round(len(idx) * (1 - HOLDOUT))):]] = True
                train = ~test
            else:
                train, test = component_split(comp[scheme], y_h, seed)
            tr, te = np.flatnonzero(train), np.flatnonzero(test)
            word = base.WordModel.fit(human["text"].iloc[tr].tolist(), y_h[tr])
            p = word.predict_proba(human["text"].iloc[te].tolist())
            frames = split_frames(binary, external, spam, seed)
            synth = base.WordModel.fit(frames["train_all"]["text"].tolist(), frames["train_all"]["label"].to_numpy())
            ps = synth.predict_proba(human["text"].iloc[te].tolist())
            test_has_near_train = None
            if scheme != "row-wise":
                test_has_near_train = 0.0
            else:
                sim = human_embeddings[te] @ human_embeddings[tr].T
                test_has_near_train = float((sim.max(1) >= 0.90).mean())
            sensitivity_rows.append({
                "scheme": str(scheme), "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
                "test_share": float(len(te) / len(human)),
                "test_prevalence": float(y_h[te].mean()),
                "test_rows_with_train_neighbour_at_0.90": test_has_near_train,
                "ceiling_f1": float(f1_score(y_h[te], p >= 0.5)),
                "ceiling_pr_auc": float(average_precision_score(y_h[te], p)),
                "synthetic_f1": float(f1_score(y_h[te], ps >= 0.5)),
                "synthetic_oracle_f1": oracle_f1(y_h[te], ps)[0],
                "gap_f1": float(f1_score(y_h[te], p >= 0.5) - f1_score(y_h[te], ps >= 0.5)),
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(RESULTS / "revision_r4_l3_partition_sensitivity.csv", index=False)
    print("partition sensitivity done", flush=True)

    # ---------------- 7. k-shot with uncertainty and controls -------------
    shot_rows = []
    syn_len = binary["text"].str.len().to_numpy()
    syn_deciles = np.quantile(syn_len, np.linspace(0, 1, 11))
    fast = FastWord(human["text"].tolist() + binary["text"].tolist() + external["text"].tolist())
    binary_row = {index: len(human) + position for position, index in enumerate(binary.index)}
    external_rows = np.arange(len(external)) + len(human) + len(binary)
    checked = False
    for seed in SEEDS:
        train, test = component_split(comp[0.90], y_h, seed)
        tr, te = np.flatnonzero(train), np.flatnonzero(test)
        y_te = y_h[te]
        te_texts = human["text"].iloc[te].tolist()
        te_len = human["text"].iloc[te].str.len().to_numpy()
        quartile = np.digitize(te_len, np.quantile(te_len, [0.25, 0.5, 0.75]))
        frames = split_frames(binary, external, spam, seed)
        synth = base.WordModel.fit(frames["train_all"]["text"].tolist(), frames["train_all"]["label"].to_numpy())
        ps = synth.predict_proba(te_texts)
        linked = frames["linked_rewrites"]
        reference = {
            "f1_0.5": float(f1_score(y_te, ps >= 0.5)),
            "f1_oracle": oracle_f1(y_te, ps)[0],
            "pr_auc": float(average_precision_score(y_te, ps)),
            "roc_auc": float(roc_auc_score(y_te, ps)),
            "f1_0.5_longest_quartile": float(f1_score(y_te[quartile == 3], ps[quartile == 3] >= 0.5)),
            "pr_auc_longest_quartile": float(average_precision_score(y_te[quartile == 3], ps[quartile == 3])),
            "L1_f1": float(f1_score(linked["label"], synth.predict_proba(linked["text"].tolist()) >= 0.5)),
            "L2_f1": float(f1_score(external["label"].astype(int), synth.predict_proba(external["text"].tolist()) >= 0.5)),
        }
        shot_rows.append({"seed": seed, "k": 9054, "sampling": "synthetic (all)", "draw": -1, **reference})
        tr_len = human["text"].iloc[tr].str.len().to_numpy()
        tr_bin = np.clip(np.digitize(tr_len, syn_deciles[1:-1]), 0, 9)
        rng = np.random.default_rng(seed + 31)
        prevalence = y_h.mean()
        pos_pool, neg_pool = tr[y_h[tr] == 1], tr[y_h[tr] == 0]
        for k in K_SHOTS:
            n_pos = max(1, int(round(k * prevalence)))
            n_neg = k - n_pos
            for sampling in ("random", "length-matched"):
                for draw in range(DRAWS):
                    if sampling == "random":
                        chosen = np.r_[rng.choice(pos_pool, n_pos, replace=False),
                                       rng.choice(neg_pool, n_neg, replace=False)]
                    else:
                        # draw length bins from the synthetic training distribution
                        chosen = []
                        for pool, m in ((pos_pool, n_pos), (neg_pool, n_neg)):
                            bins = tr_bin[np.searchsorted(tr, pool)]
                            want = rng.integers(0, 10, size=m)
                            for b in want:
                                candidates = pool[bins == b]
                                if len(candidates) == 0:
                                    candidates = pool[np.argsort(np.abs(bins - b))[:50]]
                                chosen.append(rng.choice(candidates))
                        chosen = np.unique(np.array(chosen))
                        if len(np.unique(y_h[chosen])) < 2:
                            continue
                    if sampling == "length-matched" and draw >= 50:
                        break
                    want_reverse = k == 25 and draw < 50 and sampling == "random"
                    linked_rows = np.array([binary_row[i] for i in linked.index])
                    blocks = [te] + ([linked_rows, external_rows] if want_reverse else [])
                    predictions = fast.fit_predict(chosen, y_h[chosen], blocks)
                    p = predictions[0]
                    if not checked:  # the fast path must reproduce base.WordModel exactly
                        reference_model = base.WordModel.fit(human["text"].iloc[chosen].tolist(), y_h[chosen])
                        gap = float(np.abs(reference_model.predict_proba(te_texts) - p).max())
                        if gap > 1e-6:
                            raise AssertionError(f"FastWord deviates from WordModel by {gap}")
                        print(f"FastWord check passed (max |diff| = {gap:.2e})", flush=True)
                        checked = True
                    row = {"seed": seed, "k": k, "sampling": sampling, "draw": draw,
                           "train_median_chars": float(np.median(human["text"].iloc[chosen].str.len())),
                           "f1_0.5": float(f1_score(y_te, p >= 0.5)),
                           "f1_oracle": oracle_f1(y_te, p)[0],
                           "pr_auc": float(average_precision_score(y_te, p)),
                           "roc_auc": float(roc_auc_score(y_te, p)),
                           "f1_0.5_longest_quartile": float(f1_score(y_te[quartile == 3], p[quartile == 3] >= 0.5)),
                           "pr_auc_longest_quartile": float(average_precision_score(y_te[quartile == 3], p[quartile == 3]))}
                    if want_reverse:
                        row["L1_f1"] = float(f1_score(linked["label"], predictions[1] >= 0.5))
                        row["L2_f1"] = float(f1_score(external["label"].astype(int), predictions[2] >= 0.5))
                    shot_rows.append(row)
        print(f"k-shot seed {seed} done ({time.time() - started:.0f}s)", flush=True)
    shots = pd.DataFrame(shot_rows)
    shots.to_csv(RESULTS / "revision_r4_l3_kshot_draws.csv", index=False)
    synthetic = shots[shots["k"] == 9054].set_index("seed")
    summary_rows = []
    for (k, sampling), block in shots[shots["k"] != 9054].groupby(["k", "sampling"]):
        record = {"k": k, "sampling": sampling, "draws": int(len(block))}
        for metric in ("f1_0.5", "f1_oracle", "pr_auc", "roc_auc", "f1_0.5_longest_quartile",
                       "pr_auc_longest_quartile", "L1_f1", "L2_f1"):
            if metric not in block or block[metric].isna().all():
                continue
            values = block[metric].dropna()
            record[f"{metric}_median"] = float(values.median())
            record[f"{metric}_p025"] = float(values.quantile(0.025))
            record[f"{metric}_p975"] = float(values.quantile(0.975))
            ref = block.loc[values.index, "seed"].map(synthetic[metric])
            record[f"{metric}_synthetic"] = float(synthetic[metric].mean())
            record[f"{metric}_share_beating_synthetic"] = float((values > ref).mean())
        record["train_median_chars"] = float(block["train_median_chars"].median())
        summary_rows.append(record)
    pd.DataFrame(summary_rows).to_csv(RESULTS / "revision_r4_l3_kshot_summary.csv", index=False)

    out = {
        "vocabulary": vocab_summary,
        "coverage_and_detection": coverage_detect,
        "component_structure": structure,
        "synthetic_training_median_chars": float(np.median(syn_len)),
        "human_phishing_median_chars": float(np.median(human.loc[human["label"] == 1, "text"].str.len())),
        "seconds": time.time() - started,
    }
    (RESULTS / "revision_r4_l3_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.set_option("display.width", 250)
    print(json.dumps(out, indent=2))
    print(topics.round(3).to_string(index=False))
    print(coverage.groupby(["space", "reference", "query"])["share_inside"].mean().round(3).to_string())
    print(sensitivity.groupby("scheme").mean(numeric_only=True).round(3).to_string())
    print(pd.DataFrame(summary_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
