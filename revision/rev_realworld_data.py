#!/usr/bin/env python3
"""Loaders for the two human-authored (non-generated) evaluation corpora.

Neither corpus is produced by a language model, so provenance is verifiable in
exactly the sense the review asks for: every message predates, or is otherwise
independent of, any LLM generation pipeline, and both corpora are trivially
source-disjoint from the PhishFuzzer training pool.

* ``load_human_phishing`` -- aggregated public phishing/legitimate email
  (``zefang-liu/phishing-email-dataset``, LGPL-3.0), used as a real-attack
  transfer test.
* ``load_spamassassin`` -- Apache SpamAssassin public corpus (2002--2004),
  used as a real legitimate-mail and real spam false-positive stress test.

Both are downloaded by ``fetch_realworld.sh`` and are never used for training,
feature selection, calibration, or threshold choice.
"""

from __future__ import annotations

import email
import pathlib

import pandas as pd

from rev_common import base

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
MAX_CHARS = 6000
MIN_CHARS = 20

SPAMASSASSIN_FOLDERS = [
    ("easy_ham", "ham"),
    ("easy_ham_2", "ham"),
    ("hard_ham", "ham"),
    ("spam", "spam"),
    ("spam_2", "spam"),
]


def _clean(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.slice(0, MAX_CHARS)
        .map(base.neutralize)
    )


def load_human_phishing() -> pd.DataFrame:
    """Real phishing versus real legitimate email, human-authored."""
    frame = pd.read_csv(DATA / "human_phishing_email.csv")
    frame = frame.rename(
        columns={"Email Text": "raw_text", "Email Type": "email_type"}
    )
    frame = frame.dropna(subset=["raw_text", "email_type"])
    frame["text"] = _clean(frame["raw_text"])
    frame = frame[frame["text"].str.len() >= MIN_CHARS]
    frame["label"] = frame["email_type"].map(
        {"Phishing Email": 1, "Safe Email": 0}
    )
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    frame["group_id"] = [f"human-{index}" for index in range(len(frame))]
    return frame[["text", "label", "group_id"]]


def _body(message) -> str:
    if message.is_multipart():
        parts = []
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() not in ("text/plain", "text/html"):
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            parts.append(
                payload.decode(
                    part.get_content_charset() or "latin-1", errors="replace"
                )
            )
        return "\n".join(parts)
    payload = message.get_payload(decode=True)
    if payload is None:
        payload = message.get_payload()
        return payload if isinstance(payload, str) else ""
    return payload.decode(message.get_content_charset() or "latin-1", errors="replace")


def load_spamassassin() -> pd.DataFrame:
    """Real legitimate mail and real spam, body text only."""
    root = DATA / "spamassassin"
    rows = []
    for folder, stream in SPAMASSASSIN_FOLDERS:
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.name.startswith("cmds"):
                continue
            try:
                raw = path.read_bytes()
                message = email.message_from_bytes(raw)
                subject = message.get("Subject", "") or ""
                sender = message.get("From", "") or ""
                body = _body(message)
            except Exception:  # pragma: no cover - malformed archive members
                continue
            text = base.neutralize(f"Subject: {subject}\nSender: {sender}\n{body}")[
                :MAX_CHARS
            ]
            if len(text) < MIN_CHARS:
                continue
            rows.append({"text": text, "stream": stream, "source_file": path.name})
    frame = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    frame["label"] = 0  # every SpamAssassin message is a negative for phishing intent
    frame["group_id"] = [f"sa-{index}" for index in range(len(frame))]
    return frame[["text", "label", "group_id", "stream"]]


def realworld_conditions() -> dict[str, pd.DataFrame]:
    human = load_human_phishing()
    spamassassin = load_spamassassin()
    return {
        "human_phishing": human,
        "spamassassin_ham": spamassassin[spamassassin["stream"] == "ham"]
        .reset_index(drop=True),
        "spamassassin_spam": spamassassin[spamassassin["stream"] == "spam"]
        .reset_index(drop=True),
    }
