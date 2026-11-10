#!/usr/bin/env python3
"""Loader for the era-matched real-phishing corpus (round 4, comment R1.9).

The human-authored corpus used at L3 is 2000s-era plain-text mail, so its
collapse mixes authorship with era, formatting and topic. Reviewer 1 asks for an
era-matched human corpus. Jose Nazario's public phishing corpus
(https://monkey.org/~jose/phishing/, CC-BY-4.0) is hand-classified real phishing
collected from one inbox, published as one mbox file per year. We use the
eleven yearly files 2015-2025:

* 2015-2022 predate the public release of instruction-tuned chat models
  (November 2022) and are treated as human-authored;
* 2023-2025 are contemporary real phishing whose authorship can no longer be
  assumed to be human, and are reported separately.

Messages keep their own Subject and From headers, so each message is rendered in
two views: ``text_full`` uses exactly the training corpus's
``Subject: ... / Sender: ... / body`` layout, and ``text_body`` drops the header
block, matching the L3 corpus's body-only layout. Bodies prefer the text/plain
part and fall back to tag-stripped text/html. URLs, e-mail addresses and IP
addresses are neutralized by the study's own ``neutralize`` before anything else
sees the text. Attachments are never decoded and links are never followed.

The corpus is downloaded by ``fetch_era_matched.sh`` and is not redistributed.
It is used for evaluation only -- never for training, feature selection,
calibration or threshold choice.
"""

from __future__ import annotations

import email.header
import email.utils
import html
import mailbox
import pathlib
import re
from html.parser import HTMLParser

import pandas as pd

from rev_common import base

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "nazario"
YEARS = list(range(2015, 2026))
PRE_LLM_YEARS = list(range(2015, 2023))
MAX_CHARS = 6000
MIN_CHARS = 20


class _TextExtractor(HTMLParser):
    """Visible text of an HTML part; script and style contents are dropped."""

    BLOCK = {"p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "td"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "title"):
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "title") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # pragma: no cover - malformed markup
        return re.sub(r"<[^>]+>", " ", html.unescape(markup))
    return "".join(parser.parts)


def _decode_header(value) -> str:
    if value is None:
        return ""
    try:
        return str(email.header.make_header(email.header.decode_header(str(value))))
    except Exception:  # pragma: no cover - malformed encoded words
        return str(value)


def _payload_text(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "latin-1"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("latin-1", errors="replace")


def message_body(message) -> tuple[str, str]:
    """Return (body text, source part type) without decoding attachments."""
    plain, markup = [], []
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_filename() or "attachment" in str(part.get("Content-Disposition", "")):
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain.append(_payload_text(part))
        elif content_type == "text/html":
            markup.append(_payload_text(part))
    text = "\n".join(t for t in plain if t.strip())
    if text.strip():
        return text, "text/plain"
    if markup:
        return html_to_text("\n".join(markup)), "text/html"
    return "", "none"


def load_era_matched() -> pd.DataFrame:
    rows = []
    missing = [year for year in YEARS if not (DATA / f"phishing-{year}").exists()]
    if missing:
        raise FileNotFoundError(
            f"era-matched corpus incomplete, missing years {missing}; run fetch_era_matched.sh"
        )
    for year in YEARS:
        path = DATA / f"phishing-{year}"
        for index, message in enumerate(mailbox.mbox(str(path), create=False)):
            try:
                subject = _decode_header(message.get("Subject"))
                sender = _decode_header(message.get("From"))
                body, source = message_body(message)
            except Exception:  # pragma: no cover - malformed archive members
                continue
            body_text = base.neutralize(body)[:MAX_CHARS]
            if len(body_text) < MIN_CHARS:
                continue
            full_text = base.neutralize(
                f"Subject: {subject}\nSender: {sender}\n{body}"
            )[:MAX_CHARS]
            rows.append(
                {
                    "year": year,
                    "mbox_index": index,
                    "text_full": full_text,
                    "text_body": body_text,
                    "body_source": source,
                    "has_subject": bool(subject.strip()),
                }
            )
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["text_full"]).reset_index(drop=True)
    frame["label"] = 1
    frame["era"] = frame["year"].map(
        lambda y: "2015-2022 (pre-LLM)" if y in PRE_LLM_YEARS else "2023-2025 (LLM era)"
    )
    frame["group_id"] = [f"nazario-{y}-{i}" for y, i in zip(frame["year"], frame["mbox_index"])]
    return frame


if __name__ == "__main__":
    corpus = load_era_matched()
    print(corpus.groupby("year").agg(n=("label", "size"),
                                      html_only=("body_source", lambda s: (s == "text/html").mean()),
                                      median_chars=("text_body", lambda s: s.str.len().median())))
    print(len(corpus), "messages")
