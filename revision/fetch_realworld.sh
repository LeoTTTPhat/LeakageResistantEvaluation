#!/usr/bin/env bash
# Download the two human-authored corpora used by the L3 evaluation and verify
# that they are byte-identical to the copies the study used.
#
# Neither corpus is produced by a language model. They are evaluation corpora;
# the only analyses that fit anything to them are the target-data (k-shot)
# experiment and the label-free operating rules, both documented in the paper.
#
#   data/human_phishing_email.csv  aggregated public phishing/legitimate email
#                                  (zefang-liu/phishing-email-dataset, LGPL-3.0),
#                                  pinned to dataset commit 34085a03 (2024-01-17)
#   data/spamassassin/*            Apache SpamAssassin public corpus, 2002-2004
#
# The aggregate contains the SpamAssassin corpus itself (rev_t3_l3_sa_overlap.py
# lists the duplicated rows). Neither corpus is redistributed by this artifact.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/spamassassin
HF_REVISION=34085a032c123ca237f314a01a67909cdea35e34

if [ ! -f data/human_phishing_email.csv ]; then
  echo "Fetching the human-authored phishing corpus ..."
  curl -sL --max-time 300 -o data/human_phishing_email.csv \
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/${HF_REVISION}/Phishing_Email.csv"
fi

SA="https://spamassassin.apache.org/old/publiccorpus"
for archive in 20030228_easy_ham.tar.bz2 20030228_easy_ham_2.tar.bz2 \
               20030228_hard_ham.tar.bz2 20030228_spam.tar.bz2 \
               20030228_spam_2.tar.bz2; do
  folder=$(basename "$archive" .tar.bz2 | sed 's/^[0-9]*_//')
  if [ ! -d "data/spamassassin/$folder" ]; then
    echo "Fetching $archive ..."
    [ -f "data/spamassassin/$archive" ] || \
      curl -sL --max-time 300 -o "data/spamassassin/$archive" "$SA/$archive"
    tar xjf "data/spamassassin/$archive" -C data/spamassassin/
  fi
done

# Verify content: SHA-256 of the aggregate, and for each SpamAssassin folder the
# SHA-256 of its sorted "name<TAB>sha256" listing (dotfiles excluded).
python3 - <<'PY'
import hashlib
import pathlib
import sys

expected_file = "18ef4fff1acb8986f0ab01e83ede6025420c829656af5a377560fce17e153b97"
expected_folders = {
    "easy_ham": (2501, "1a1e7222edc4a4998b559ae1a745b511f5f6bc867efbc2bb8125519beef2654b"),
    "easy_ham_2": (1401, "e3869dd08cb6e3e6f985f05d81f3cea1d8a74cc6f55d9fe2dacb52a59d84981f"),
    "hard_ham": (251, "7c6c6c71ad23ea2f2147e7ef045063d2454c1eb1cb2cb1142823919b6d24027b"),
    "spam": (501, "bcf73eb70aea38339c7186b66eade27e8f1ac5cbf64858a1b281988059d694e2"),
    "spam_2": (1398, "87a684580a67a3e495392dd3e5ddb81a0a95fed0a3b8b3e19061d3090b333ada"),
}
ok = True
digest = hashlib.sha256(pathlib.Path("data/human_phishing_email.csv").read_bytes()).hexdigest()
if digest != expected_file:
    ok = False
    print(f"MISMATCH human_phishing_email.csv: {digest}")
for name, (count, expected) in expected_folders.items():
    folder = pathlib.Path("data/spamassassin") / name
    files = sorted(f for f in folder.iterdir() if f.is_file() and not f.name.startswith("."))
    listing = "".join(f"{f.name}\t{hashlib.sha256(f.read_bytes()).hexdigest()}\n" for f in files)
    digest = hashlib.sha256(listing.encode()).hexdigest()
    if len(files) != count or digest != expected:
        ok = False
        print(f"MISMATCH spamassassin/{name}: {len(files)} files, {digest}")
print("checksums verified" if ok else "checksum verification FAILED")
sys.exit(0 if ok else 1)
PY

python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from rev_realworld_data import realworld_conditions

for name, frame in realworld_conditions().items():
    counts = frame["label"].value_counts().to_dict()
    print(f"{name}: {len(frame)} messages, labels {counts}")
PY
