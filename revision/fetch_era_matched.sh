#!/usr/bin/env bash
# Download the era-matched real-phishing corpus used by the round-4 analyses.
#
# J. Nazario, "The Online Phishing Corpus" (https://monkey.org/~jose/phishing/):
# hand-classified real phishing collected from one inbox, one mbox file per
# year. Eleven yearly files are used, 2015-2025:
#
#   data/nazario/phishing-2015 ... phishing-2025   yearly mbox files (~102 MB)
#   data/nazario/README.txt, LICENSE.txt           the corpus's own notes and licence
#
# 2015-2022 are treated as human-authored (the cut-off is the public release of
# instruction-tuned chat models, November 2022); the authorship of 2023-2025
# cannot be verified, so those years are always reported separately. The corpus is used for evaluation
# only -- never for training, feature selection, calibration, or threshold
# selection -- and is read by rev_era_data.py.
#
# Licence CC-BY-4.0, attribution to J. Nazario. The corpus is not
# redistributed by this artifact; see data/nazario_README.md.
#
# Files already present are not downloaded again, but every yearly file is
# checked against the SHA-256 pinned below on every run. A mismatch stops the
# script with a non-zero exit status. A fresh download is written to a .part
# file and only moved into place once its checksum matches.
#
# NAZARIO_BASE overrides the source URL (e.g. a local mirror); the pinned
# checksums apply whatever the source.
set -euo pipefail
cd "$(dirname "$0")"

BASE="${NAZARIO_BASE:-https://monkey.org/~jose/phishing}"
DEST="data/nazario"
mkdir -p "$DEST"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mismatch() {  # mismatch <file> <expected> <actual> <what to do>
  {
    echo
    echo "=================================================================="
    echo "CHECKSUM MISMATCH: $1"
    echo "  expected $2"
    echo "  got      $3"
    echo "  $4"
    echo "=================================================================="
  } >&2
}

failed=0
while read -r year expected; do
  file="$DEST/phishing-$year"
  if [ ! -f "$file" ]; then
    echo "Fetching phishing-$year ..."
    curl -fsSL --retry 3 --max-time 600 -o "$file.part" "$BASE/phishing-$year"
    actual=$(sha256_of "$file.part")
    if [ "$actual" != "$expected" ]; then
      mismatch "$file.part (fresh download)" "$expected" "$actual" \
        "Nothing was installed; the download is kept as $file.part for inspection."
      failed=$((failed + 1))
      continue
    fi
    mv "$file.part" "$file"
  fi
  actual=$(sha256_of "$file")
  if [ "$actual" != "$expected" ]; then
    mismatch "$file" "$expected" "$actual" \
      "Delete the file and re-run this script to download it again."
    failed=$((failed + 1))
    continue
  fi
  printf '  phishing-%s  verified  %12s bytes\n' "$year" "$(wc -c < "$file" | tr -d ' ')"
done <<'SHA256'
2015 7523e62a5e6e4ee8134ab8b1a628e423097a7779e1003b28422864aa34f1c319
2016 42393e4d686193bfb879d1608c23fd4d3b698ccd4d73f0e1d2276ba709718483
2017 110fba876b38b44035bf065788952674bc68a7b6879c345a515a9ac844b1d704
2018 83f49cf2a76c7b29b205d22d8778e46848e36af0f15ee1df126a543696acd48d
2019 cab39d6fcb38cf80cdaabb3e6ffd1222777e2fcaac7e8480321b7f729c2e3c0e
2020 99e20bf25f5e8556d37103f2c8140cd75181a62d3df91df8acf9d9aece9f08bb
2021 636707cff42245d1482e4ae8a2c03369717a5e06bf9d1a0f4ec5dcd846661cee
2022 a6291e2cb990204766fce425423949b403cce4ae6a660fc8d08812e40a063fa7
2023 5007f1c1dc91b15e93e988afa3a1cf402e15c88f765d3057be8bfec50dae1fe8
2024 60afa40a2757170093a2bde31ec9cda4e54d33992203ce04bfbbc86d2e5e5b99
2025 f1fa7e0fe35c9a16f36d9aa20ade7e1d3908d1d0c1d917b8c53c8aa799ec1c8f
SHA256

if [ "$failed" -ne 0 ]; then
  echo "$failed yearly file(s) failed SHA-256 verification; stopping." >&2
  exit 1
fi

# The corpus's own notes and licence. Fetched for attribution; they are not
# checksummed because the maintainer updates them, and no analysis reads them.
for doc in README.txt LICENSE.txt; do
  if [ ! -f "$DEST/$doc" ]; then
    echo "Fetching $doc ..."
    curl -fsSL --retry 3 --max-time 120 -o "$DEST/$doc.part" "$BASE/$doc"
    mv "$DEST/$doc.part" "$DEST/$doc"
  fi
done

echo "All eleven yearly files verified."
echo "Corpus: J. Nazario, The Online Phishing Corpus, CC-BY-4.0 (attribution required)."

# Parse the files with the study's own loader and compare the cleaned corpus
# with the composition the released results were computed on. A difference
# here, with verified files, points to the parsing environment (Python and its
# email package), not to the download.
if ! "${PYTHON:-python3}" - <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, ".")
from rev_era_data import load_era_matched

corpus = load_era_matched()
counts = {str(year): int(n) for year, n in corpus["year"].value_counts().sort_index().items()}
for year, n in counts.items():
    print(f"  {year}: {n} messages after cleaning")
print(f"  total: {len(corpus)} messages "
      f"({int((corpus['year'] <= 2022).sum())} from 2015-2022, "
      f"{int((corpus['year'] >= 2023).sum())} from 2023-2025)")

released = pathlib.Path("..") / "results" / "revision_r4_era_corpus.json"
if released.exists():
    reference = json.loads(released.read_text())["nazario_by_year"]
    if reference == counts:
        print("  matches the composition in results/revision_r4_era_corpus.json")
    else:
        print("WARNING: the cleaned corpus differs from results/revision_r4_era_corpus.json:",
              file=sys.stderr)
        for year in sorted(set(reference) | set(counts)):
            if reference.get(year) != counts.get(year):
                print(f"  {year}: released {reference.get(year)}, here {counts.get(year)}",
                      file=sys.stderr)
PY
then
  echo "Corpus summary skipped: the study's loader could not run with ${PYTHON:-python3}." >&2
  echo "The files themselves are verified; set PYTHON to the study interpreter to see the summary." >&2
fi
