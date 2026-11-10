#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python3.11 download_data.py
python3.11 run_experiment.py
python3.11 run_rigor_experiments.py
python3.11 run_transformer.py
python3.11 make_figures.py
python3.11 make_manifest.py
# run_all.sh reproduces the main experiment only; the revision analyses are run
# from revision/ (see revision/README.md), so verify the main experiment here.
python3.11 verify_artifact.py --main-only
