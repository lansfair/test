#!/usr/bin/env bash
# Example launch script for extracting UniverSat embeddings on PASTIS-R.
#
# Run from the MMSegmentation root (the directory that contains ``tools/``).
# The UniverSat project must be visible on PYTHONPATH.

set -euo pipefail

# Adjust these paths to your local MMSegmentation installation and data.
MSEG_ROOT="${MSEG_ROOT:-.}"
UNIVERSAT_PROJECT="${UNIVERSAT_PROJECT:-$MSEG_ROOT/projects/universat}"
PASTIS_PROJECT="${UNIVERSAT_PROJECT}/pastis"

export PYTHONPATH="${MSEG_ROOT}:${UNIVERSAT_PROJECT}:${PASTIS_PROJECT}:${PYTHONPATH:-}"

python "${UNIVERSAT_PROJECT}/tools/extract_embeddings.py" \
    "${UNIVERSAT_PROJECT}/configs/extract_embeddings_pastisr_universat-base.py" \
    --output-root "work_dirs/universat_pastisr_embeddings" \
    --splits train val test \
    --batch-size 1 \
    --tile-size 0 \
    --device auto \
    --precision bf16 \
    --skip-existing \
    "$@"
