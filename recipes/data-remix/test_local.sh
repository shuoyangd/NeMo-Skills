#!/usr/bin/env bash
# Local test for the data-remix pipeline using the nemo-skills pipeline machinery.
# Run from the repo root:
#   bash recipes/data-remix/test_local.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "$REPO_ROOT/recipes/data-remix/pipeline/remix.py" \
    --mode test \
    --stages "separate_data,remix_data"
