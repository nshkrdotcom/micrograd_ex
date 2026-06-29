#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK="notebooks/micrograd_extras.livemd"

test -f "$NOTEBOOK"

grep -q "Mix.install" "$NOTEBOOK"
grep -q "MicrogradEx" "$NOTEBOOK"
grep -q "Datasets.moons" "$NOTEBOOK"
grep -q "Datasets.spiral" "$NOTEBOOK"
grep -q "Trainer.train" "$NOTEBOOK"
grep -q "decision" "$NOTEBOOK"
grep -q "regularization" "$NOTEBOOK"
grep -q "learning" "$NOTEBOOK"
grep -q "\\[32, 32, 1\\]" "$NOTEBOOK"

echo "Extras Livebook validation passed: $NOTEBOOK"
