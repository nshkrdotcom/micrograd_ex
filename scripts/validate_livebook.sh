#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK="notebooks/micrograd_demo.livemd"

test -f "$NOTEBOOK"

grep -q "Mix.install" "$NOTEBOOK"
grep -q "MicrogradEx" "$NOTEBOOK"
grep -q "Datasets.moons" "$NOTEBOOK"
grep -q "MLP.new(2, \\[16, 16, 1\\]" "$NOTEBOOK"
grep -q "337" "$NOTEBOOK"
grep -q "Losses.max_margin" "$NOTEBOOK"
grep -q "Trainer.train" "$NOTEBOOK"
grep -q "PlotData.decision_boundary" "$NOTEBOOK"
grep -q "Graph.nodes" "$NOTEBOOK"
grep -q "Graph.edges" "$NOTEBOOK"
grep -q "Graph.to_dot" "$NOTEBOOK"
grep -q "loss" "$NOTEBOOK"
grep -q "accuracy" "$NOTEBOOK"

echo "Livebook validation passed: $NOTEBOOK"
