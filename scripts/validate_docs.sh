#!/usr/bin/env bash
set -euo pipefail

test -f README.md
test -f guides/getting_started_with_livebook.md
test -f guides/micrograd_demo_walkthrough.md
test -f guides/elixir_design_notes.md
test -f guides/api_reference.md
test -f guides/troubleshooting.md

grep -q "notebooks/micrograd_demo.livemd" README.md
grep -q "notebooks/micrograd_extras.livemd" README.md
grep -q "337" README.md
grep -q "zero_grad" README.md
grep -q "Value.backward" guides/elixir_design_notes.md
grep -q "MLP.new(2, \\[16, 16, 1\\]" guides/micrograd_demo_walkthrough.md
grep -q "PlotData.decision_boundary" guides/api_reference.md

mix docs

echo "Docs validation passed"
