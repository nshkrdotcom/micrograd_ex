# Getting Started with Livebook

## Prerequisites

You need Elixir and Livebook installed. Use your preferred Elixir installation method for your operating system, then install or run Livebook using the official Livebook instructions.

The demo notebook uses `Mix.install/2` to load this local package and notebook-only visualization dependencies. Kino and Vega-Lite are not required by the core library.

## Clone the repo

```bash
git clone <repo-url>
cd micrograd_ex
```

If this repository is already on your machine, start from the repository root.

## Start Livebook

```bash
livebook server
```

Open the URL printed by Livebook.

## Open the demo notebook

Open:

```text
notebooks/micrograd_demo.livemd
```

The setup cell expects the notebook to live under `notebooks/` because it installs the local package with:

```elixir
Mix.install([
  {:micrograd_ex, path: ".."}
])
```

If you move the notebook, update that path.

## Run all cells

Use Livebook's run-all command from a clean runtime. Running top-to-bottom matters because later cells depend on values created earlier, such as `dataset`, `model`, and `run`.

## Expected outputs

You should see:

* a scalar autodiff value and gradient;
* graph node and edge tables;
* a two-moons dataset table;
* a dataset scatter chart;
* parameter count `337`;
* initial loss and accuracy;
* a 100-step training history table;
* a loss chart;
* an accuracy chart;
* a decision-boundary chart.

The deterministic demo should show loss decreasing and accuracy improving.

## Common setup problems

If Livebook cannot find `micrograd_ex`, confirm you opened `notebooks/micrograd_demo.livemd` from this repository and did not copy the file elsewhere.

If dependency installation fails, restart the Livebook runtime and run the setup cell again. Update notebook dependency versions only after testing them.

If training feels slow, remember that MicrogradEx is scalar and educational. Reduce sample count, hidden width, training steps, or use a coarser decision-boundary grid.

## Next steps

Read [Micrograd demo walkthrough](micrograd_demo_walkthrough.md) for the math and workflow, then [Elixir design notes](elixir_design_notes.md) for the immutable implementation choices.
