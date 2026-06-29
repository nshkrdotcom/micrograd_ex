# Troubleshooting

## Livebook cannot find the local package

The notebook setup cell uses:

```elixir
Mix.install([
  {:micrograd_ex, path: ".."}
])
```

That assumes the notebook is at `notebooks/micrograd_demo.livemd`. If you move the notebook, update the path.

## Dependency installation fails

Restart the Livebook runtime and run the setup cell again. The visualization dependencies are notebook-local. Do not add Kino or Vega-Lite to the core application unless you intentionally need them outside notebooks.

If you update dependency versions, run the notebook top-to-bottom afterward.

## The notebook is slow

MicrogradEx is scalar educational autodiff. It is intentionally much slower than tensor libraries. For faster experiments:

* reduce `n_samples`;
* reduce hidden width;
* reduce `steps`;
* increase decision-boundary `h`.

In `notebooks/micrograd_extras.livemd`, prefer `[8, 8, 1]` while experimenting. Move to `[32, 32, 1]` only after the smaller runs behave as expected.

## The decision boundary is slow

The decision boundary evaluates the trained scalar model at every grid point. Smaller `h` values create many more forward passes.

Use a coarser grid:

```elixir
PlotData.decision_boundary(model, dataset, h: 0.35)
```

Avoid very small `h` values in the extras notebook unless you are ready to wait for many scalar forward passes.

## Parameter count is not 337

The official demo count requires exactly:

```elixir
MLP.new(2, [16, 16, 1])
```

The count is:

```text
First layer: 16 * (2 + 1) = 48
Second layer: 16 * (16 + 1) = 272
Output layer: 1 * (16 + 1) = 17
Total: 337
```

Changing input count, hidden widths, or output count changes the total.

## Loss does not decrease

Common causes:

* high dataset noise;
* too few training steps;
* too large or too small learning rate;
* changed seed;
* different architecture;
* notebook cells run out of order.

Start from a clean runtime and run all cells top-to-bottom.

## Notebook cells were run out of order

Use Livebook's run-all command from a clean runtime. Later cells expect values such as `dataset`, `model`, `initial_loss`, and `run` to come from earlier cells.

## `mix docs` fails

Run:

```bash
mix deps.get
mix docs
```

If it still fails, check that every guide listed in `mix.exs` exists.

## Tests fail after changing examples

Run:

```bash
mix format
mix test
```

If a README or guide example changed a public workflow, update the corresponding library tests too. Documentation should describe tested behavior, not a separate path.
