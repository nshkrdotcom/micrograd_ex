<p align="center">
  <img src="assets/micrograd_ex.svg" width="200" height="200" alt="MicrogradEx logo" />
</p>

<p align="center">
  <a href="https://github.com/nshkrdotcom/micrograd_ex">
    <img alt="GitHub: micrograd" src="https://img.shields.io/badge/GitHub-micrograd_ex-0b0f14?logo=github" />
  </a>
  <a href="https://github.com/nshkrdotcom/micrograd_ex/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0b0f14.svg" />
  </a>
</p>


# MicrogradEx

MicrogradEx is a small educational scalar autodiff and neural-network library in Elixir, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

It implements reverse-mode autodiff over scalar `Value` nodes, a tiny neural-network layer, and a Livebook-first recreation of the classic micrograd two-moons classification demo.

The goal is not performance. The goal is to make the mechanics of backpropagation visible in idiomatic Elixir.

## What is this?

MicrogradEx is a pure-Elixir learning project for reverse-mode automatic differentiation. It keeps the same educational scope as the original Python micrograd: scalar values, explicit computation graphs, neurons, layers, and small MLPs.

The main user experience is `notebooks/micrograd_demo.livemd`, which mirrors the official micrograd workflow in pure Elixir: make a two-moons dataset, train `MLP.new(2, [16, 16, 1])`, plot loss and accuracy, and visualize a decision boundary.

## What this repo includes

* scalar `Value` objects with reverse-mode autodiff;
* immutable `Gradients` tables returned from `Value.backward/1`;
* tiny neural-network modules: `Neuron`, `Layer`, and `MLP`;
* deterministic pure-Elixir dataset generators: moons, spiral, and blobs;
* max-margin classification loss with L2 regularization;
* immutable SGD-style training;
* plot-data helpers for Livebook/Vega-Lite;
* scalar graph inspection and DOT export;
* a flagship Livebook demo;
* ExDoc guides.

## Quick start: Livebook demo

Install livebook if needed:

```bash
mix do local.rebar --force, local.hex --force
mix escript.install hex livebook
```

Path may not be found. If needed:

```bash
asdf reshim elixir
```


Clone the repo and open the main notebook:

```bash
git clone <repo-url>
cd micrograd_ex
livebook server
```

Then open:

```text
notebooks/micrograd_demo.livemd
```

Run the notebook from top to bottom. It will:

1. build a two-moons dataset;
2. inspect a scalar autodiff graph;
3. initialize `MLP.new(2, [16, 16, 1])`;
4. confirm the model has `337` parameters;
5. train for 100 steps;
6. plot loss and accuracy;
7. visualize the learned decision boundary.

The notebook uses `Mix.install/2` for Kino and Vega-Lite. Those visualization packages are not runtime dependencies of the core library.

## Extra experiments

After running the main demo, open:

```text
notebooks/micrograd_extras.livemd
```

The extras notebook explores dataset noise, model size, regularization, learning-rate schedules, decision-boundary resolution, and the spiral dataset. It is optional; the main demo remains the official parity path.

## Quick start: IEx

```bash
iex -S mix
```

```elixir
alias MicrogradEx.NN.MLP
alias MicrogradEx.{Datasets, Losses, Trainer}

dataset = Datasets.moons(100, noise: 0.1, seed: {1337, 1337, 1337})
model = MLP.new(2, [16, 16, 1], seed: {1337, 1337, 1337})

initial = Losses.max_margin(model, dataset.xs, dataset.ys)

run =
  Trainer.train(model, dataset,
    steps: 100,
    alpha: 1.0e-4
  )

%{
  initial_loss: initial.total_loss.data,
  final_loss: run.final_loss,
  final_accuracy: run.final_accuracy
}
```

## Official micrograd demo parity

The main Livebook mirrors the original micrograd demo workflow. It does not byte-match sklearn or Matplotlib output; it reproduces the educational workflow in pure Elixir.

| Official Python micrograd | MicrogradEx |
|---|---|
| `sklearn.datasets.make_moons` | `MicrogradEx.Datasets.moons/2` |
| `MLP(2, [16, 16, 1])` | `MicrogradEx.NN.MLP.new(2, [16, 16, 1])` |
| 337 parameters | 337 parameters |
| max-margin loss | `MicrogradEx.Losses.max_margin/4` |
| `model.zero_grad()` | not needed |
| `total_loss.backward()` | `Value.backward(total_loss)` |
| mutate `p.data` | return updated model structs |
| Matplotlib plots | Livebook + Vega-Lite plots |

Expected visuals in the notebook:

* a scatter plot of the two-moons dataset;
* a loss chart showing total, data, and regularization loss;
* an accuracy chart in percent;
* a decision-boundary plot behind the training points.

## Elixir design differences

Python micrograd stores mutable state directly on each `Value`:

* `value.data`;
* `value.grad`.

MicrogradEx keeps `Value` immutable. A backward pass returns a separate `Gradients` table:

```elixir
gradients = MicrogradEx.Value.backward(loss)
dx = MicrogradEx.Value.grad(x, gradients)
```

Model updates are immutable too:

```elixir
next_model = MicrogradEx.NN.apply_gradients(model, gradients, learning_rate)
```

Because gradients are returned fresh from each backward pass, there is no `zero_grad` step.

## Project layout

```text
lib/
  micrograd_ex/
    value.ex          # scalar autodiff values
    gradients.ex      # immutable gradient table
    nn.ex             # Neuron, Layer, MLP
    datasets.ex       # pure-Elixir toy datasets
    losses.ex         # max-margin loss
    trainer.ex        # immutable training loop
    plot_data.ex      # plain rows for plotting
    graph.ex          # graph inspection and DOT export

notebooks/
  micrograd_demo.livemd
  micrograd_extras.livemd

guides/
  getting_started_with_livebook.md
  micrograd_demo_walkthrough.md
  elixir_design_notes.md
  api_reference.md
  troubleshooting.md

test/
```

## Development

```bash
mix deps.get
mix format --check-formatted
mix test
mix credo
mix dialyzer
```

`mix quality` runs the formatter check, Credo, and Dialyzer together.

Validate the Livebook structure:

```bash
bash scripts/validate_livebook.sh
```

Validate documentation:

```bash
bash scripts/validate_docs.sh
```

## Documentation

Generate local docs with:

```bash
mix docs
```

The docs include generated API pages and the guides in `guides/`.

## Guides

* [Getting started with Livebook](guides/getting_started_with_livebook.md)
* [Micrograd demo walkthrough](guides/micrograd_demo_walkthrough.md)
* [Elixir design notes](guides/elixir_design_notes.md)
* [API reference](guides/api_reference.md)
* [Troubleshooting](guides/troubleshooting.md)

## Troubleshooting

If Livebook cannot find the local package, confirm the notebook is still at `notebooks/micrograd_demo.livemd`. Its setup cell uses `Mix.install([{:micrograd_ex, path: ".."}])`.

If training or decision-boundary plotting is slow, remember that this is scalar educational autodiff. Reduce sample count, hidden width, training steps, or use a coarser boundary grid such as `h: 0.35`.

If the parameter count is not `337`, confirm the architecture is exactly:

```elixir
MicrogradEx.NN.MLP.new(2, [16, 16, 1])
```

More setup notes are in [Troubleshooting](guides/troubleshooting.md).

## Attribution

MicrogradEx is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). The Elixir implementation preserves the scalar educational spirit while adapting state flow to immutable data.

## License

See the LICENSE file.
