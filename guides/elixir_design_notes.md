# Elixir Design Notes

## Why immutable values?

Elixir data is immutable by default. MicrogradEx follows that model instead of simulating Python objects with mutable fields. Every scalar operation returns a new `Value` that carries the graph needed for backpropagation.

## Why gradients are external

Python micrograd writes gradients onto each `Value`. MicrogradEx returns gradients as data:

```elixir
alias MicrogradEx.Value

gradients = Value.backward(loss)
Value.grad(x, gradients)
```

This makes gradient flow explicit and prevents accidental reuse of stale gradient state.

## Why there is no zero_grad

In Python micrograd, gradients accumulate on each `Value`, so training must call `model.zero_grad()` before each backward pass. In MicrogradEx, each call to `Value.backward/1` returns a fresh `Gradients` table. Nothing is stored back onto the model parameters, so there is nothing to zero.

## Why model updates return new structs

Model parameters are `Value` structs, and models are immutable Elixir structs. A training step returns the next model:

```elixir
next_model = MicrogradEx.NN.apply_gradients(model, gradients, learning_rate)
```

The previous model remains available for inspection or comparison.

## Why this stays scalar

MicrogradEx is for education, not throughput. It intentionally does not introduce tensors, GPU kernels, Nx, Axon, Torchx, or Python numerical backends. The point is to see additions, multiplications, powers, ReLUs, and local derivatives.

## Why datasets are pure Elixir

The Livebook mirrors the official two-moons workflow, but it does not import sklearn. `MicrogradEx.Datasets.moons/2` generates deterministic interleaving half-moons directly in Elixir.

## Why plotting is separated from core logic

`MicrogradEx.PlotData` returns plain lists of maps. The notebook renders those rows with Kino and Vega-Lite. This keeps Livebook, Kino, and Vega-Lite out of the core runtime dependencies.

## Tradeoffs

Scalar autodiff is slow compared with tensor libraries. That is acceptable here because the repository is a learning tool. The small examples make graph construction, gradients, immutable updates, and training behavior easy to inspect.
