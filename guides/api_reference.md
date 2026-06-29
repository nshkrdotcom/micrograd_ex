# Practical API Reference

This guide is a compact cookbook. Generated ExDoc pages provide the full module reference.

## Values and gradients

```elixir
alias MicrogradEx.Value

x = Value.new(2.0, label: "x")
y = x |> Value.mul(x) |> Value.add(1.0)

gradients = Value.backward(y)

%{
  y: y.data,
  dy_dx: Value.grad(x, gradients)
}
```

## Neural networks

```elixir
alias MicrogradEx.NN
alias MicrogradEx.NN.MLP

model = MLP.new(2, [16, 16, 1], seed: {1337, 1337, 1337})

score = NN.forward(model, [0.2, -0.3])
count = NN.parameter_count(model)
```

`count` is `337` for `MLP.new(2, [16, 16, 1])`.

## Datasets

```elixir
alias MicrogradEx.Datasets

dataset = Datasets.moons(100, noise: 0.1, seed: {1337, 1337, 1337})

length(dataset.xs)
length(dataset.ys)
Enum.take(dataset.points, 3)
```

Other deterministic helpers:

```elixir
Datasets.spiral(100, noise: 0.1, seed: {1, 2, 3})
Datasets.blobs(100, noise: 0.2, seed: {1, 2, 3})
```

## Losses

```elixir
alias MicrogradEx.Losses

loss = Losses.max_margin(model, dataset.xs, dataset.ys)

%{
  total_loss: loss.total_loss.data,
  data_loss: loss.data_loss.data,
  reg_loss: loss.reg_loss.data,
  accuracy: loss.accuracy
}
```

## Training

```elixir
alias MicrogradEx.Trainer

run =
  Trainer.train(model, dataset,
    steps: 100,
    alpha: 1.0e-4,
    learning_rate: &Trainer.official_micrograd_learning_rate/1
  )

%{
  final_loss: run.final_loss,
  final_accuracy: run.final_accuracy
}
```

## Plot data

```elixir
alias MicrogradEx.PlotData

points = PlotData.dataset_points(dataset)
history = PlotData.loss_history(run)
accuracy = PlotData.accuracy_history(run)
boundary = PlotData.decision_boundary(run.final_model, dataset, h: 0.25)
```

These functions return plain rows. They do not depend on Livebook or Vega-Lite.

## Graph inspection

```elixir
alias MicrogradEx.Graph
alias MicrogradEx.Value

gradients = Value.backward(loss.total_loss)

nodes = Graph.nodes(loss.total_loss, gradients)
edges = Graph.edges(loss.total_loss)
dot = Graph.to_dot(loss.total_loss, gradients)
```

The DOT text can be copied into a Graphviz renderer, but Graphviz is not required by MicrogradEx.
