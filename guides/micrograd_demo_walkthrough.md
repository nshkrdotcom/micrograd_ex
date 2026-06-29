# Micrograd Demo Walkthrough

## The goal

The demo teaches backpropagation by training a small scalar MLP classifier on a two-dimensional toy dataset. It mirrors the official micrograd workflow while keeping all data generation, loss computation, training, and graph inspection in pure Elixir.

## Scalar autodiff warmup

The notebook starts with scalar values:

```elixir
alias MicrogradEx.Value

x = Value.new(-4.0, label: "x")
y = x |> Value.mul(x) |> Value.relu()

gradients = Value.backward(y)
Value.grad(x, gradients)
```

Each scalar operation creates a new `Value` and records a small local derivative edge. `Value.backward/1` walks the graph in reverse and returns a `Gradients` table.

## The two-moons dataset

The official Python notebook uses sklearn's `make_moons`. MicrogradEx uses `MicrogradEx.Datasets.moons/2`, a deterministic pure-Elixir generator with the same educational role.

Labels are `-1.0` and `1.0` because the max-margin loss uses `yi * scorei`.

## The MLP

The main model is:

```elixir
alias MicrogradEx.NN.MLP

model = MLP.new(2, [16, 16, 1], seed: {1337, 1337, 1337})
```

This means:

* 2 input values;
* first hidden layer with 16 neurons;
* second hidden layer with 16 neurons;
* output layer with 1 neuron.

Hidden layers use ReLU. The final layer is linear.

## Parameter count

The official demo shape has `337` parameters:

```text
First layer: 16 * (2 + 1) = 48
Second layer: 16 * (16 + 1) = 272
Output layer: 1 * (16 + 1) = 17
Total: 337
```

The `+ 1` in each layer is the bias parameter per neuron.

## The max-margin loss

The classification score is the scalar model output. A positive score predicts class `1`; a non-positive score predicts class `-1`.

The loss is:

```text
loss_i = relu(1 - yi * score_i)
data_loss = mean(loss_i)
reg_loss = alpha * sum(p * p)
total_loss = data_loss + reg_loss
```

In code this is `MicrogradEx.Losses.max_margin/4`.

## L2 regularization

The regularization term penalizes large parameters:

```elixir
alpha * sum(p * p for p <- NN.parameters(model))
```

The default `alpha` is `1.0e-4`, matching the official demo.

## The training loop

Training is immutable:

```elixir
gradients = Value.backward(total_loss)
next_model = NN.apply_gradients(model, gradients, learning_rate)
```

`MicrogradEx.Trainer.train/3` runs this loop for 100 steps by default and records loss, data loss, regularization loss, accuracy, and learning rate.

## Plotting loss and accuracy

`MicrogradEx.PlotData` converts training runs into plain rows:

```elixir
PlotData.loss_history(run)
PlotData.accuracy_history(run)
```

The notebook renders those rows with Vega-Lite.

## Decision boundary

The decision boundary is built by evaluating the trained model over a padded grid:

```elixir
PlotData.decision_boundary(run.final_model, dataset, h: 0.25)
```

Every grid point is classified by score sign, then plotted behind the training data.

## Graph inspection

The scalar graph is built during forward operations. MicrogradEx exposes it without mutation:

* `Graph.nodes/2` shows scalar node data and gradients;
* `Graph.edges/1` shows parent-to-child dependencies and local gradients;
* `Graph.to_dot/2` exports DOT text for optional Graphviz rendering.

## What to try next

Change one variable at a time:

* `noise: 0.2`;
* `MLP.new(2, [8, 8, 1])`;
* `steps: 50`;
* `alpha: 0.0`;
* `h: 0.35` for a faster decision-boundary grid.

For a broader set of experiments, open `notebooks/micrograd_extras.livemd`. It compares datasets, model sizes, regularization, learning-rate schedules, decision-boundary resolution, and a spiral dataset challenge.
