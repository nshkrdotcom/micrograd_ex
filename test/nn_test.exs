defmodule MicrogradEx.NNTest do
  use ExUnit.Case, async: true

  alias MicrogradEx.NN
  alias MicrogradEx.NN.Layer
  alias MicrogradEx.NN.MLP
  alias MicrogradEx.NN.Neuron
  alias MicrogradEx.Value

  describe "neuron" do
    test "computes an affine forward pass and parameter gradients" do
      neuron =
        Neuron.new(2,
          weights: [0.5, -1.0],
          bias: 0.25,
          nonlin: false
        )

      [w0, w1, bias] = NN.parameters(neuron)
      output = NN.forward(neuron, [2.0, -3.0])
      gradients = Value.backward(output)

      assert output.data == 4.25
      assert Value.grad(w0, gradients) == 2.0
      assert Value.grad(w1, gradients) == -3.0
      assert Value.grad(bias, gradients) == 1.0
    end

    test "applies ReLU when requested" do
      neuron =
        Neuron.new(1,
          weights: [-2.0],
          bias: 0.0,
          nonlin: true
        )

      [weight, bias] = NN.parameters(neuron)
      output = NN.forward(neuron, 3.0)
      gradients = Value.backward(output)

      assert output.data == 0.0
      assert Value.grad(weight, gradients) == 0.0
      assert Value.grad(bias, gradients) == 0.0
    end

    test "updates parameters immutably" do
      neuron =
        Neuron.new(1,
          weights: [0.0],
          bias: 0.0,
          nonlin: false
        )

      [old_weight, old_bias] = NN.parameters(neuron)
      prediction = NN.forward(neuron, [2.0])
      loss = prediction |> Value.sub(4.0) |> Value.pow(2)
      gradients = Value.backward(loss)
      updated = NN.apply_gradients(neuron, gradients, 0.1)
      [new_weight, new_bias] = NN.parameters(updated)

      assert old_weight.data == 0.0
      assert old_bias.data == 0.0
      assert new_weight.data == 1.6
      assert new_bias.data == 0.8
      assert new_weight.id != old_weight.id
      assert new_bias.id != old_bias.id
    end
  end

  describe "layer and MLP construction" do
    test "layer forward unwraps singleton output while forward_many always returns a list" do
      layer = Layer.new(2, 1, weights: [1.0, 2.0], bias: -1.0, nonlin: false)

      assert %Value{} = NN.forward(layer, [3.0, 4.0])
      assert [%Value{}] = Layer.forward_many(layer, [3.0, 4.0])
      assert NN.forward(layer, [3.0, 4.0]).data == 10.0
    end

    test "MLP parameter count matches the original micrograd shape" do
      mlp = MLP.new(3, [4, 4, 1], seed: {101, 102, 103})

      assert NN.parameter_count(mlp) == 41
    end

    test "seeded MLP initialization is deterministic without reusing identical neurons" do
      first = MLP.new(2, [3, 1], seed: {1, 2, 3})
      second = MLP.new(2, [3, 1], seed: {1, 2, 3})

      first_weights = first |> NN.parameters() |> Enum.map(& &1.data)
      second_weights = second |> NN.parameters() |> Enum.map(& &1.data)

      assert first_weights == second_weights
      assert first_weights |> Enum.uniq() |> length() > 2
    end

    test "MLP supports a one-neuron hidden layer by passing lists internally" do
      mlp = MLP.new(1, [1, 1], seed: {7, 8, 9})

      assert %Value{} = NN.forward(mlp, [0.5])
    end
  end

  describe "training" do
    test "learns a one-dimensional linear function with gradient descent" do
      initial =
        Neuron.new(1,
          weights: [0.0],
          bias: 0.0,
          nonlin: false
        )

      data = [
        {-2.0, -5.0},
        {-1.0, -3.0},
        {0.0, -1.0},
        {1.0, 1.0},
        {2.0, 3.0}
      ]

      initial_loss = linear_regression_loss(initial, data).data

      trained =
        Enum.reduce(1..80, initial, fn _step, model ->
          loss = linear_regression_loss(model, data)
          gradients = Value.backward(loss)
          NN.apply_gradients(model, gradients, 0.03)
        end)

      final_loss = linear_regression_loss(trained, data).data
      [weight, bias] = NN.parameters(trained)

      assert final_loss < initial_loss
      assert final_loss < 1.0e-8
      assert_in_delta weight.data, 2.0, 1.0e-4
      assert_in_delta bias.data, -1.0, 1.0e-4
    end
  end

  defp linear_regression_loss(model, data) do
    data
    |> Enum.map(fn {x, target} ->
      model
      |> NN.forward([x])
      |> Value.sub(target)
      |> Value.pow(2)
    end)
    |> Value.sum()
  end
end
