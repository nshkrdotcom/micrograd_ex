defmodule MicrogradEx.LossesTest do
  use ExUnit.Case, async: true

  alias MicrogradEx.Losses
  alias MicrogradEx.NN
  alias MicrogradEx.NN.MLP
  alias MicrogradEx.Value

  describe "max_margin/4" do
    test "returns Value losses and scalar accuracy" do
      model = tiny_model()
      %{xs: xs, ys: ys} = tiny_dataset()

      result = Losses.max_margin(model, xs, ys)

      assert %Losses.Result{} = result
      assert %Value{} = result.total_loss
      assert %Value{} = result.data_loss
      assert %Value{} = result.reg_loss
      assert is_float(result.accuracy)
      assert result.accuracy >= 0.0
      assert result.accuracy <= 1.0
      assert length(result.scores) == length(xs)
    end

    test "computes gradients that reach model parameters" do
      model = tiny_model()
      %{xs: xs, ys: ys} = tiny_dataset()

      result = Losses.max_margin(model, xs, ys)
      gradients = Value.backward(result.total_loss)

      parameter_grads =
        model
        |> NN.parameters()
        |> Enum.map(&Value.grad(&1, gradients))

      assert Enum.any?(parameter_grads, &(&1 != 0.0))
    end

    test "regularization increases total loss" do
      model = tiny_model()
      %{xs: xs, ys: ys} = tiny_dataset()

      no_reg = Losses.max_margin(model, xs, ys, alpha: 0.0)
      reg = Losses.max_margin(model, xs, ys, alpha: 1.0e-2)

      assert reg.total_loss.data > no_reg.total_loss.data
      assert reg.reg_loss.data > 0.0
    end

    test "supports integer labels" do
      model = tiny_model()

      result = Losses.max_margin(model, [[-1.0, 0.0], [1.0, 0.0]], [-1, 1])

      assert %Losses.Result{} = result
      assert length(result.scores) == 2
    end

    test "supports deterministic mini-batches" do
      model = tiny_model()
      xs = for x <- 1..10, do: [x * 1.0, 0.0]
      ys = Enum.map(1..10, fn x -> if rem(x, 2) == 0, do: 1.0, else: -1.0 end)

      first = Losses.max_margin(model, xs, ys, batch_size: 4, seed: {1, 2, 3})
      second = Losses.max_margin(model, xs, ys, batch_size: 4, seed: {1, 2, 3})
      different = Losses.max_margin(model, xs, ys, batch_size: 4, seed: {3, 2, 1})

      assert Enum.map(first.scores, & &1.data) == Enum.map(second.scores, & &1.data)
      assert first.total_loss.data == second.total_loss.data
      assert Enum.map(first.scores, & &1.data) != Enum.map(different.scores, & &1.data)
    end

    test "returns all examples when batch size is nil or larger than the dataset" do
      model = tiny_model()
      %{xs: xs, ys: ys} = tiny_dataset()

      full = Losses.max_margin(model, xs, ys, batch_size: nil)
      oversized = Losses.max_margin(model, xs, ys, batch_size: 99)

      assert length(full.scores) == length(xs)
      assert length(oversized.scores) == length(xs)
      assert full.total_loss.data == oversized.total_loss.data
    end

    test "rejects mismatched xs and ys" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        Losses.max_margin(tiny_model(), [[0.0, 0.0]], [-1.0, 1.0])
      end
    end

    test "rejects empty examples" do
      assert_raise ArgumentError, ~r/at least one/, fn ->
        Losses.max_margin(tiny_model(), [], [])
      end
    end

    test "rejects invalid input rows" do
      assert_raise ArgumentError, ~r/input row/, fn ->
        Losses.max_margin(tiny_model(), [[1.0, :bad]], [1.0])
      end

      assert_raise ArgumentError, ~r/input row/, fn ->
        Losses.max_margin(tiny_model(), [[]], [1.0])
      end
    end

    test "rejects invalid labels" do
      assert_raise ArgumentError, ~r/labels/, fn ->
        Losses.max_margin(tiny_model(), [[1.0, 0.0]], [0.0])
      end
    end

    test "rejects invalid alpha" do
      assert_raise ArgumentError, ~r/alpha/, fn ->
        Losses.max_margin(tiny_model(), [[1.0, 0.0]], [1.0], alpha: -0.1)
      end
    end

    test "rejects invalid batch options" do
      assert_raise ArgumentError, ~r/batch_size/, fn ->
        Losses.max_margin(tiny_model(), [[1.0, 0.0]], [1.0], batch_size: 0)
      end

      assert_raise ArgumentError, ~r/seed/, fn ->
        Losses.max_margin(tiny_model(), [[1.0, 0.0]], [1.0], batch_size: 1, seed: :bad)
      end
    end
  end

  defp tiny_model do
    MLP.new(2, [2, 1], seed: {1, 2, 3})
  end

  defp tiny_dataset do
    %{
      xs: [[-1.0, 0.0], [1.0, 0.0]],
      ys: [-1.0, 1.0]
    }
  end
end
