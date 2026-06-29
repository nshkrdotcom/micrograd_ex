defmodule MicrogradEx.TrainerTest do
  use ExUnit.Case, async: true

  alias MicrogradEx.Datasets
  alias MicrogradEx.Losses
  alias MicrogradEx.NN
  alias MicrogradEx.NN.MLP
  alias MicrogradEx.Trainer

  describe "official_micrograd_learning_rate/1" do
    test "matches the official demo schedule" do
      assert Trainer.official_micrograd_learning_rate(0) == 1.0
      assert Trainer.official_micrograd_learning_rate(50) == 0.55
      assert_in_delta Trainer.official_micrograd_learning_rate(99), 0.109, 1.0e-12
      assert_in_delta Trainer.official_micrograd_learning_rate(100), 0.1, 1.0e-12
    end
  end

  describe "train/3" do
    test "returns exactly one history row per step by default" do
      model = MLP.new(2, [4, 1], seed: {1, 2, 3})
      dataset = Datasets.moons(12, noise: 0.0, seed: {4, 5, 6})

      run = Trainer.train(model, dataset, steps: 5, learning_rate: 0.05)

      assert %Trainer.Run{} = run
      assert length(run.history) == 5
      assert Enum.map(run.history, & &1.step) == [0, 1, 2, 3, 4]
      assert is_float(run.final_loss)
      assert is_float(run.final_accuracy)
      assert run.options.steps == 5
    end

    test "history includes loss, data loss, regularization, accuracy, and learning rate" do
      model = MLP.new(2, [4, 1], seed: {1, 2, 3})
      dataset = Datasets.moons(12, noise: 0.0, seed: {4, 5, 6})

      %{history: [row | _]} = Trainer.train(model, dataset, steps: 1, learning_rate: 0.05)

      assert %{
               step: 0,
               loss: loss,
               data_loss: data_loss,
               reg_loss: reg_loss,
               accuracy: accuracy,
               learning_rate: 0.05
             } = row

      assert is_float(loss)
      assert is_float(data_loss)
      assert is_float(reg_loss)
      assert is_float(accuracy)
      assert accuracy >= 0.0
      assert accuracy <= 1.0
    end

    test "preserves parameter count and returns a new final model" do
      model = MLP.new(2, [4, 4, 1], seed: {1, 2, 3})
      dataset = Datasets.moons(20, noise: 0.0, seed: {4, 5, 6})

      run = Trainer.train(model, dataset, steps: 3, learning_rate: 0.05)

      assert NN.parameter_count(run.initial_model) == NN.parameter_count(run.final_model)
      assert run.initial_model != run.final_model
    end

    test "the official demo model shape has 337 parameters" do
      model = MLP.new(2, [16, 16, 1], seed: {1337, 1337, 1337})

      assert NN.parameter_count(model) == 337
    end

    test "reduces loss and improves accuracy on deterministic moons" do
      dataset = Datasets.moons(30, noise: 0.02, seed: {4, 5, 6})
      model = MLP.new(2, [8, 8, 1], seed: {1, 2, 3})
      initial = Losses.max_margin(model, dataset.xs, dataset.ys)

      run =
        Trainer.train(model, dataset,
          steps: 35,
          learning_rate: fn step -> 0.35 - 0.25 * step / 35.0 end,
          alpha: 1.0e-4
        )

      assert run.final_loss < initial.total_loss.data
      assert run.final_accuracy >= initial.accuracy
      assert run.final_accuracy >= 0.75
    end

    test "is deterministic with fixed dataset and model seeds" do
      dataset = Datasets.moons(20, noise: 0.05, seed: {4, 5, 6})
      first_model = MLP.new(2, [4, 4, 1], seed: {1, 2, 3})
      second_model = MLP.new(2, [4, 4, 1], seed: {1, 2, 3})

      first = Trainer.train(first_model, dataset, steps: 8, learning_rate: 0.1, seed: {7, 8, 9})
      second = Trainer.train(second_model, dataset, steps: 8, learning_rate: 0.1, seed: {7, 8, 9})

      assert first.history == second.history

      assert first.final_model |> NN.parameters() |> Enum.map(& &1.data) ==
               second.final_model |> NN.parameters() |> Enum.map(& &1.data)
    end

    test "supports raw {xs, ys} examples" do
      model = MLP.new(2, [4, 1], seed: {1, 2, 3})
      dataset = Datasets.moons(12, noise: 0.0, seed: {4, 5, 6})

      run = Trainer.train(model, {dataset.xs, dataset.ys}, steps: 2, learning_rate: 0.05)

      assert length(run.history) == 2
    end

    test "always logs the final step when log_every skips it" do
      model = MLP.new(2, [4, 1], seed: {1, 2, 3})
      dataset = Datasets.moons(12, noise: 0.0, seed: {4, 5, 6})

      run = Trainer.train(model, dataset, steps: 5, learning_rate: 0.05, log_every: 3)

      assert Enum.map(run.history, & &1.step) == [0, 3, 4]
      assert run.final_loss == List.last(run.history).loss
    end

    test "rejects invalid trainer options" do
      model = MLP.new(2, [4, 1], seed: {1, 2, 3})
      dataset = Datasets.moons(12, noise: 0.0, seed: {4, 5, 6})

      assert_raise ArgumentError, ~r/steps/, fn ->
        Trainer.train(model, dataset, steps: 0)
      end

      assert_raise ArgumentError, ~r/log_every/, fn ->
        Trainer.train(model, dataset, log_every: 0)
      end

      assert_raise ArgumentError, ~r/learning_rate/, fn ->
        Trainer.train(model, dataset, learning_rate: -0.1)
      end
    end
  end
end
