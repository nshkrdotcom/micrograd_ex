defmodule MicrogradEx.PlotDataTest do
  use ExUnit.Case, async: true

  alias MicrogradEx.Datasets
  alias MicrogradEx.NN.MLP
  alias MicrogradEx.PlotData
  alias MicrogradEx.Trainer

  describe "dataset_points/1" do
    test "returns one row per dataset point" do
      dataset = Datasets.moons(20, seed: {1, 2, 3})

      rows = PlotData.dataset_points(dataset)

      assert length(rows) == length(dataset.points)
    end

    test "includes x y label and label_value" do
      dataset = Datasets.moons(4, noise: 0.0, shuffle: false)

      [row | _] = PlotData.dataset_points(dataset)

      assert %{x: x, y: y, label: "class -1", label_value: -1.0} = row
      assert is_float(x)
      assert is_float(y)
    end
  end

  describe "training_history/1" do
    test "returns one row per logged training step" do
      run = tiny_run()

      rows = PlotData.training_history(run)

      assert length(rows) == length(run.history)
      assert Enum.map(rows, & &1.step) == Enum.map(run.history, & &1.step)
    end

    test "adds accuracy_percent" do
      [row | _] = tiny_run() |> PlotData.training_history()

      assert row.accuracy_percent == row.accuracy * 100.0
    end
  end

  describe "loss_history/1" do
    test "returns total data and regularization loss rows" do
      run = tiny_run()

      rows = PlotData.loss_history(run)

      assert length(rows) == length(run.history) * 3

      assert rows |> Enum.map(& &1.metric) |> Enum.uniq() |> Enum.sort() == [
               "data loss",
               "regularization loss",
               "total loss"
             ]

      assert Enum.all?(rows, &is_float(&1.value))
    end
  end

  describe "accuracy_history/1" do
    test "returns percentage accuracy rows" do
      run = tiny_run()

      rows = PlotData.accuracy_history(run)

      assert length(rows) == length(run.history)
      assert Enum.all?(rows, &(&1.metric == "accuracy"))
      assert Enum.all?(rows, &(&1.value >= 0.0 and &1.value <= 100.0))
    end
  end

  describe "decision_boundary/3" do
    test "returns nonempty grid rows" do
      {model, dataset} = boundary_fixture()

      rows = PlotData.decision_boundary(model, dataset, h: 1.0, padding: 0.5)

      assert rows != []
    end

    test "grid rows include x y score and predicted class" do
      {model, dataset} = boundary_fixture()

      [row | _] = PlotData.decision_boundary(model, dataset, h: 1.0, padding: 0.5)

      assert %{x: x, y: y, score: score, predicted: predicted, predicted_value: predicted_value} =
               row

      assert is_float(x)
      assert is_float(y)
      assert is_float(score)
      assert predicted in ["class -1", "class 1"]
      assert predicted_value in [-1.0, 1.0]
    end

    test "grid bounds include dataset with padding" do
      {model, dataset} = boundary_fixture()

      rows = PlotData.decision_boundary(model, dataset, h: 0.5, padding: 0.5)

      grid_xs = Enum.map(rows, & &1.x)
      grid_ys = Enum.map(rows, & &1.y)
      point_xs = Enum.map(dataset.points, & &1.x)
      point_ys = Enum.map(dataset.points, & &1.y)

      assert Enum.min(grid_xs) <= Enum.min(point_xs) - 0.5
      assert Enum.max(grid_xs) >= Enum.max(point_xs)
      assert Enum.min(grid_ys) <= Enum.min(point_ys) - 0.5
      assert Enum.max(grid_ys) >= Enum.max(point_ys)
    end

    test "smaller h creates more rows" do
      {model, dataset} = boundary_fixture()

      coarse = PlotData.decision_boundary(model, dataset, h: 1.0, padding: 0.5)
      fine = PlotData.decision_boundary(model, dataset, h: 0.5, padding: 0.5)

      assert length(fine) > length(coarse)
    end

    test "rejects invalid h" do
      {model, dataset} = boundary_fixture()

      assert_raise ArgumentError, ~r/h/, fn ->
        PlotData.decision_boundary(model, dataset, h: 0.0)
      end
    end

    test "rejects invalid padding" do
      {model, dataset} = boundary_fixture()

      assert_raise ArgumentError, ~r/padding/, fn ->
        PlotData.decision_boundary(model, dataset, padding: -0.1)
      end
    end
  end

  defp tiny_run do
    dataset = Datasets.moons(20, noise: 0.05, seed: {1, 2, 3})
    model = MLP.new(2, [4, 1], seed: {1, 2, 3})

    Trainer.train(model, dataset,
      steps: 3,
      learning_rate: 0.1
    )
  end

  defp boundary_fixture do
    dataset = Datasets.moons(10, noise: 0.0, seed: {1, 2, 3})
    model = MLP.new(2, [4, 1], seed: {1, 2, 3})

    {model, dataset}
  end
end
