defmodule MicrogradEx.DatasetsTest do
  use ExUnit.Case, async: true

  alias MicrogradEx.Datasets
  alias MicrogradEx.Datasets.Dataset

  describe "moons/2" do
    test "returns requested sample count with matching xs ys and points" do
      dataset = Datasets.moons(100, seed: {1, 2, 3})

      assert %Dataset{} = dataset
      assert length(dataset.xs) == 100
      assert length(dataset.ys) == 100
      assert length(dataset.points) == 100
      assert dataset.metadata.name == :moons
      assert dataset.metadata.n_samples == 100
    end

    test "returns two-dimensional numeric inputs" do
      dataset = Datasets.moons(25, seed: {1, 2, 3})

      assert Enum.all?(dataset.xs, fn row ->
               is_list(row) and length(row) == 2 and Enum.all?(row, &is_float/1)
             end)
    end

    test "returns only -1.0 and 1.0 labels" do
      dataset = Datasets.moons(100, seed: {1, 2, 3})

      assert dataset.ys |> Enum.uniq() |> Enum.sort() == [-1.0, 1.0]
      assert Enum.all?(dataset.points, &(&1.label in [-1.0, 1.0]))
    end

    test "is deterministic with the same seed" do
      first = Datasets.moons(100, seed: {1, 2, 3})
      second = Datasets.moons(100, seed: {1, 2, 3})

      assert first.xs == second.xs
      assert first.ys == second.ys
      assert first.points == second.points
      assert first.metadata == second.metadata
    end

    test "changes with a different seed" do
      first = Datasets.moons(100, seed: {1, 2, 3})
      second = Datasets.moons(100, seed: {3, 2, 1})

      assert first.xs != second.xs
    end

    test "noise changes coordinates without changing labels" do
      clean = Datasets.moons(100, noise: 0.0, seed: {1, 2, 3}, shuffle: false)
      noisy = Datasets.moons(100, noise: 0.1, seed: {1, 2, 3}, shuffle: false)

      assert clean.xs != noisy.xs
      assert clean.ys == noisy.ys
    end

    test "supports odd sample counts" do
      dataset = Datasets.moons(101, seed: {1, 2, 3})

      assert length(dataset.xs) == 101
      assert length(dataset.ys) == 101
      assert Enum.count(dataset.ys, &(&1 == -1.0)) == 50
      assert Enum.count(dataset.ys, &(&1 == 1.0)) == 51
    end

    test "can disable shuffling" do
      dataset = Datasets.moons(6, noise: 0.0, seed: {1, 2, 3}, shuffle: false)

      assert dataset.ys == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    end

    test "rejects invalid n_samples" do
      for invalid <- [0, 1, -10, 100.0, "100"] do
        assert_raise ArgumentError, ~r/n_samples/, fn ->
          Datasets.moons(invalid)
        end
      end
    end

    test "rejects invalid noise" do
      for invalid <- [-0.1, "0.1"] do
        assert_raise ArgumentError, ~r/noise/, fn ->
          Datasets.moons(10, noise: invalid)
        end
      end
    end

    test "rejects invalid seed" do
      for invalid <- [{1, 2}, {1, 2, 3.0}, "seed"] do
        assert_raise ArgumentError, ~r/seed/, fn ->
          Datasets.moons(10, seed: invalid)
        end
      end
    end

    test "rejects invalid shuffle" do
      assert_raise ArgumentError, ~r/shuffle/, fn ->
        Datasets.moons(10, shuffle: :yes)
      end
    end
  end

  describe "spiral/2" do
    test "returns requested sample count with two-dimensional inputs and signed labels" do
      dataset = Datasets.spiral(101, seed: {1, 2, 3})

      assert %Dataset{} = dataset
      assert length(dataset.xs) == 101
      assert length(dataset.ys) == 101

      assert Enum.all?(dataset.xs, fn row ->
               is_list(row) and length(row) == 2 and Enum.all?(row, &is_float/1)
             end)

      assert dataset.ys |> Enum.uniq() |> Enum.sort() == [-1.0, 1.0]
      assert dataset.metadata.name == :spiral
    end

    test "is deterministic with the same seed" do
      first = Datasets.spiral(100, seed: {1, 2, 3})
      second = Datasets.spiral(100, seed: {1, 2, 3})

      assert first.xs == second.xs
      assert first.ys == second.ys
      assert first.points == second.points
    end

    test "changes with a different seed" do
      first = Datasets.spiral(100, seed: {1, 2, 3})
      second = Datasets.spiral(100, seed: {3, 2, 1})

      assert first.xs != second.xs
    end

    test "can disable shuffling" do
      dataset = Datasets.spiral(6, noise: 0.0, seed: {1, 2, 3}, shuffle: false)

      assert dataset.ys == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    end

    test "rejects invalid options" do
      assert_raise ArgumentError, ~r/n_samples/, fn -> Datasets.spiral(1) end
      assert_raise ArgumentError, ~r/noise/, fn -> Datasets.spiral(10, noise: -1.0) end
      assert_raise ArgumentError, ~r/seed/, fn -> Datasets.spiral(10, seed: :bad) end
      assert_raise ArgumentError, ~r/shuffle/, fn -> Datasets.spiral(10, shuffle: :bad) end
      assert_raise ArgumentError, ~r/turns/, fn -> Datasets.spiral(10, turns: 0.0) end
    end
  end

  describe "blobs/2" do
    test "returns deterministic clustered examples" do
      first = Datasets.blobs(50, seed: {1, 2, 3})
      second = Datasets.blobs(50, seed: {1, 2, 3})

      assert %Dataset{} = first
      assert length(first.xs) == 50
      assert first.xs == second.xs
      assert first.ys == second.ys
      assert first.metadata.name == :blobs
    end
  end
end
