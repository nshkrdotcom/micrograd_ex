defmodule MicrogradEx.Datasets.Dataset do
  @moduledoc """
  A small supervised two-dimensional classification dataset.

  The duplicated shapes are intentional: `xs` and `ys` are convenient for
  training, while `points` are convenient for Livebook tables and charts.
  """

  @enforce_keys [:xs, :ys, :points, :metadata]
  defstruct [:xs, :ys, :points, :metadata]

  @type label :: float()

  @type point :: %{
          x: float(),
          y: float(),
          label: label()
        }

  @type input :: [float()]

  @type t :: %__MODULE__{
          xs: [input()],
          ys: [label()],
          points: [point()],
          metadata: map()
        }
end

defmodule MicrogradEx.Datasets.Random do
  @moduledoc false

  @algorithm :exsss

  def state_from_seed(nil), do: nil

  def state_from_seed({a, b, c} = seed)
      when is_integer(a) and is_integer(b) and is_integer(c) do
    :rand.seed_s(@algorithm, seed)
  end

  def state_from_seed(seed) do
    raise ArgumentError,
          "expected :seed to be nil or a three-integer tuple such as {1, 2, 3}, got: #{inspect(seed)}"
  end

  def uniform(nil), do: {:rand.uniform(), nil}
  def uniform(state), do: :rand.uniform_s(state)
end

defmodule MicrogradEx.Datasets do
  @moduledoc """
  Small deterministic two-dimensional datasets for MicrogradEx demos.

  These helpers intentionally avoid Python, sklearn, NumPy, Nx, and external ML
  dependencies. They return plain Elixir data structures suitable for scalar
  micrograd-style training and Livebook visualization.
  """

  alias MicrogradEx.Datasets.Dataset
  alias MicrogradEx.Datasets.Random

  @default_seed {1337, 1337, 1337}
  @default_noise 0.1
  @default_shuffle true

  @type seed :: {integer(), integer(), integer()} | nil
  @type opts :: [
          noise: number(),
          seed: seed(),
          shuffle: boolean()
        ]

  @doc """
  Builds a deterministic two-moons classification dataset.

  This is a pure-Elixir equivalent of the dataset used by the official
  micrograd demo. Labels are returned as `-1.0` and `1.0`, matching the
  max-margin classification loss used by the training notebook.

  ## Options

    * `:noise` - non-negative Gaussian noise scale, default `0.1`
    * `:seed` - deterministic random seed tuple, default `{1337, 1337, 1337}`
    * `:shuffle` - whether to deterministically shuffle rows, default `true`

  ## Example

      iex> dataset = MicrogradEx.Datasets.moons(4, noise: 0.0, shuffle: false)
      iex> length(dataset.xs)
      4
  """
  @spec moons(pos_integer(), opts()) :: Dataset.t()
  def moons(n_samples, opts \\ []) do
    n_samples = validate_n_samples!(n_samples)
    noise = opts |> Keyword.get(:noise, @default_noise) |> validate_noise!()
    seed = opts |> Keyword.get(:seed, @default_seed) |> validate_seed!()
    shuffle? = opts |> Keyword.get(:shuffle, @default_shuffle) |> validate_shuffle!()

    state = Random.state_from_seed(seed)
    {first_count, second_count} = class_counts(n_samples)

    {first_points, state} =
      first_count
      |> moon_angles()
      |> Enum.map_reduce(state, fn theta, state ->
        noisy_point(:math.cos(theta), :math.sin(theta), -1.0, noise, state)
      end)

    {second_points, state} =
      second_count
      |> moon_angles()
      |> Enum.map_reduce(state, fn theta, state ->
        noisy_point(
          1.0 - :math.cos(theta),
          1.0 - :math.sin(theta) - 0.5,
          1.0,
          noise,
          state
        )
      end)

    {points, _state} = maybe_shuffle(first_points ++ second_points, state, shuffle?)

    build_dataset(:moons, points,
      n_samples: n_samples,
      noise: noise,
      seed: seed,
      shuffle: shuffle?
    )
  end

  @doc """
  Builds a deterministic two-class spiral classification dataset.

  The spiral dataset is useful for showing non-linear decision boundaries. Two
  classes share an increasing radius and are separated by a phase shift of
  `pi`.

  ## Options

    * `:noise` - non-negative Gaussian noise scale, default `0.1`
    * `:seed` - deterministic random seed tuple, default `{1337, 1337, 1337}`
    * `:shuffle` - whether to deterministically shuffle rows, default `true`
    * `:turns` - positive number of turns, default `1.5`

  ## Example

      iex> dataset = MicrogradEx.Datasets.spiral(6, noise: 0.0, shuffle: false)
      iex> Enum.uniq(dataset.ys)
      [-1.0, 1.0]
  """
  @spec spiral(pos_integer(), Keyword.t()) :: Dataset.t()
  def spiral(n_samples, opts \\ []) do
    n_samples = validate_n_samples!(n_samples)
    noise = opts |> Keyword.get(:noise, @default_noise) |> validate_noise!()
    seed = opts |> Keyword.get(:seed, @default_seed) |> validate_seed!()
    shuffle? = opts |> Keyword.get(:shuffle, @default_shuffle) |> validate_shuffle!()
    turns = opts |> Keyword.get(:turns, 1.5) |> validate_turns!()

    state = Random.state_from_seed(seed)
    {first_count, second_count} = class_counts(n_samples)

    {first_points, state} =
      spiral_points(first_count, -1.0, 0.0, turns, noise, state)

    {second_points, state} =
      spiral_points(second_count, 1.0, :math.pi(), turns, noise, state)

    {points, _state} = maybe_shuffle(first_points ++ second_points, state, shuffle?)

    build_dataset(:spiral, points,
      n_samples: n_samples,
      noise: noise,
      seed: seed,
      shuffle: shuffle?,
      turns: turns
    )
  end

  @doc """
  Builds a deterministic two-class blob dataset.

  This simple baseline is useful for sanity-checking classification and
  visualization code before moving to moons or spirals.

  ## Options

    * `:noise` - non-negative Gaussian noise scale, default `0.1`
    * `:seed` - deterministic random seed tuple, default `{1337, 1337, 1337}`
    * `:shuffle` - whether to deterministically shuffle rows, default `true`
    * `:centers` - two `{x, y}` class centers, default `[{-1.0, 0.0}, {1.0, 0.0}]`

  ## Example

      iex> dataset = MicrogradEx.Datasets.blobs(4, noise: 0.0, shuffle: false)
      iex> length(dataset.points)
      4
  """
  @spec blobs(pos_integer(), Keyword.t()) :: Dataset.t()
  def blobs(n_samples, opts \\ []) do
    n_samples = validate_n_samples!(n_samples)
    noise = opts |> Keyword.get(:noise, @default_noise) |> validate_noise!()
    seed = opts |> Keyword.get(:seed, @default_seed) |> validate_seed!()
    shuffle? = opts |> Keyword.get(:shuffle, @default_shuffle) |> validate_shuffle!()
    centers = opts |> Keyword.get(:centers, [{-1.0, 0.0}, {1.0, 0.0}]) |> validate_centers!()

    state = Random.state_from_seed(seed)
    {first_count, second_count} = class_counts(n_samples)
    [{x0, y0}, {x1, y1}] = centers

    {first_points, state} =
      repeated_noisy_points(first_count, x0, y0, -1.0, noise, state)

    {second_points, state} =
      repeated_noisy_points(second_count, x1, y1, 1.0, noise, state)

    {points, _state} = maybe_shuffle(first_points ++ second_points, state, shuffle?)

    build_dataset(:blobs, points,
      n_samples: n_samples,
      noise: noise,
      seed: seed,
      shuffle: shuffle?,
      centers: centers
    )
  end

  defp class_counts(n_samples) do
    first_count = div(n_samples, 2)
    {first_count, n_samples - first_count}
  end

  defp moon_angles(1), do: [:math.pi() / 2.0]

  defp moon_angles(count) do
    for i <- 0..(count - 1) do
      :math.pi() * i / (count - 1)
    end
  end

  defp spiral_points(count, label, phase, turns, noise, state) do
    Enum.map_reduce(0..(count - 1)//1, state, fn index, state ->
      radius = if count == 1, do: 1.0, else: index / (count - 1)
      theta = turns * 2.0 * :math.pi() * radius + phase

      noisy_point(
        radius * :math.cos(theta),
        radius * :math.sin(theta),
        label,
        noise,
        state
      )
    end)
  end

  defp repeated_noisy_points(count, x, y, label, noise, state) do
    Enum.map_reduce(1..count//1, state, fn _index, state ->
      noisy_point(x, y, label, noise, state)
    end)
  end

  defp noisy_point(x, y, label, noise, state) when noise == 0.0,
    do: {point(x, y, label), state}

  defp noisy_point(x, y, label, noise, state) do
    {x_noise, state} = normal(state)
    {y_noise, state} = normal(state)

    {point(x + noise * x_noise, y + noise * y_noise, label), state}
  end

  defp point(x, y, label) do
    %{x: x * 1.0, y: y * 1.0, label: label * 1.0}
  end

  defp normal(state) do
    {u1, state} = uniform_nonzero(state)
    {u2, state} = Random.uniform(state)

    radius = :math.sqrt(-2.0 * :math.log(u1))
    theta = 2.0 * :math.pi() * u2

    {radius * :math.cos(theta), state}
  end

  defp uniform_nonzero(state) do
    {u, state} = Random.uniform(state)

    if u <= 0.0 do
      uniform_nonzero(state)
    else
      {u, state}
    end
  end

  defp maybe_shuffle(points, state, false), do: {points, state}

  defp maybe_shuffle(points, state, true) do
    {keyed_points, state} =
      points
      |> Enum.with_index()
      |> Enum.map_reduce(state, fn {point, index}, state ->
        {key, state} = Random.uniform(state)
        {{key, index, point}, state}
      end)

    points =
      keyed_points
      |> Enum.sort_by(fn {key, index, _point} -> {key, index} end)
      |> Enum.map(fn {_key, _index, point} -> point end)

    {points, state}
  end

  defp build_dataset(name, points, metadata_opts) do
    xs = Enum.map(points, fn %{x: x, y: y} -> [x, y] end)
    ys = Enum.map(points, & &1.label)

    %Dataset{
      xs: xs,
      ys: ys,
      points: points,
      metadata:
        metadata_opts
        |> Map.new()
        |> Map.put(:name, name)
    }
  end

  defp validate_n_samples!(n_samples) when is_integer(n_samples) and n_samples > 1,
    do: n_samples

  defp validate_n_samples!(n_samples) do
    raise ArgumentError,
          "expected n_samples to be an integer greater than 1, got: #{inspect(n_samples)}"
  end

  defp validate_noise!(noise) when is_number(noise) and noise >= 0.0, do: noise * 1.0

  defp validate_noise!(noise) do
    raise ArgumentError,
          "expected :noise to be a non-negative number, got: #{inspect(noise)}"
  end

  defp validate_seed!(nil), do: nil

  defp validate_seed!({a, b, c} = seed)
       when is_integer(a) and is_integer(b) and is_integer(c),
       do: seed

  defp validate_seed!(seed) do
    raise ArgumentError,
          "expected :seed to be nil or a three-integer tuple such as {1, 2, 3}, got: #{inspect(seed)}"
  end

  defp validate_shuffle!(shuffle?) when is_boolean(shuffle?), do: shuffle?

  defp validate_shuffle!(shuffle?) do
    raise ArgumentError,
          "expected :shuffle to be true or false, got: #{inspect(shuffle?)}"
  end

  defp validate_turns!(turns) when is_number(turns) and turns > 0.0, do: turns * 1.0

  defp validate_turns!(turns) do
    raise ArgumentError,
          "expected :turns to be a positive number, got: #{inspect(turns)}"
  end

  defp validate_centers!([{x0, y0}, {x1, y1}] = centers)
       when is_number(x0) and is_number(y0) and is_number(x1) and is_number(y1) do
    Enum.map(centers, fn {x, y} -> {x * 1.0, y * 1.0} end)
  end

  defp validate_centers!(centers) do
    raise ArgumentError,
          "expected :centers to be two {x, y} numeric tuples, got: #{inspect(centers)}"
  end
end
