defmodule MicrogradEx.Losses.Result do
  @moduledoc """
  Result of evaluating a supervised scalar loss.
  """

  alias MicrogradEx.Value

  @enforce_keys [:total_loss, :data_loss, :reg_loss, :accuracy, :scores]
  defstruct [:total_loss, :data_loss, :reg_loss, :accuracy, :scores]

  @type t :: %__MODULE__{
          total_loss: Value.t(),
          data_loss: Value.t(),
          reg_loss: Value.t(),
          accuracy: float(),
          scores: [Value.t()]
        }
end

defmodule MicrogradEx.Losses do
  @moduledoc """
  Loss functions for small scalar MicrogradEx models.

  The main Livebook demo uses the same max-margin classification objective as
  the official Python micrograd notebook: mean SVM-style hinge loss plus L2
  regularization over model parameters.
  """

  alias MicrogradEx.Losses.Result
  alias MicrogradEx.NN
  alias MicrogradEx.Value

  @default_seed {1337, 1337, 1337}

  @type opts :: [
          alpha: number(),
          batch_size: nil | pos_integer(),
          seed: {integer(), integer(), integer()} | nil
        ]

  @doc """
  Computes max-margin classification loss for signed labels.

  Labels may be `-1`, `-1.0`, `1`, or `1.0`; they are normalized to floats
  internally. The returned losses are `%MicrogradEx.Value{}` structs so callers
  can run `MicrogradEx.Value.backward/1` on `result.total_loss`.

  ## Options

    * `:alpha` - non-negative L2 regularization scale, default `1.0e-4`
    * `:batch_size` - `nil` for full batch or a positive integer
    * `:seed` - deterministic mini-batch seed, default `{1337, 1337, 1337}`

  ## Example

      iex> alias MicrogradEx.{Losses, NN}
      iex> model = MicrogradEx.NN.MLP.new(2, [2, 1], seed: {1, 2, 3})
      iex> result = Losses.max_margin(model, [[-1.0, 0.0], [1.0, 0.0]], [-1.0, 1.0])
      iex> %MicrogradEx.Value{} = result.total_loss
      iex> result.accuracy >= 0.0 and result.accuracy <= 1.0
      true
  """
  @spec max_margin(NN.model(), [[number()]], [number()], opts()) :: Result.t()
  def max_margin(model, xs, ys, opts \\ []) do
    alpha = opts |> Keyword.get(:alpha, 1.0e-4) |> validate_alpha!()
    batch_size = opts |> Keyword.get(:batch_size, nil) |> validate_batch_size!()
    seed = opts |> Keyword.get(:seed, @default_seed) |> validate_seed!()

    examples =
      xs
      |> validate_examples!(ys)
      |> maybe_batch(batch_size, seed)

    scores =
      Enum.map(examples, fn {x, _y} ->
        scalar_score!(NN.forward(model, x))
      end)

    losses =
      examples
      |> Enum.zip(scores)
      |> Enum.map(fn {{_x, y}, score} ->
        score
        |> Value.mul(-y)
        |> Value.add(1.0)
        |> Value.relu()
      end)

    data_loss =
      losses
      |> Value.sum()
      |> Value.mul(1.0 / length(losses))

    reg_loss =
      model
      |> NN.parameters()
      |> Enum.map(&Value.mul(&1, &1))
      |> Value.sum()
      |> Value.mul(alpha)

    total_loss = Value.add(data_loss, reg_loss)

    accuracy =
      examples
      |> Enum.zip(scores)
      |> Enum.count(fn {{_x, y}, score} ->
        (y > 0.0) == (score.data > 0.0)
      end)
      |> Kernel./(length(examples))

    %Result{
      total_loss: total_loss,
      data_loss: data_loss,
      reg_loss: reg_loss,
      accuracy: accuracy,
      scores: scores
    }
  end

  defp scalar_score!(%Value{} = score), do: score

  defp scalar_score!(score) do
    raise ArgumentError,
          "expected model to return a scalar %MicrogradEx.Value{}, got: #{inspect(score)}"
  end

  defp validate_examples!(xs, ys) when is_list(xs) and is_list(ys) do
    if length(xs) != length(ys) do
      raise ArgumentError,
            "expected xs and ys to have the same length, got #{length(xs)} xs and #{length(ys)} ys"
    end

    if xs == [] do
      raise ArgumentError, "expected at least one training example"
    end

    xs
    |> Enum.zip(ys)
    |> Enum.map(fn {x, y} -> {validate_x!(x), validate_label!(y)} end)
  end

  defp validate_examples!(xs, ys) do
    raise ArgumentError,
          "expected xs and ys to be lists, got: #{inspect(xs)} and #{inspect(ys)}"
  end

  defp validate_x!(x) when is_list(x) and x != [] do
    Enum.map(x, fn
      value when is_number(value) ->
        value * 1.0

      value ->
        raise ArgumentError,
              "expected input row values to be numbers, got: #{inspect(value)}"
    end)
  end

  defp validate_x!(x) do
    raise ArgumentError,
          "expected each input row to be a non-empty list of numbers, got: #{inspect(x)}"
  end

  defp validate_label!(label) when label in [-1, -1.0], do: -1.0
  defp validate_label!(label) when label in [1, 1.0], do: 1.0

  defp validate_label!(label) do
    raise ArgumentError,
          "expected labels to be -1.0 or 1.0, got: #{inspect(label)}"
  end

  defp validate_alpha!(alpha) when is_number(alpha) and alpha >= 0.0, do: alpha * 1.0

  defp validate_alpha!(alpha) do
    raise ArgumentError,
          "expected :alpha to be a non-negative number, got: #{inspect(alpha)}"
  end

  defp validate_batch_size!(nil), do: nil

  defp validate_batch_size!(batch_size) when is_integer(batch_size) and batch_size > 0,
    do: batch_size

  defp validate_batch_size!(batch_size) do
    raise ArgumentError,
          "expected :batch_size to be nil or a positive integer, got: #{inspect(batch_size)}"
  end

  defp validate_seed!(nil), do: nil

  defp validate_seed!({a, b, c} = seed)
       when is_integer(a) and is_integer(b) and is_integer(c),
       do: seed

  defp validate_seed!(seed) do
    raise ArgumentError,
          "expected :seed to be nil or a three-integer tuple such as {1, 2, 3}, got: #{inspect(seed)}"
  end

  defp maybe_batch(examples, nil, _seed), do: examples

  defp maybe_batch(examples, batch_size, seed) do
    if batch_size >= length(examples) do
      examples
    else
      examples
      |> deterministic_shuffle(seed)
      |> Enum.take(batch_size)
    end
  end

  defp deterministic_shuffle(examples, seed) do
    state = state_from_seed(seed)

    {keyed_examples, _state} =
      examples
      |> Enum.with_index()
      |> Enum.map_reduce(state, fn {example, index}, state ->
        {key, state} = uniform(state)
        {{key, index, example}, state}
      end)

    keyed_examples
    |> Enum.sort_by(fn {key, index, _example} -> {key, index} end)
    |> Enum.map(fn {_key, _index, example} -> example end)
  end

  defp state_from_seed(nil), do: nil
  defp state_from_seed(seed), do: :rand.seed_s(:exsss, seed)

  defp uniform(nil), do: {:rand.uniform(), nil}
  defp uniform(state), do: :rand.uniform_s(state)
end
