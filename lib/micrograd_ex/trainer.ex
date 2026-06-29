defmodule MicrogradEx.Trainer.Run do
  @moduledoc """
  Result of a MicrogradEx training run.
  """

  @enforce_keys [
    :initial_model,
    :final_model,
    :history,
    :final_loss,
    :final_accuracy,
    :options
  ]
  defstruct [
    :initial_model,
    :final_model,
    :history,
    :final_loss,
    :final_accuracy,
    :options
  ]

  @type history_row :: %{
          step: non_neg_integer(),
          loss: float(),
          data_loss: float(),
          reg_loss: float(),
          accuracy: float(),
          learning_rate: float()
        }

  @type t :: %__MODULE__{
          initial_model: term(),
          final_model: term(),
          history: [history_row()],
          final_loss: float(),
          final_accuracy: float(),
          options: map()
        }
end

defmodule MicrogradEx.Trainer do
  @moduledoc """
  Small immutable training loops for MicrogradEx models.

  `train/3` follows the official micrograd demo rhythm: compute a scalar loss,
  run reverse-mode autodiff, and apply an SGD-style parameter update that
  returns a new model.
  """

  alias MicrogradEx.Datasets.Dataset
  alias MicrogradEx.Losses
  alias MicrogradEx.NN
  alias MicrogradEx.Trainer.Run
  alias MicrogradEx.Value

  @default_seed {1337, 1337, 1337}

  @type dataset_or_examples :: Dataset.t() | {list(), list()}
  @type learning_rate :: number() | (non_neg_integer() -> number())
  @type opts :: [
          steps: pos_integer(),
          alpha: number(),
          batch_size: nil | pos_integer(),
          learning_rate: learning_rate(),
          seed: {integer(), integer(), integer()} | nil,
          log_every: pos_integer(),
          loss: :max_margin
        ]

  @doc """
  The learning-rate schedule used by the official micrograd demo.

  It preserves the original formula exactly:

      1.0 - 0.9 * k / 100

  ## Examples

      iex> MicrogradEx.Trainer.official_micrograd_learning_rate(0)
      1.0
      iex> MicrogradEx.Trainer.official_micrograd_learning_rate(100)
      0.1
  """
  @spec official_micrograd_learning_rate(non_neg_integer()) :: float()
  def official_micrograd_learning_rate(step) when is_integer(step) and step >= 0 do
    1.0 - 0.9 * step / 100.0
  end

  def official_micrograd_learning_rate(step) do
    raise ArgumentError,
          "expected step to be a non-negative integer, got: #{inspect(step)}"
  end

  @doc """
  Trains a model on a dataset or `{xs, ys}` examples.

  Options:

    * `:steps` - positive integer, default `100`
    * `:alpha` - max-margin L2 regularization scale, default `1.0e-4`
    * `:batch_size` - `nil` for full-batch or a positive integer
    * `:learning_rate` - non-negative number or one-argument function
    * `:seed` - deterministic mini-batch seed, default `{1337, 1337, 1337}`
    * `:log_every` - positive integer history cadence, default `1`
    * `:loss` - currently only `:max_margin`

  The final step is always present in history even when `:log_every` would
  otherwise skip it.
  """
  @spec train(NN.model(), dataset_or_examples(), opts()) :: Run.t()
  def train(model, dataset_or_examples, opts \\ []) do
    {xs, ys} = unpack_examples!(dataset_or_examples)

    steps = opts |> Keyword.get(:steps, 100) |> validate_steps!()
    alpha = Keyword.get(opts, :alpha, 1.0e-4)
    batch_size = Keyword.get(opts, :batch_size, nil)
    learning_rate = Keyword.get(opts, :learning_rate, &official_micrograd_learning_rate/1)
    seed = opts |> Keyword.get(:seed, @default_seed) |> validate_seed!()
    log_every = opts |> Keyword.get(:log_every, 1) |> validate_log_every!()
    loss = opts |> Keyword.get(:loss, :max_margin) |> validate_loss!()

    {final_model, history} =
      Enum.reduce(0..(steps - 1)//1, {model, []}, fn step, {current_model, history} ->
        lr = learning_rate_for_step(learning_rate, step)

        loss_result =
          compute_loss(loss, current_model, xs, ys,
            alpha: alpha,
            batch_size: batch_size,
            seed: step_seed(seed, step)
          )

        gradients = Value.backward(loss_result.total_loss)
        next_model = NN.apply_gradients(current_model, gradients, lr)

        row = %{
          step: step,
          loss: loss_result.total_loss.data,
          data_loss: loss_result.data_loss.data,
          reg_loss: loss_result.reg_loss.data,
          accuracy: loss_result.accuracy,
          learning_rate: lr
        }

        history =
          if log_step?(step, steps, log_every) do
            [row | history]
          else
            history
          end

        {next_model, history}
      end)

    history = Enum.reverse(history)
    final_row = List.last(history)

    %Run{
      initial_model: model,
      final_model: final_model,
      history: history,
      final_loss: final_row.loss,
      final_accuracy: final_row.accuracy,
      options: %{
        steps: steps,
        alpha: alpha,
        batch_size: batch_size,
        seed: seed,
        log_every: log_every,
        loss: loss
      }
    }
  end

  defp compute_loss(:max_margin, model, xs, ys, opts), do: Losses.max_margin(model, xs, ys, opts)

  defp unpack_examples!(%Dataset{xs: xs, ys: ys}), do: {xs, ys}
  defp unpack_examples!({xs, ys}), do: {xs, ys}

  defp unpack_examples!(other) do
    raise ArgumentError,
          "expected a MicrogradEx.Datasets.Dataset or {xs, ys} tuple, got: #{inspect(other)}"
  end

  defp log_step?(step, steps, log_every) do
    rem(step, log_every) == 0 or step == steps - 1
  end

  defp learning_rate_for_step(learning_rate, _step) when is_number(learning_rate) do
    validate_learning_rate_value!(learning_rate)
  end

  defp learning_rate_for_step(learning_rate, step) when is_function(learning_rate, 1) do
    learning_rate
    |> then(& &1.(step))
    |> validate_learning_rate_value!()
  end

  defp learning_rate_for_step(learning_rate, _step) do
    raise ArgumentError,
          "expected :learning_rate to be a non-negative number or one-argument function, got: #{inspect(learning_rate)}"
  end

  defp validate_learning_rate_value!(learning_rate)
       when is_number(learning_rate) and learning_rate >= 0.0,
       do: learning_rate * 1.0

  defp validate_learning_rate_value!(learning_rate) do
    raise ArgumentError,
          "expected :learning_rate to be non-negative, got: #{inspect(learning_rate)}"
  end

  defp validate_steps!(steps) when is_integer(steps) and steps > 0, do: steps

  defp validate_steps!(steps) do
    raise ArgumentError,
          "expected :steps to be a positive integer, got: #{inspect(steps)}"
  end

  defp validate_log_every!(log_every) when is_integer(log_every) and log_every > 0,
    do: log_every

  defp validate_log_every!(log_every) do
    raise ArgumentError,
          "expected :log_every to be a positive integer, got: #{inspect(log_every)}"
  end

  defp validate_loss!(:max_margin), do: :max_margin

  defp validate_loss!(loss) do
    raise ArgumentError,
          "expected :loss to be :max_margin, got: #{inspect(loss)}"
  end

  defp validate_seed!(nil), do: nil

  defp validate_seed!({a, b, c} = seed)
       when is_integer(a) and is_integer(b) and is_integer(c),
       do: seed

  defp validate_seed!(seed) do
    raise ArgumentError,
          "expected :seed to be nil or a three-integer tuple such as {1, 2, 3}, got: #{inspect(seed)}"
  end

  defp step_seed(nil, _step), do: nil

  defp step_seed({a, b, c}, step) do
    {a + step, b + step * 31, c + step * 131}
  end
end
