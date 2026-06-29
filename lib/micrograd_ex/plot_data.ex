defmodule MicrogradEx.PlotData do
  @moduledoc """
  Converts MicrogradEx datasets and training runs into plain plotting rows.

  The returned values are ordinary lists of maps. They are designed to be
  consumed by Livebook, Vega-Lite, CSV exporters, or tests without making the
  core library depend on any plotting package.
  """

  alias MicrogradEx.Datasets.Dataset
  alias MicrogradEx.NN
  alias MicrogradEx.Trainer.Run
  alias MicrogradEx.Value

  @doc """
  Converts dataset points into chart-friendly rows.

  Numeric labels are preserved as `:label_value`, while `:label` is a friendly
  legend string.
  """
  @spec dataset_points(Dataset.t()) :: [map()]
  def dataset_points(%Dataset{points: points}) do
    Enum.map(points, fn %{x: x, y: y, label: label} ->
      %{
        x: x * 1.0,
        y: y * 1.0,
        label: label_name(label),
        label_value: label * 1.0
      }
    end)
  end

  def dataset_points(other) do
    raise ArgumentError,
          "expected a MicrogradEx.Datasets.Dataset, got: #{inspect(other)}"
  end

  @doc """
  Returns full training-history rows with percentage accuracy added.
  """
  @spec training_history(Run.t()) :: [map()]
  def training_history(%Run{history: history}) do
    Enum.map(history, fn row ->
      Map.put(row, :accuracy_percent, row.accuracy * 100.0)
    end)
  end

  def training_history(other) do
    raise ArgumentError,
          "expected a MicrogradEx.Trainer.Run, got: #{inspect(other)}"
  end

  @doc """
  Expands training history into loss metric rows.
  """
  @spec loss_history(Run.t()) :: [map()]
  def loss_history(%Run{history: history}) do
    Enum.flat_map(history, fn row ->
      [
        %{step: row.step, metric: "total loss", value: row.loss},
        %{step: row.step, metric: "data loss", value: row.data_loss},
        %{step: row.step, metric: "regularization loss", value: row.reg_loss}
      ]
    end)
  end

  def loss_history(other) do
    raise ArgumentError,
          "expected a MicrogradEx.Trainer.Run, got: #{inspect(other)}"
  end

  @doc """
  Converts training accuracy into percentage rows for charts.
  """
  @spec accuracy_history(Run.t()) :: [map()]
  def accuracy_history(%Run{history: history}) do
    Enum.map(history, fn row ->
      %{step: row.step, metric: "accuracy", value: row.accuracy * 100.0}
    end)
  end

  def accuracy_history(other) do
    raise ArgumentError,
          "expected a MicrogradEx.Trainer.Run, got: #{inspect(other)}"
  end

  @doc """
  Evaluates a model over a padded two-dimensional grid.

  Options:

    * `:h` - positive grid spacing, default `0.25`
    * `:padding` - non-negative padding around dataset bounds, default `1.0`
  """
  @spec decision_boundary(NN.model(), Dataset.t(), Keyword.t()) :: [map()]
  def decision_boundary(model, dataset, opts \\ [])

  def decision_boundary(model, %Dataset{} = dataset, opts) do
    h = opts |> Keyword.get(:h, 0.25) |> validate_h!()
    padding = opts |> Keyword.get(:padding, 1.0) |> validate_padding!()
    {x_min, x_max, y_min, y_max} = bounds(dataset, padding)

    for x <- range(x_min, x_max, h),
        y <- range(y_min, y_max, h) do
      score = scalar_score!(NN.forward(model, [x, y])).data
      predicted_value = if score > 0.0, do: 1.0, else: -1.0

      %{
        x: x,
        y: y,
        predicted: label_name(predicted_value),
        predicted_value: predicted_value,
        score: score
      }
    end
  end

  def decision_boundary(_model, dataset, _opts) do
    raise ArgumentError,
          "expected a MicrogradEx.Datasets.Dataset, got: #{inspect(dataset)}"
  end

  defp scalar_score!(%Value{} = score), do: score

  defp scalar_score!(score) do
    raise ArgumentError,
          "expected model to return a scalar %MicrogradEx.Value{}, got: #{inspect(score)}"
  end

  defp bounds(%Dataset{points: points}, padding) do
    xs = Enum.map(points, & &1.x)
    ys = Enum.map(points, & &1.y)

    {
      Enum.min(xs) - padding,
      Enum.max(xs) + padding,
      Enum.min(ys) - padding,
      Enum.max(ys) + padding
    }
  end

  defp range(min, max, step) do
    count = Kernel.max(0, floor((max - min) / step))

    for i <- 0..count//1 do
      min + i * step
    end
  end

  defp validate_h!(h) when is_number(h) and h > 0.0, do: h * 1.0

  defp validate_h!(h) do
    raise ArgumentError,
          "expected :h to be a positive number, got: #{inspect(h)}"
  end

  defp validate_padding!(padding) when is_number(padding) and padding >= 0.0,
    do: padding * 1.0

  defp validate_padding!(padding) do
    raise ArgumentError,
          "expected :padding to be a non-negative number, got: #{inspect(padding)}"
  end

  defp label_name(label) when label < 0.0, do: "class -1"
  defp label_name(_label), do: "class 1"
end
