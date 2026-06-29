defmodule MicrogradEx.NN.Random do
  @moduledoc false

  alias MicrogradEx.Value

  @algorithm :exsss

  def state_from_seed(nil), do: nil

  def state_from_seed({a, b, c}) when is_integer(a) and is_integer(b) and is_integer(c) do
    :rand.seed_s(@algorithm, {a, b, c})
  end

  def state_from_seed(seed) do
    raise ArgumentError,
          "expected :seed to be a three-integer tuple such as {1, 2, 3}, got: #{inspect(seed)}"
  end

  def uniform_values(count, state) when count >= 0 do
    Enum.map_reduce(1..count//1, state, fn _index, state ->
      uniform_value(state)
    end)
  end

  defp uniform_value(nil) do
    {Value.new(:rand.uniform() * 2.0 - 1.0), nil}
  end

  defp uniform_value(state) do
    {unit_interval, next_state} = :rand.uniform_s(state)
    {Value.new(unit_interval * 2.0 - 1.0), next_state}
  end
end

defmodule MicrogradEx.NN.Neuron do
  @moduledoc """
  A scalar neuron: weighted sum, bias, and optional ReLU.

  This mirrors `micrograd.nn.Neuron`. The Elixir struct stores immutable
  parameter values. Calling `MicrogradEx.NN.apply_gradients/3` returns a new
  neuron with updated parameter values instead of mutating the existing neuron.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.NN.Random
  alias MicrogradEx.Value

  @enforce_keys [:weights, :bias, :nonlin]
  defstruct [:weights, :bias, :nonlin]

  @type t :: %__MODULE__{
          weights: [Value.t()],
          bias: Value.t(),
          nonlin: boolean()
        }

  @doc """
  Creates a neuron.

  Options:

    * `:nonlin` - when `true`, apply ReLU after the affine transform.
    * `:weights` - exact initial weights, useful for tests and examples.
    * `:bias` - exact initial bias, defaults to `0.0`.
    * `:seed` - `{a, b, c}` tuple for deterministic random weights.
  """
  def new(input_count, opts \\ []) do
    validate_input_count!(input_count)
    state = Random.state_from_seed(Keyword.get(opts, :seed))
    {neuron, _state} = new_with_state(input_count, opts, state)
    neuron
  end

  @doc false
  def new_with_state(input_count, opts, state) do
    validate_input_count!(input_count)

    {weights, state} =
      case Keyword.fetch(opts, :weights) do
        {:ok, weights} ->
          {coerce_weights!(weights, input_count), state}

        :error ->
          Random.uniform_values(input_count, state)
      end

    neuron = %__MODULE__{
      weights: weights,
      bias: opts |> Keyword.get(:bias, 0.0) |> Value.coerce(),
      nonlin: Keyword.get(opts, :nonlin, true)
    }

    {neuron, state}
  end

  @doc """
  Runs a forward pass through the neuron.

  Inputs may be plain numbers, `Value` structs, or a single scalar when the
  neuron has exactly one input. Numbers are promoted to differentiable leaves so
  input gradients can be inspected too.
  """
  def forward(%__MODULE__{} = neuron, inputs) do
    inputs = coerce_inputs!(inputs, length(neuron.weights))

    activation =
      neuron.weights
      |> Enum.zip(inputs)
      |> Enum.reduce(neuron.bias, fn {weight, input}, acc ->
        # Each term is scalar multiplication followed by scalar addition. The
        # graph therefore exposes the same tiny operations as original
        # micrograd rather than hiding this inside a vector primitive.
        acc
        |> Value.add(Value.mul(weight, input))
      end)

    if neuron.nonlin do
      Value.relu(activation)
    else
      activation
    end
  end

  @doc """
  Returns this neuron's trainable parameters in weight-then-bias order.
  """
  def parameters(%__MODULE__{} = neuron), do: neuron.weights ++ [neuron.bias]

  @doc false
  def apply_gradients(%__MODULE__{} = neuron, %Gradients{} = gradients, learning_rate)
      when is_number(learning_rate) do
    %__MODULE__{
      neuron
      | weights: Enum.map(neuron.weights, &step_parameter(&1, gradients, learning_rate)),
        bias: step_parameter(neuron.bias, gradients, learning_rate)
    }
  end

  defp step_parameter(%Value{} = parameter, %Gradients{} = gradients, learning_rate) do
    gradient = Gradients.get(gradients, parameter)
    Value.new(parameter.data - learning_rate * gradient, label: parameter.label)
  end

  defp coerce_weights!(weights, input_count) when is_list(weights) do
    if length(weights) != input_count do
      raise ArgumentError,
            "expected #{input_count} weights, got #{length(weights)}"
    end

    Enum.map(weights, &Value.coerce/1)
  end

  defp coerce_weights!(weights, _input_count) do
    raise ArgumentError, "expected :weights to be a list, got: #{inspect(weights)}"
  end

  defp coerce_inputs!(input, 1) when is_number(input) or is_struct(input, Value) do
    [Value.coerce(input)]
  end

  defp coerce_inputs!(inputs, expected_count) when is_list(inputs) do
    if length(inputs) != expected_count do
      raise ArgumentError,
            "expected #{expected_count} inputs, got #{length(inputs)}"
    end

    Enum.map(inputs, &Value.coerce/1)
  end

  defp coerce_inputs!(inputs, expected_count) do
    raise ArgumentError,
          "expected #{expected_count} inputs as a list, got: #{inspect(inputs)}"
  end

  defp validate_input_count!(input_count)
       when is_integer(input_count) and input_count >= 0,
       do: :ok

  defp validate_input_count!(input_count) do
    raise ArgumentError,
          "expected input_count to be a non-negative integer, got: #{inspect(input_count)}"
  end
end

defmodule MicrogradEx.NN.Layer do
  @moduledoc """
  A layer is a list of neurons with the same input width.

  The original Python `Layer.__call__` returns a single `Value` when the layer
  has one neuron and a list otherwise. This module keeps that convenience in
  `forward/2`, and also exposes `forward_many/2` so `MLP` can pass lists between
  layers consistently.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.NN.Neuron
  alias MicrogradEx.NN.Random

  @enforce_keys [:neurons]
  defstruct [:neurons]

  @type t :: %__MODULE__{neurons: [Neuron.t()]}

  @doc """
  Creates a layer with `output_count` neurons.

  Options are forwarded to each neuron. `:seed` is consumed once and advanced
  across all weights, so deterministic layers do not repeat identical neurons.
  """
  def new(input_count, output_count, opts \\ []) do
    validate_output_count!(output_count)
    state = Random.state_from_seed(Keyword.get(opts, :seed))
    {layer, _state} = new_with_state(input_count, output_count, opts, state)
    layer
  end

  @doc false
  def new_with_state(input_count, output_count, opts, state) do
    validate_output_count!(output_count)
    opts = Keyword.delete(opts, :seed)

    {neurons, state} =
      Enum.map_reduce(1..output_count//1, state, fn _index, state ->
        Neuron.new_with_state(input_count, opts, state)
      end)

    {%__MODULE__{neurons: neurons}, state}
  end

  @doc """
  Runs the layer and unwraps singleton output layers.
  """
  def forward(%__MODULE__{} = layer, inputs) do
    layer
    |> forward_many(inputs)
    |> unwrap_single()
  end

  @doc """
  Runs the layer and always returns a list of output values.
  """
  def forward_many(%__MODULE__{} = layer, inputs) do
    Enum.map(layer.neurons, &Neuron.forward(&1, inputs))
  end

  @doc """
  Returns all trainable parameters in neuron order.
  """
  def parameters(%__MODULE__{} = layer) do
    Enum.flat_map(layer.neurons, &Neuron.parameters/1)
  end

  @doc false
  def apply_gradients(%__MODULE__{} = layer, %Gradients{} = gradients, learning_rate)
      when is_number(learning_rate) do
    %__MODULE__{
      layer
      | neurons: Enum.map(layer.neurons, &Neuron.apply_gradients(&1, gradients, learning_rate))
    }
  end

  defp unwrap_single([only]), do: only
  defp unwrap_single(values), do: values

  defp validate_output_count!(output_count)
       when is_integer(output_count) and output_count > 0,
       do: :ok

  defp validate_output_count!(output_count) do
    raise ArgumentError,
          "expected output_count to be a positive integer, got: #{inspect(output_count)}"
  end
end

defmodule MicrogradEx.NN.MLP do
  @moduledoc """
  A multi-layer perceptron composed of `Layer` structs.

  This is the Elixir counterpart of `micrograd.nn.MLP`. Hidden layers use ReLU
  neurons and the final layer is linear, matching the original implementation.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.NN.Layer
  alias MicrogradEx.NN.Random

  @enforce_keys [:layers]
  defstruct [:layers]

  @type t :: %__MODULE__{layers: [Layer.t()]}

  @doc """
  Creates an MLP.

  `input_count` is the number of scalar input features. `output_counts` is the
  list of layer widths, for example `[16, 16, 1]` for two hidden layers and one
  scalar output.
  """
  def new(input_count, output_counts, opts \\ []) do
    validate_input_count!(input_count)
    validate_output_counts!(output_counts)

    state = Random.state_from_seed(Keyword.get(opts, :seed))
    opts = Keyword.delete(opts, :seed)
    sizes = [input_count | output_counts]
    layer_specs = sizes |> Enum.chunk_every(2, 1, :discard) |> Enum.with_index()
    final_index = length(layer_specs) - 1

    {layers, _state} =
      Enum.map_reduce(layer_specs, state, fn {[layer_input_count, layer_output_count], index},
                                             state ->
        layer_opts = Keyword.put(opts, :nonlin, index != final_index)
        Layer.new_with_state(layer_input_count, layer_output_count, layer_opts, state)
      end)

    %__MODULE__{layers: layers}
  end

  @doc """
  Runs a forward pass through all layers.

  The final result follows original micrograd ergonomics: a one-output MLP
  returns a single `Value`, while a multi-output MLP returns a list.
  """
  def forward(%__MODULE__{} = mlp, inputs) do
    mlp.layers
    |> Enum.reduce(inputs, fn layer, layer_inputs ->
      Layer.forward_many(layer, layer_inputs)
    end)
    |> unwrap_single()
  end

  @doc """
  Returns all trainable parameters in layer order.
  """
  def parameters(%__MODULE__{} = mlp) do
    Enum.flat_map(mlp.layers, &Layer.parameters/1)
  end

  @doc false
  def apply_gradients(%__MODULE__{} = mlp, %Gradients{} = gradients, learning_rate)
      when is_number(learning_rate) do
    %__MODULE__{
      mlp
      | layers: Enum.map(mlp.layers, &Layer.apply_gradients(&1, gradients, learning_rate))
    }
  end

  defp unwrap_single([only]), do: only
  defp unwrap_single(values), do: values

  defp validate_input_count!(input_count)
       when is_integer(input_count) and input_count >= 0,
       do: :ok

  defp validate_input_count!(input_count) do
    raise ArgumentError,
          "expected input_count to be a non-negative integer, got: #{inspect(input_count)}"
  end

  defp validate_output_counts!(output_counts)
       when is_list(output_counts) and output_counts != [] do
    Enum.each(output_counts, fn output_count ->
      unless is_integer(output_count) and output_count > 0 do
        raise ArgumentError,
              "expected output_counts to contain only positive integers, got: #{inspect(output_counts)}"
      end
    end)
  end

  defp validate_output_counts!(output_counts) do
    raise ArgumentError,
          "expected output_counts to be a non-empty list, got: #{inspect(output_counts)}"
  end
end

defmodule MicrogradEx.NN do
  @moduledoc """
  Public facade for the tiny neural-network library.

  The modules below intentionally match the scope of original micrograd:
  `Neuron`, `Layer`, and `MLP`. There are no tensors, optimizers, batching
  primitives, or GPU kernels. The purpose is to make backpropagation visible.

  Because models are immutable, training code uses this rhythm:

      loss = ...
      gradients = MicrogradEx.Value.backward(loss)
      model = MicrogradEx.NN.apply_gradients(model, gradients, 0.05)

  That is the Elixir equivalent of `loss.backward(); p.data += -lr * p.grad`.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.NN.Layer
  alias MicrogradEx.NN.MLP
  alias MicrogradEx.NN.Neuron

  @type model :: Neuron.t() | Layer.t() | MLP.t()

  @doc """
  Runs a model forward.
  """
  def forward(%Neuron{} = neuron, inputs), do: Neuron.forward(neuron, inputs)
  def forward(%Layer{} = layer, inputs), do: Layer.forward(layer, inputs)
  def forward(%MLP{} = mlp, inputs), do: MLP.forward(mlp, inputs)

  @doc """
  Returns trainable parameters from any neural-network module.
  """
  def parameters(%Neuron{} = neuron), do: Neuron.parameters(neuron)
  def parameters(%Layer{} = layer), do: Layer.parameters(layer)
  def parameters(%MLP{} = mlp), do: MLP.parameters(mlp)

  @doc """
  Counts trainable parameters.
  """
  def parameter_count(model), do: model |> parameters() |> length()

  @doc """
  Applies one stochastic-gradient-descent style update and returns a new model.

  `learning_rate` is the positive step size usually called `lr` in training
  loops. The update rule is:

      new_parameter = parameter - learning_rate * gradient
  """
  def apply_gradients(model, %Gradients{} = gradients, learning_rate)
      when is_number(learning_rate) do
    case model do
      %Neuron{} -> Neuron.apply_gradients(model, gradients, learning_rate)
      %Layer{} -> Layer.apply_gradients(model, gradients, learning_rate)
      %MLP{} -> MLP.apply_gradients(model, gradients, learning_rate)
    end
  end
end
