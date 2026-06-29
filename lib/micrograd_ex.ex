defmodule MicrogradEx do
  @moduledoc """
  Elixir-native micrograd: a tiny scalar reverse-mode automatic differentiation
  engine plus the small neural-network library from the original project.

  The original Python `micrograd` stores `data` and `grad` on mutable objects.
  Elixir data is immutable, so this port keeps the forward graph in each
  `MicrogradEx.Value` and returns a separate `MicrogradEx.Gradients` table from
  `backward/1`. That one design difference is the main FP adaptation:

    * forward expressions create new values;
    * `backward/1` creates a new gradient table;
    * neural-network training creates a new updated model.

  ## Example

      iex> x = MicrogradEx.value(3.0)
      iex> y = MicrogradEx.pow(x, 2)
      iex> gradients = MicrogradEx.backward(y)
      iex> MicrogradEx.grad(x, gradients)
      6.0
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value

  @doc """
  Creates a scalar differentiable value.

  This is a convenience wrapper around `MicrogradEx.Value.new/2`; use the
  `Value` module directly when writing longer expressions.
  """
  def value(data, opts \\ []), do: Value.new(data, opts)

  @doc """
  Adds two values or numbers.

  Numbers are automatically promoted to leaf `Value` structs. This mirrors the
  Python implementation's `other = Value(other)` coercion, but it stays explicit
  and regular because Elixir has no operator overloading.
  """
  def add(left, right), do: Value.add(left, right)

  @doc """
  Subtracts the second value or number from the first.
  """
  def sub(left, right), do: Value.sub(left, right)

  @doc """
  Multiplies two values or numbers.
  """
  def mul(left, right), do: Value.mul(left, right)

  @doc """
  Divides the first value or number by the second.
  """
  def divide(left, right), do: Value.divide(left, right)

  @doc """
  Raises a value or number to a scalar power.
  """
  def pow(value, exponent), do: Value.pow(value, exponent)

  @doc """
  Applies the rectified linear unit activation.
  """
  def relu(value), do: Value.relu(value)

  @doc """
  Runs reverse-mode automatic differentiation from an output value.
  """
  def backward(%Value{} = output), do: Value.backward(output)

  @doc """
  Looks up the gradient for a value in a gradient table.
  """
  def grad(%Value{} = value, %Gradients{} = gradients), do: Value.grad(value, gradients)
end
