defmodule MicrogradEx.Value.Edge do
  @moduledoc """
  A single local derivative from one operation output back to one parent value.

  In the Python version each `Value` stores a `_backward` closure that mutates
  its parents. This struct is the functional replacement for those closures.
  An edge says:

      d(output) / d(parent) = local_gradient

  During backpropagation we multiply that local gradient by the gradient that
  has already reached the output node. That is the chain rule in one line.
  """

  @enforce_keys [:parent_id, :local_gradient]
  defstruct [:parent_id, :local_gradient]

  @type t :: %__MODULE__{
          parent_id: pos_integer(),
          local_gradient: float()
        }

  @doc false
  def new(parent_id, local_gradient) do
    %__MODULE__{
      parent_id: parent_id,
      local_gradient: local_gradient * 1.0
    }
  end
end

defmodule MicrogradEx.Value.Node do
  @moduledoc """
  The immutable record stored in a value's computation graph.

  A `MicrogradEx.Value` is the small public handle that users pass around. The
  graph itself is a map of node id to this struct. Keeping nodes separate from
  values avoids recursive structs while still giving every output enough
  information to walk backward through the full expression that produced it.
  """

  alias MicrogradEx.Value.Edge

  @enforce_keys [:id, :data, :op, :parents]
  defstruct [:id, :data, :op, :label, parents: []]

  @type op :: :leaf | :+ | :- | :* | :neg | :relu | {:pow, number()}

  @type t :: %__MODULE__{
          id: pos_integer(),
          data: float(),
          op: op(),
          label: String.t() | nil,
          parents: [Edge.t()]
        }

  @doc false
  def leaf(id, data, label) do
    %__MODULE__{
      id: id,
      data: data * 1.0,
      op: :leaf,
      label: label,
      parents: []
    }
  end

  @doc false
  def operation(id, data, op, parents, label) do
    %__MODULE__{
      id: id,
      data: data * 1.0,
      op: op,
      label: label,
      parents: parents
    }
  end
end

defmodule MicrogradEx.Value do
  @moduledoc """
  A scalar value that remembers the expression graph that produced it.

  `Value` is intentionally scalar, just like Andrej Karpathy's original
  micrograd. Vectors, neurons, layers, and MLPs are built by composing many
  scalar values. That keeps the math visible: every addition, multiplication,
  power, and ReLU contributes a small local derivative edge to the graph.

  The important Elixir-specific difference is that `grad` is not mutated during
  `backward/1`. The field exists only as a convenient annotation for inspected
  values. The source of truth is the `MicrogradEx.Gradients` table returned by
  `backward/1`.
  """

  import Kernel, except: [div: 2]

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value.Edge
  alias MicrogradEx.Value.Node

  @enforce_keys [:id, :data, :graph]
  defstruct [:id, :data, :graph, :label, grad: 0.0]

  @type t :: %__MODULE__{
          id: pos_integer(),
          data: float(),
          grad: float(),
          label: String.t() | nil,
          graph: %{pos_integer() => Node.t()}
        }

  @doc """
  Creates a new leaf value.

  A leaf is an input to a computation: a training example, a model parameter, a
  constant promoted into the graph, or any other scalar whose gradient may be
  interesting later.

  The optional `:label` is never used for math. It exists for debugging,
  examples, and graph inspection.
  """
  def new(data, opts \\ []) when is_number(data) do
    id = System.unique_integer([:positive, :monotonic])
    label = Keyword.get(opts, :label)
    data = data * 1.0

    %__MODULE__{
      id: id,
      data: data,
      graph: %{id => Node.leaf(id, data, label)},
      label: label,
      grad: 0.0
    }
  end

  @doc """
  Promotes plain numbers to values and leaves existing values unchanged.

  This helper is what lets the public arithmetic functions accept both
  `%Value{}` structs and numbers:

      iex> x = MicrogradEx.Value.new(2.0)
      iex> MicrogradEx.Value.mul(x, 3).data
      6.0
  """
  def coerce(%__MODULE__{} = value), do: value

  def coerce(number) when is_number(number), do: new(number)

  def coerce(other) do
    raise ArgumentError,
          "expected a number or %MicrogradEx.Value{}, got: #{inspect(other)}"
  end

  @doc """
  Adds two values or numbers.

  The derivative of `a + b` with respect to each parent is `1`, so the output
  node stores two parent edges with local gradient `1.0`.
  """
  def add(left, right, opts \\ []) do
    left = coerce(left)
    right = coerce(right)

    operation(
      left.data + right.data,
      :+,
      [{left, 1.0}, {right, 1.0}],
      opts
    )
  end

  @doc """
  Negates a value or number.

  This is implemented as its own operation instead of `mul(value, -1)` so the
  graph remains compact and the local derivative is explicit: `d(-x)/dx = -1`.
  """
  def neg(value, opts \\ []) do
    value = coerce(value)

    operation(
      -value.data,
      :neg,
      [{value, -1.0}],
      opts
    )
  end

  @doc """
  Subtracts the second value or number from the first.

  The output stores local derivatives `1` for the left parent and `-1` for the
  right parent, which is exactly the derivative of `a - b`.
  """
  def sub(left, right, opts \\ []) do
    left = coerce(left)
    right = coerce(right)

    operation(
      left.data - right.data,
      :-,
      [{left, 1.0}, {right, -1.0}],
      opts
    )
  end

  @doc """
  Multiplies two values or numbers.

  For `a * b`, the local derivative with respect to `a` is `b`, and the local
  derivative with respect to `b` is `a`. These parent data values are captured
  at graph-construction time, just like the closure in the Python version.
  """
  def mul(left, right, opts \\ []) do
    left = coerce(left)
    right = coerce(right)

    operation(
      left.data * right.data,
      :*,
      [{left, right.data}, {right, left.data}],
      opts
    )
  end

  @doc """
  Raises a value or number to a scalar exponent.

  The exponent must be a plain number, not another `Value`. That matches the
  original micrograd implementation and keeps the local derivative simple:

      d(x ** n) / dx = n * x ** (n - 1)
  """
  def pow(value, exponent, opts \\ []) when is_number(exponent) do
    value = coerce(value)
    data = :math.pow(value.data, exponent)
    local_gradient = exponent * :math.pow(value.data, exponent - 1)

    operation(
      data,
      {:pow, exponent},
      [{value, local_gradient}],
      opts
    )
  end

  @doc """
  Divides the first value or number by the second.

  Division is represented as multiplication by `right ** -1`, the same identity
  used in the original Python source. Keeping it as composition means the graph
  naturally contains the reciprocal operation and then the multiplication.
  """
  def divide(left, right) do
    right = coerce(right)
    mul(left, pow(right, -1.0))
  end

  @doc """
  Alias for `divide/2`.

  `Kernel.div/2` is integer division, so longer examples generally read better
  with `divide/2`. This alias exists for users who expect the shorter name from
  the original arithmetic operation.
  """
  def div(left, right), do: divide(left, right)

  @doc """
  Applies the rectified linear unit activation.

  ReLU keeps positive inputs and clamps negative inputs to zero. At exactly
  zero this port follows the original micrograd code: the gradient is `0.0`
  because the output is not greater than zero.
  """
  def relu(value, opts \\ []) do
    value = coerce(value)
    data = if value.data < 0.0, do: 0.0, else: value.data
    local_gradient = if data > 0.0, do: 1.0, else: 0.0

    operation(
      data,
      :relu,
      [{value, local_gradient}],
      opts
    )
  end

  @doc """
  Sums a list of values or numbers.

  This is useful when writing losses because Elixir does not have Python's
  overloaded `sum` for custom objects. The optional initial value defaults to a
  differentiable zero leaf.
  """
  def sum(values, initial \\ new(0.0)) when is_list(values) do
    Enum.reduce(values, coerce(initial), fn value, acc -> add(acc, value) end)
  end

  @doc """
  Runs reverse-mode autodiff and returns an immutable gradient table.
  """
  def backward(%__MODULE__{} = output), do: Gradients.backward(output)

  @doc """
  Fetches this value's gradient from a gradient table.

  Values that do not influence the output have gradient `0.0`.
  """
  def grad(%__MODULE__{} = value, %Gradients{} = gradients), do: Gradients.get(gradients, value)

  @doc """
  Returns a copy of `value` with its `grad` field filled from a gradient table.

  This is only for display or debugging. The returned struct is not special and
  does not mutate the original computation graph.
  """
  def with_grad(%__MODULE__{} = value, %Gradients{} = gradients) do
    %{value | grad: grad(value, gradients)}
  end

  # All arithmetic constructors eventually call this helper. It creates a fresh
  # node id, records the local derivative edges, merges the parent graphs, and
  # returns the public `Value` handle for the new output node.
  defp operation(data, op, parent_specs, opts) do
    id = System.unique_integer([:positive, :monotonic])
    label = Keyword.get(opts, :label)
    data = data * 1.0

    parents =
      Enum.map(parent_specs, fn {%__MODULE__{} = parent, local_gradient} ->
        Edge.new(parent.id, local_gradient)
      end)

    graph =
      parent_specs
      |> Enum.map(fn {%__MODULE__{} = parent, _local_gradient} -> parent end)
      |> merge_graphs()
      |> Map.put(id, Node.operation(id, data, op, parents, label))

    %__MODULE__{
      id: id,
      data: data,
      graph: graph,
      label: label,
      grad: 0.0
    }
  end

  # Parent values often share large parts of a graph. `Map.merge/2` is enough
  # because ids are unique and a shared id always refers to the same node.
  defp merge_graphs(values) do
    Enum.reduce(values, %{}, fn %__MODULE__{graph: graph}, merged ->
      Map.merge(merged, graph)
    end)
  end
end
