defmodule MicrogradEx.Gradients do
  @moduledoc """
  The immutable result of a reverse-mode automatic differentiation pass.

  Python micrograd writes gradients into mutable `Value.grad` fields. In Elixir,
  the more natural shape is to return the gradients as data:

      gradients = MicrogradEx.Value.backward(loss)
      dloss_dw = MicrogradEx.Gradients.get(gradients, weight)

  The table is keyed by the stable ids of the `Value` nodes that participated in
  the forward expression.
  """

  alias MicrogradEx.Value
  alias MicrogradEx.Value.Edge

  @enforce_keys [:output_id, :values]
  defstruct [:output_id, values: %{}]

  @type t :: %__MODULE__{
          output_id: pos_integer(),
          values: %{pos_integer() => float()}
        }

  @doc """
  Builds the gradient table for an output value.

  The output starts with gradient `1.0` because `d(output)/d(output) = 1`.
  Then each node sends that gradient to its parents through the local derivative
  edges recorded during the forward pass.
  """
  def backward(%Value{} = output) do
    gradients =
      output
      |> backward_ids()
      |> Enum.reduce(%{output.id => 1.0}, fn node_id, gradients ->
        node = Map.fetch!(output.graph, node_id)
        upstream_gradient = Map.get(gradients, node_id, 0.0)

        Enum.reduce(node.parents, gradients, fn %Edge{} = edge, updated_gradients ->
          contribution = upstream_gradient * edge.local_gradient

          Map.update(
            updated_gradients,
            edge.parent_id,
            contribution,
            &(&1 + contribution)
          )
        end)
      end)

    %__MODULE__{output_id: output.id, values: gradients}
  end

  @doc """
  Fetches the gradient for a value or node id.

  Missing entries return `0.0`, which is the correct derivative for values that
  are independent of the chosen output.
  """
  def get(%__MODULE__{} = gradients, %Value{id: id}), do: get(gradients, id)

  def get(%__MODULE__{values: values}, id) when is_integer(id) do
    Map.get(values, id, 0.0)
  end

  @doc """
  Returns the raw gradient map.

  This is mostly useful for assertions, debugging, or building custom optimizers.
  """
  def to_map(%__MODULE__{values: values}), do: values

  @doc """
  Returns graph node ids in forward topological order.

  Leaves appear before the output. This order is educational and useful for
  debugging; `backward/1` internally uses the reverse order so gradients flow
  from outputs back to inputs.
  """
  def topological_ids(%Value{} = output) do
    output
    |> backward_ids()
    |> Enum.reverse()
  end

  # Returns ids in the order needed for backpropagation: output first, then its
  # dependencies after all of their consumers. The DFS post-order is prepended,
  # so no expensive list append is needed while walking the graph.
  defp backward_ids(%Value{id: output_id, graph: graph}) do
    {_visited, ids} = visit(output_id, graph, MapSet.new(), [])
    ids
  end

  defp visit(node_id, graph, visited, ids) do
    if MapSet.member?(visited, node_id) do
      {visited, ids}
    else
      node = Map.fetch!(graph, node_id)
      visited = MapSet.put(visited, node_id)

      {visited, ids} =
        Enum.reduce(node.parents, {visited, ids}, fn %Edge{parent_id: parent_id},
                                                     {visited, ids} ->
          visit(parent_id, graph, visited, ids)
        end)

      {visited, [node_id | ids]}
    end
  end
end
