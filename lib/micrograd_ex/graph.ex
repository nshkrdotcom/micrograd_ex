defmodule MicrogradEx.Graph do
  @moduledoc """
  Extracts scalar computation graphs from `MicrogradEx.Value` expressions.

  This module returns plain Elixir data structures suitable for Livebook tables,
  documentation, and Graphviz DOT export. It does not depend on Livebook,
  Graphviz, or browser rendering.
  """

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value
  alias MicrogradEx.Value.Edge
  alias MicrogradEx.Value.Node

  @type node_row :: %{
          required(:id) => String.t(),
          required(:label) => String.t(),
          required(:op) => String.t(),
          required(:data) => float(),
          required(:arity) => non_neg_integer(),
          required(:is_leaf) => boolean(),
          optional(:grad) => float()
        }

  @type edge_row :: %{
          required(:from) => String.t(),
          required(:to) => String.t(),
          required(:local_gradient) => float(),
          required(:parent_label) => String.t(),
          required(:child_label) => String.t(),
          required(:child_op) => String.t()
        }

  @doc """
  Returns graph nodes reachable from a final `Value`.

  Nodes are returned in topological order from inputs toward the selected
  output. When a `Gradients` table is supplied, each row also includes `:grad`.
  """
  @spec nodes(Value.t(), Gradients.t() | nil) :: [node_row()]
  def nodes(value, gradients \\ nil)

  def nodes(%Value{} = value, gradients) do
    value
    |> Gradients.topological_ids()
    |> Enum.map(fn node_id ->
      value.graph
      |> Map.fetch!(node_id)
      |> node_row(gradients)
    end)
  end

  @doc """
  Returns parent-to-child graph edges reachable from a final `Value`.
  """
  @spec edges(Value.t()) :: [edge_row()]
  def edges(%Value{} = value) do
    value
    |> Gradients.topological_ids()
    |> Enum.flat_map(fn child_id ->
      child = Map.fetch!(value.graph, child_id)

      Enum.map(child.parents, fn %Edge{} = edge ->
        parent = Map.fetch!(value.graph, edge.parent_id)

        %{
          from: graph_id(edge.parent_id),
          to: graph_id(child.id),
          local_gradient: edge.local_gradient,
          parent_label: node_label(parent),
          child_label: node_label(child),
          child_op: op_label(child)
        }
      end)
    end)
  end

  @doc """
  Returns a Graphviz DOT representation of a scalar computation graph.

  Graphviz is not required to call this function; it only returns text.
  """
  @spec to_dot(Value.t(), Gradients.t() | nil) :: String.t()
  def to_dot(value, gradients \\ nil)

  def to_dot(%Value{} = value, gradients) do
    node_lines =
      value
      |> nodes(gradients)
      |> Enum.map(&dot_node/1)

    edge_lines =
      value
      |> edges()
      |> Enum.map(&dot_edge/1)

    [
      "digraph MicrogradEx {",
      "  rankdir=LR;",
      Enum.join(node_lines, "\n"),
      Enum.join(edge_lines, "\n"),
      "}"
    ]
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("\n")
  end

  @doc """
  Returns both nodes and edges for a scalar computation graph.
  """
  @spec graph(Value.t(), Gradients.t() | nil) :: %{nodes: [node_row()], edges: [edge_row()]}
  def graph(value, gradients \\ nil) do
    %{nodes: nodes(value, gradients), edges: edges(value)}
  end

  defp node_row(%Node{} = node, nil) do
    base_node_row(node)
  end

  defp node_row(%Node{} = node, %Gradients{} = gradients) do
    node
    |> base_node_row()
    |> Map.put(:grad, Gradients.get(gradients, node.id))
  end

  defp base_node_row(%Node{} = node) do
    %{
      id: graph_id(node.id),
      label: node_label(node),
      op: op_label(node),
      data: node.data,
      arity: length(node.parents),
      is_leaf: node.parents == []
    }
  end

  defp dot_node(row) do
    label =
      row
      |> label_parts()
      |> Enum.join(" | ")
      |> escape_dot()

    ~s(  #{row.id} [label="#{label}"];)
  end

  defp label_parts(row) do
    [
      display_name(row),
      "data=#{format_float(row.data)}",
      if(Map.has_key?(row, :grad), do: "grad=#{format_float(row.grad)}"),
      if(row.op != "leaf", do: "op=#{row.op}")
    ]
    |> Enum.reject(&is_nil/1)
  end

  defp display_name(%{label: ""} = row), do: row.id
  defp display_name(%{label: label}), do: label

  defp dot_edge(%{from: from, to: to}) do
    "  #{from} -> #{to};"
  end

  defp node_label(%Node{label: nil}), do: ""
  defp node_label(%Node{label: label}), do: to_string(label)

  defp op_label(%Node{op: :leaf}), do: "leaf"
  defp op_label(%Node{op: :+}), do: "+"
  defp op_label(%Node{op: :-}), do: "-"
  defp op_label(%Node{op: :*}), do: "*"
  defp op_label(%Node{op: :neg}), do: "neg"
  defp op_label(%Node{op: :relu}), do: "relu"
  defp op_label(%Node{op: {:pow, exponent}}), do: "**#{format_float(exponent)}"
  defp op_label(%Node{op: op}), do: to_string(op)

  defp graph_id(id) do
    "v#{id}"
    |> String.replace(~r/[^a-zA-Z0-9_]/, "_")
  end

  defp escape_dot(value) do
    value
    |> to_string()
    |> String.replace("\\", "\\\\")
    |> String.replace("\"", "\\\"")
    |> String.replace("\n", "\\n")
  end

  defp format_float(value) when is_float(value) do
    :erlang.float_to_binary(value, decimals: 4)
  end

  defp format_float(value) when is_integer(value), do: Integer.to_string(value)
end
