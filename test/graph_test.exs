defmodule MicrogradEx.GraphTest do
  use ExUnit.Case, async: true

  alias MicrogradEx.Graph
  alias MicrogradEx.Value

  describe "nodes/2" do
    test "returns a node for a leaf value" do
      x = Value.new(2.0, label: "x")

      assert [
               %{
                 label: "x",
                 op: "leaf",
                 data: 2.0,
                 arity: 0,
                 is_leaf: true
               }
             ] = Graph.nodes(x)
    end

    test "returns all nodes in a scalar expression" do
      {_x, _y, z} = expression()

      nodes = Graph.nodes(z)

      assert length(nodes) == 5
      assert List.last(nodes).op == "relu"
      assert Enum.map(nodes, & &1.id) |> Enum.uniq() == Enum.map(nodes, & &1.id)
    end

    test "includes labels operations and data" do
      {x, y, z} = expression()

      nodes = Graph.nodes(z)

      assert Enum.any?(nodes, &(&1.label == "x" and &1.data == x.data))
      assert Enum.any?(nodes, &(&1.label == "y" and &1.data == y.data))
      assert Enum.any?(nodes, &(&1.op == "*"))
      assert Enum.any?(nodes, &(&1.op == "+"))
      assert Enum.any?(nodes, &(&1.op == "relu"))
    end

    test "includes gradients when supplied" do
      x = Value.new(2.0, label: "x")
      y = Value.mul(x, x)

      gradients = Value.backward(y)
      nodes = Graph.nodes(y, gradients)

      x_node = Enum.find(nodes, &(&1.label == "x"))

      assert Map.has_key?(x_node, :grad)
      assert_in_delta x_node.grad, 4.0, 1.0e-9
    end

    test "does not include gradients when omitted" do
      x = Value.new(2.0, label: "x")
      y = Value.mul(x, x)

      nodes = Graph.nodes(y)

      refute nodes |> Enum.find(&(&1.label == "x")) |> Map.has_key?(:grad)
    end
  end

  describe "edges/1" do
    test "returns no edges for a leaf value" do
      assert Graph.edges(Value.new(2.0, label: "x")) == []
    end

    test "returns parent to child edges for a binary expression" do
      {x, y, z} = expression()

      edges = Graph.edges(z)

      assert length(edges) == 5
      assert Enum.any?(edges, &(&1.from == graph_id(x.id) and &1.child_op == "*"))
      assert Enum.any?(edges, &(&1.from == graph_id(y.id) and &1.child_op == "*"))
      assert Enum.any?(edges, &(&1.from == graph_id(x.id) and &1.child_op == "+"))
    end

    test "edge endpoints exist in nodes" do
      {_x, _y, z} = expression()

      node_ids =
        z
        |> Graph.nodes()
        |> Enum.map(& &1.id)
        |> MapSet.new()

      for edge <- Graph.edges(z) do
        assert MapSet.member?(node_ids, edge.from)
        assert MapSet.member?(node_ids, edge.to)
      end
    end
  end

  describe "to_dot/2" do
    test "exports a digraph" do
      x = Value.new(2.0, label: "x")
      y = Value.mul(x, x)

      dot = Graph.to_dot(y)

      assert dot =~ "digraph MicrogradEx"
      assert dot =~ "rankdir=LR"
      assert dot =~ "x"
      assert dot =~ "->"
    end

    test "includes nodes and edges" do
      {_x, _y, z} = expression()

      dot = Graph.to_dot(z)

      assert dot =~ "op=*"
      assert dot =~ "op=+"
      assert dot =~ "op=relu"
      assert dot =~ " -> "
    end

    test "includes gradients when supplied" do
      x = Value.new(2.0, label: "x")
      y = Value.mul(x, x)
      gradients = Value.backward(y)

      dot = Graph.to_dot(y, gradients)

      assert dot =~ "grad=4.0000"
    end

    test "escapes labels" do
      x = Value.new(2.0, label: ~s(x "quoted"))

      dot = Graph.to_dot(x)

      assert dot =~ ~s(x \\"quoted\\")
    end
  end

  defp expression do
    x = Value.new(2.0, label: "x")
    y = Value.new(-3.0, label: "y")

    z =
      x
      |> Value.mul(y)
      |> Value.add(x)
      |> Value.relu()

    {x, y, z}
  end

  defp graph_id(id), do: "v#{id}"
end
