defmodule MicrogradEx.ValueTest do
  use ExUnit.Case, async: true

  doctest MicrogradEx
  doctest MicrogradEx.Value

  alias MicrogradEx.Gradients
  alias MicrogradEx.Value

  describe "scalar autograd" do
    test "matches the original micrograd sanity-check expression" do
      x = Value.new(-4.0)
      z = Value.add(Value.add(Value.mul(2.0, x), 2.0), x)
      q = Value.add(Value.relu(z), Value.mul(z, x))
      h = Value.relu(Value.mul(z, z))
      y = Value.add(Value.add(h, q), Value.mul(q, x))

      gradients = Value.backward(y)

      assert y.data == -20.0
      assert Value.grad(y, gradients) == 1.0
      assert Value.grad(x, gradients) == 46.0
    end

    test "matches the original micrograd more-ops expression" do
      a = Value.new(-4.0)
      b = Value.new(2.0)
      c = Value.add(a, b)
      d = Value.add(Value.mul(a, b), Value.pow(b, 3))
      c = Value.add(Value.add(c, c), 1.0)
      c = Value.add(Value.add(Value.add(c, 1.0), c), Value.neg(a))
      d = Value.add(Value.add(d, Value.mul(d, 2.0)), Value.relu(Value.add(b, a)))
      d = Value.add(Value.add(d, Value.mul(3.0, d)), Value.relu(Value.sub(b, a)))
      e = Value.sub(c, d)
      f = Value.pow(e, 2)
      g = Value.add(Value.divide(f, 2.0), Value.divide(10.0, f))

      gradients = Value.backward(g)

      assert_in_delta g.data, 24.70408163265306, 1.0e-12
      assert_in_delta Value.grad(a, gradients), 138.83381924198252, 1.0e-12
      assert_in_delta Value.grad(b, gradients), 645.5772594752186, 1.0e-12
    end

    test "accumulates gradients through repeated parent edges" do
      x = Value.new(3.0)
      y = Value.add(Value.mul(x, x), Value.add(x, x))

      gradients = Value.backward(y)

      assert y.data == 15.0
      assert Value.grad(x, gradients) == 8.0
    end

    test "uses the original ReLU derivative convention at negative, zero, and positive inputs" do
      negative = Value.new(-2.0)
      zero = Value.new(0.0)
      positive = Value.new(2.0)
      output = Value.sum([Value.relu(negative), Value.relu(zero), Value.relu(positive)])

      gradients = Value.backward(output)

      assert Value.grad(negative, gradients) == 0.0
      assert Value.grad(zero, gradients) == 0.0
      assert Value.grad(positive, gradients) == 1.0
    end

    test "reports zero gradient for values outside the selected output graph" do
      x = Value.new(2.0)
      unrelated = Value.new(10.0)
      y = Value.pow(x, 3)

      gradients = Value.backward(y)

      assert Value.grad(x, gradients) == 12.0
      assert Value.grad(unrelated, gradients) == 0.0
    end

    test "exposes a debuggable topological order" do
      x = Value.new(2.0)
      y = x |> Value.mul(3.0) |> Value.add(1.0)

      ids = Gradients.topological_ids(y)

      assert List.first(ids) == x.id
      assert List.last(ids) == y.id
      assert Enum.uniq(ids) == ids
    end

    test "can annotate a value with a gradient without mutating the original value" do
      x = Value.new(5.0)
      y = Value.pow(x, 2)
      gradients = Value.backward(y)

      annotated = Value.with_grad(x, gradients)

      assert x.grad == 0.0
      assert annotated.grad == 10.0
      assert annotated.data == x.data
    end
  end
end
