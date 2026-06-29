defmodule MicrogradEx.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/nshkrdotcom/micrograd_ex"
  @homepage_url "https://hexdocs.pm/micrograd_ex"

  def project do
    [
      app: :micrograd_ex,
      version: @version,
      elixir: "~> 1.20",
      name: "MicrogradEx",
      description: description(),
      package: package(),
      source_url: @source_url,
      homepage_url: @homepage_url,
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      aliases: aliases()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:ex_doc, "~> 0.40.3", only: :dev, runtime: false}
    ]
  end

  defp aliases do
    [
      quality: ["format --check-formatted", "credo --strict", "dialyzer"]
    ]
  end

  defp description do
    "An educational scalar reverse-mode autodiff and neural-network library for Elixir."
  end

  defp package do
    [
      files: ~w(assets guides lib LICENSE mix.exs README.md notebooks),
      licenses: ["MIT"],
      maintainers: ["nshkrdotcom"],
      links: %{
        "Docs" => @homepage_url,
        "GitHub" => @source_url
      }
    ]
  end

  defp docs do
    [
      main: "readme",
      logo: "assets/micrograd_ex.svg",
      assets: %{"assets" => "assets"},
      source_ref: "v#{@version}",
      source_url: @source_url,
      homepage_url: @homepage_url,
      extra_section: "Guides",
      extras: extras(),
      groups_for_extras: groups_for_extras(),
      groups_for_modules: groups_for_modules()
    ]
  end

  defp extras do
    [
      "README.md",
      "guides/getting_started_with_livebook.md",
      "guides/micrograd_demo_walkthrough.md",
      "guides/elixir_design_notes.md",
      "guides/api_reference.md",
      "guides/troubleshooting.md"
    ]
  end

  defp groups_for_extras do
    [
      "Start Here": [
        "README.md",
        "guides/getting_started_with_livebook.md"
      ],
      Walkthroughs: [
        "guides/micrograd_demo_walkthrough.md"
      ],
      Reference: [
        "guides/elixir_design_notes.md",
        "guides/api_reference.md",
        "guides/troubleshooting.md"
      ]
    ]
  end

  defp groups_for_modules do
    [
      "Autodiff Core": [
        MicrogradEx,
        MicrogradEx.Value,
        MicrogradEx.Gradients,
        MicrogradEx.Graph
      ],
      "Neural Networks": [
        MicrogradEx.NN,
        MicrogradEx.NN.Neuron,
        MicrogradEx.NN.Layer,
        MicrogradEx.NN.MLP
      ],
      "Data and Training": [
        MicrogradEx.Datasets,
        MicrogradEx.Datasets.Dataset,
        MicrogradEx.Losses,
        MicrogradEx.Losses.Result,
        MicrogradEx.Trainer,
        MicrogradEx.Trainer.Run,
        MicrogradEx.PlotData
      ],
      "Value Graph Internals": [
        MicrogradEx.Value.Edge,
        MicrogradEx.Value.Node
      ]
    ]
  end
end
