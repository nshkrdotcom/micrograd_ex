defmodule MicrogradEx.MixProject do
  use Mix.Project

  def project do
    [
      app: :micrograd_ex,
      version: "0.1.0",
      elixir: "~> 1.20",
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

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "guides/getting_started_with_livebook.md",
        "guides/micrograd_demo_walkthrough.md",
        "guides/elixir_design_notes.md",
        "guides/api_reference.md",
        "guides/troubleshooting.md"
      ],
      groups_for_extras: [
        Guides: Path.wildcard("guides/*.md")
      ]
    ]
  end
end
