# Development

## Environment

```bash
uv sync --all-groups
```

| Group | Contents |
|---|---|
| (main) | Runtime dependencies |
| `dev` | `ruff`, `pytest`; installed by `uv sync` by default |
| `docs` | `sphinx`, `furo`, `myst-parser`, `sphinx-autoapi`, `sphinx-copybutton`, `sphinxcontrib-mermaid` |

## Checks

```bash
uv run ruff check .
uv run pytest
```

The tests in `tests/` safeguard the computations on which the results rest:
matching of boundaries, edge rules, 95 % intervals, SESOI, classes,
saturation point, Shapley values, saturation curve and the path from the
results table to the figures. One test recomputes the published
Shapley values and gains from the coalition values of a run.

Model calls, Whisper and video decoding are not tested: they
need a network, credentials or recordings and would not be reproducible
in CI.

## Building the documentation

```bash
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

The API reference is generated from the docstrings; the package is not
imported. `-W` turns every warning into an error – a dead cross-reference aborts
the build. For a live preview while writing, use e.g. `sphinx-autobuild`
(not part of the dependencies):

```bash
uv run --with sphinx-autobuild sphinx-autobuild docs docs/_build/html
```

The pages are Markdown (MyST); diagrams are Mermaid blocks.

## Continuous integration

`.github/workflows/pull-request.yml` runs on every pull request and push to
`main`:

| Job | Steps |
|---|---|
| Lint and package | `uv lock --check`, `ruff check .`, `uv build` |
| Test | `uv sync --locked`, `pytest tests` |
| Docs | `uv sync --locked --only-group docs`, `sphinx-build -W` |

## Conventions

- Docstrings, identifiers, texts of the user interface and the report in English.
- Docstrings in Sphinx field style (`:param x:`, `:return:`, `:raises E:`).
- Line length 100 (`ruff`).
- Computations do not import Qt; the user interface computes nothing itself.
- Every constant in {py:mod}`docupilot.config` states its origin.
- A change to a model prompt increments its `PROMPT_VERSION`.
- No code distinguishes between real, test or synthetic data.
