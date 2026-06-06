# Agents Setup

Use this file as the setup handoff for any agent that should answer JAX questions from this repository.

## Purpose

This repo is a local, queryable knowledge base built from `https://docs.jax.dev/en/latest/`.

Agents should use it before answering factual JAX questions. The KB includes:

- fetched JAX documentation pages in `docs/pages/`
- raw HTML in `docs/html/`
- route and API inventory data in `data/`
- a local TF-IDF index in `index/jax-docs-tfidf.joblib`
- a query CLI at `scripts/query.py`
- an optional skill at `skills/jax-docs/`

Do not add an MCP server. This KB is intentionally local files plus scripts.

## Clone Or Update

If the repo is not present:

```bash
git clone https://github.com/coder-2011/jax-kb.git
cd jax-kb
```

If the repo is already present:

```bash
cd jax-kb
git pull
```

Run all commands from the repo root.

## Install The Agent Skill

The bundled skill lives at:

```text
skills/jax-docs/
```

If the runtime uses a skills directory, symlink it:

```bash
SKILL_HOME="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILL_HOME"
ln -sfn "$PWD/skills/jax-docs" "$SKILL_HOME/jax-docs"
```

If symlinks are not supported, copy `skills/jax-docs` into the runtime's skills directory. If repo-local skills are supported, point the runtime directly at `skills/jax-docs/`.

## Check Dependencies

Do not create a virtual environment unless explicitly asked. First check whether the current Python already has the dependencies:

```bash
python3 - <<'PY'
import importlib.util
mods = ["joblib", "numpy", "sklearn", "scipy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
PY
```

If anything is missing, install the small requirements file with the least disruptive tool available:

```bash
python3 -m pip install -r requirements.txt
```

If system Python refuses package installation, use an existing project environment or a one-shot tool runner such as `uv run --with joblib --with numpy --with scikit-learn --with scipy ...`.

## Verify The KB

Run:

```bash
python3 scripts/query.py "How should I think about jit and tracing?" --top-k 4
python3 scripts/query.py "What does jax.vmap do?" --top-k 4
python3 scripts/query.py "Explain pytrees at a high level" --json --top-k 4
python3 -m unittest discover -s tests
```

Expected behavior:

- JIT/tracing query returns JIT, tracing, compilation, or jaxpr context.
- `jax.vmap` query returns `_autosummary/jax.vmap` with explanatory chunk text.
- Pytrees query returns readable `pytrees` or key-concepts content.
- Tests pass.

## Query Rules

For factual JAX questions, query before answering from memory:

```bash
python3 scripts/query.py "QUESTION HERE" --top-k 8
```

For structured output:

```bash
python3 scripts/query.py "QUESTION HERE" --json --top-k 8
```

JSON output is compact by default. It includes routes, URLs, scores, headings, and focused excerpts. Add `--full-text` only when the full indexed text is needed:

```bash
python3 scripts/query.py "QUESTION HERE" --json --full-text --top-k 4
```

Use `--max-chars` to tune excerpt length:

```bash
python3 scripts/query.py "QUESTION HERE" --top-k 8 --max-chars 1800
```

## Query Strategy

For broad questions, keep the wording broad and request several results:

```bash
python3 scripts/query.py "How should I think about jit and tracing?" --top-k 10
python3 scripts/query.py "Explain JAX transformations at a high level." --top-k 10
python3 scripts/query.py "What parts of JAX matter most for writing fast numerical code?" --top-k 10
```

For exact APIs, classes, methods, or properties, include the exact name:

```bash
python3 scripts/query.py "What does jax.grad do?" --top-k 8
python3 scripts/query.py "What does jax.vmap do?" --top-k 8
python3 scripts/query.py "How does jax.Array.addressable_shards work?" --top-k 8
```

For debugging or behavior questions, name the symptom and transformation:

```bash
python3 scripts/query.py "How do I debug NaNs in jit?" --top-k 8 --max-chars 1800
```

If the first result set is weak, rerun once with related exact terms:

```bash
python3 scripts/query.py "JAX transformations tracing jit grad vmap jaxpr" --top-k 12
```

## Answering Rules

- Read multiple returned excerpts, not only the first result.
- Prefer chunk results for detailed claims.
- Use section/object results as navigation hints when useful.
- Cite returned routes or URLs for factual claims.
- Do not cite a route that did not appear in the query output.
- Separate docs-grounded facts from outside reasoning.
- If the KB does not contain enough evidence, say what is missing.

## Rebuild Only When Needed

Rebuild only if the JAX docs changed or the crawler, chunker, or index logic changed:

```bash
python3 scripts/crawl_jax_docs.py
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```

`scripts/crawl_jax_docs.py` requires `pandoc` because it converts fetched HTML to Markdown.
