# Agent Instructions

Use this repo when you need answers grounded in the JAX documentation.

## Install Skill

The bundled agent skill lives at `skills/jax-docs/`.

If the runtime uses a skills directory, install it with a symlink:

```bash
SKILL_HOME="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILL_HOME"
ln -sfn "$PWD/skills/jax-docs" "$SKILL_HOME/jax-docs"
```

If symlinks are not supported, copy `skills/jax-docs` into the runtime's skills directory. If repo-local skills are supported, point the runtime directly at that folder.

## Query First

For factual JAX questions, query the local index before relying on model memory:

```bash
python3 scripts/query.py "QUESTION HERE" --top-k 8
```

Use JSON when you need machine-readable retrieval output:

```bash
python3 scripts/query.py "QUESTION HERE" --json --top-k 8
```

JSON output is compact by default so agents can read several results without pulling duplicate full chunks into context. Add `--full-text` only when you need the full indexed text, and use `--max-chars` to tune excerpt length:

```bash
python3 scripts/query.py "QUESTION HERE" --json --full-text --top-k 4
python3 scripts/query.py "QUESTION HERE" --top-k 8 --max-chars 1800
```

## Broad Questions

For broad questions, ask broad questions directly. The query tool is tuned to route these into section-level docs records and relevant chunks:

```bash
python3 scripts/query.py "How should I think about jit and tracing?"
python3 scripts/query.py "Explain JAX transformations at a high level."
python3 scripts/query.py "What parts of JAX matter most for writing fast numerical code?"
```

Synthesize answers from several returned sections/chunks. Include route or URL references when giving factual claims.

## Exact Questions

For exact APIs, classes, methods, or properties, include the exact name:

```bash
python3 scripts/query.py "What does jax.grad do?"
python3 scripts/query.py "What does jax.vmap do?"
python3 scripts/query.py "How does jax.Array.addressable_shards work?"
```

Prefer the highest-ranked chunk that contains the exact API name.

## Rebuild

If docs or chunking change:

```bash
python3 scripts/crawl_jax_docs.py
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```

Do not add an MCP server. This repo is intentionally a pure file-and-script knowledge base.
