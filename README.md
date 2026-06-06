# JAX Documentation Knowledge Base

A local knowledge base for you and your agents, all about JAX. It mirrors the shape of the CUDA guide KB, but the source is the full Sphinx documentation site at `https://docs.jax.dev/en/latest/` rather than one PDF.

The crawler discovers routes from the JAX Sphinx search index, Python object inventory, and same-site HTML links. It converts fetched HTML pages to Markdown with `pandoc`, then builds a local TF-IDF retrieval index over page chunks, headings, and API inventory records.

## Setup Guide

This repo is meant to ship with the docs, chunks, and index already built. Setup is: get the repo, make sure the small Python dependency set is available, then run a query.

### 1. Get the repo

```bash
git clone https://github.com/coder-2011/jax-kb.git
cd jax-kb
```

If the repo is already present:

```bash
cd jax-kb
git pull
```

### 2. Use your existing Python environment

Do not create a virtual environment unless you personally want one. The query tool only needs:

- `joblib`
- `numpy`
- `scikit-learn`
- `scipy`

Check whether they are already available:

```bash
python3 - <<'PY'
import importlib.util
mods = ["joblib", "numpy", "sklearn", "scipy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
PY
```

If nothing is missing, skip installation.

### 3. Install only if needed

```bash
python3 -m pip install -r requirements.txt
```

### 4. Verify it works

```bash
python3 scripts/query.py "How should I think about jit and tracing?" --top-k 4
python3 scripts/query.py "What does jax.vmap do?" --top-k 4
python3 -m unittest discover -s tests
```

The first query should return JIT, tracing, or compilation context. The second should return `jax.vmap` API context.

### 5. Query the KB

Broad question:

```bash
python3 scripts/query.py "What parts of JAX matter most for writing fast numerical code?" --top-k 10
```

Exact API or class:

```bash
python3 scripts/query.py "What does jax.grad do?" --top-k 8
python3 scripts/query.py "How does jax.Array sharding work?" --top-k 8
python3 scripts/query.py "What is jax.experimental.pallas?" --top-k 8
```

JSON output for agents or scripts:

```bash
python3 scripts/query.py "Explain pytrees at a high level" --json --top-k 8
```

### 6. Use the agent skill

The bundled skill lives at:

```text
skills/jax-docs/
```

Use it in whichever way your agent runtime supports:

- Point the agent at this repo and tell it to use `skills/jax-docs/`.
- Copy the skill folder into your normal skills directory.
- Symlink the skill folder into your normal skills directory.

For a pasteable agent instruction, use `PROMPT.md`.

### 7. Rebuild only when needed

Rebuild if the JAX docs change or if you change the crawler, chunker, or indexing logic:

```bash
python3 scripts/crawl_jax_docs.py
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```

`scripts/crawl_jax_docs.py` requires `pandoc` because it uses it as the Markdown conversion tool.

## How It Works

### 1. Source documents

The source root is:

```text
https://docs.jax.dev/en/latest/
```

The crawler stores:

- per-route Markdown pages in `docs/pages/`
- per-route source HTML in `docs/html/`
- a generated concatenation in `docs/jax-docs.md`
- route coverage records in `data/routes.jsonl`
- Sphinx API inventory records in `data/objects.jsonl`
- coverage summary in `data/coverage.json`

### 2. Route discovery

The crawler combines:

- `searchindex.js` page docnames
- `objects.inv` Python object/API targets
- recursively discovered same-host HTML links

This is intended to cover normal docs pages plus autosummary pages for functions, classes, methods, and properties.

### 3. Chunking

`scripts/chunk.py` reads the generated Markdown pages and writes:

- `data/chunks.jsonl`: route-aware text chunks
- `data/sections.jsonl`: heading records plus object inventory records

Each chunk keeps source path, source URL, route, heading path, line number, and searchable text.

### 4. Local index

`scripts/build_index.py` builds `index/jax-docs-tfidf.joblib` from the chunks, headings, and API object records.

The index uses scikit-learn TF-IDF vectors:

- Word n-grams catch normal JAX concepts and phrases.
- Character n-grams help with exact API names such as `jax.grad`, `jax.numpy.sum`, and `jax.Array.addressable_shards`.
- Object records improve recall for exact API/function questions.

This is not an embedding API and it does not require network access at query time.

Source docs:

https://docs.jax.dev/
