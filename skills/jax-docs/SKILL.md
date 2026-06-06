---
name: jax-docs
description: Ground JAX answers in the local docs.jax.dev knowledge base. Use for JAX transformations, jit, grad, vmap, pmap, arrays, pytrees, sharding, jax.numpy, jax.lax, random, Pallas, callbacks, errors, API functions/classes/methods/properties, and broad JAX questions where the agent is unsure or should cite docs URLs. Pure skill only; no MCP server. Query the repo CLI before answering.
---

# JAX Docs

Use the local JAX documentation index before answering JAX questions from memory.

## Locate the Repo

Run commands from the repo root containing `scripts/query.py`. If needed:

```bash
cd /home/ubuntu/jax-kb
```

## Query First

For any factual JAX answer, query before answering:

```bash
python3 scripts/query.py "QUESTION" --top-k 8
```

Use JSON when another tool or script will consume the results:

```bash
python3 scripts/query.py "QUESTION" --json --top-k 8
```

## Query Strategy

For broad questions, keep the user wording broad and ask for more results:

```bash
python3 scripts/query.py "How should I think about jit and tracing?" --top-k 10
python3 scripts/query.py "What parts of JAX matter most for writing fast numerical code?" --top-k 10
```

For exact API, class, method, or property questions, include the exact name:

```bash
python3 scripts/query.py "What does jax.grad do?" --top-k 8
python3 scripts/query.py "What does jax.vmap do?" --top-k 8
python3 scripts/query.py "How does jax.Array.addressable_shards work?" --top-k 8
```

If the first result set is weak, rerun once with a broader phrase and `--top-k 12`:

```bash
python3 scripts/query.py "JAX transformations tracing jit grad vmap jaxpr" --top-k 12
```

## Answering

- Read several returned excerpts, not only the first one.
- Prefer chunks over bare section or object records for detailed claims.
- Use section/object records as a map when the question is broad or API-exact.
- Cite docs routes or URLs when making factual JAX claims.
- Separate docs-grounded facts from outside reasoning.
- If the local index does not contain enough evidence, say so plainly.

## Common Failure Modes

- Do not answer JAX details from memory before querying.
- Do not treat query expansion terms as answers; they only help retrieval.
- Do not cite a route unless that route appears in the returned results.
- Do not add an MCP server. This skill is intentionally file-and-script only.
