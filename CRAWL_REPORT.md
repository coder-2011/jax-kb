# JAX Docs Crawl Report

Generated from `https://docs.jax.dev/en/latest/`.

## Acquisition Method

The crawler uses three Sphinx-backed discovery sources:

- `searchindex.js` for rendered documentation routes.
- `objects.inv` for API, function, class, method, and property targets.
- Same-host HTML link discovery from fetched documentation pages.

Each fetched page is converted from HTML to Markdown with `pandoc` and stored under `docs/pages/`. Raw HTML is stored under `docs/html/`. The concatenated Markdown corpus is `docs/jax-docs.md`.

## Coverage

Current `data/coverage.json` reports:

- Fetched routes: 1,713
- Failed expected routes: 0
- Sphinx search-index routes: 1,706
- Missing search-index routes: 0
- Sphinx inventory-backed routes: 1,709
- Missing inventory routes: 0
- Sphinx inventory objects: 3,985

The crawler also saw 3 stale links that return HTTP 404 and are not present in either Sphinx source:

- `https://docs.jax.dev/en/latest/jax.experimental.host_callback.html`
- `https://docs.jax.dev/en/latest/design_notes/jax_versioning.html`
- `https://docs.jax.dev/index.html`

Those are recorded as `skipped_not_found` in `data/coverage.json`, not as missing documentation routes.

## Generated KB

Generated retrieval files:

- `data/routes.jsonl`: one record per fetched documentation route.
- `data/objects.jsonl`: Sphinx inventory object records.
- `data/chunks.jsonl`: route-aware text chunks.
- `data/sections.jsonl`: heading records plus inventory object records.
- `index/jax-docs-tfidf.joblib`: local query index.

Validation commands:

```bash
python3 scripts/crawl_jax_docs.py
python3 scripts/chunk.py
python3 scripts/build_index.py
python3 -m unittest discover -s tests
```

In the current Homebrew-managed Python environment, validation was run with:

```bash
uv run --with joblib --with numpy --with scikit-learn --with scipy python scripts/build_index.py
uv run --with joblib --with numpy --with scikit-learn --with scipy python -m unittest discover -s tests
```
