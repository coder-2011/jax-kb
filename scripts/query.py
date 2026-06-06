#!/usr/bin/env python3
"""Query the local JAX documentation index."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "index" / "jax-docs-tfidf.joblib"

TOPIC_EXPANSIONS = {
    "jit": "jax.jit compilation tracing lowering static_argnums static_argnames donate_argnums sharding pjit xla compilation cache",
    "grad": "jax.grad automatic differentiation value_and_grad jacfwd jacrev vjp jvp custom_vjp custom_jvp holomorphic allow_int",
    "vmap": "jax.vmap vectorization batching in_axes out_axes axis_name spmd_axis_name",
    "pmap": "jax.pmap parallel map devices axis_name collectives sharding",
    "random": "jax.random PRNGKey key split fold_in typed keys random numbers",
    "array": "jax.Array sharding device committed addressable_shards global_shards numpy ndarray",
    "sharding": "jax.sharding Mesh NamedSharding PartitionSpec devices pjit shard_map",
    "transform": "jit grad vmap pmap scan remat checkpoint jvp vjp transformations tracing jaxpr",
    "compilation": "jit lower compile ahead-of-time XLA lowering stages export",
    "debug": "jax.debug.print callback checkify debug_nans transfer_guard",
    "pallas": "jax.experimental.pallas kernel BlockSpec GridSpec TPU GPU",
    "numpy": "jax.numpy ndarray NumPy-compatible functions broadcasting dtype promotion",
    "broad": "JAX transformations jit grad vmap pmap arrays sharding pytrees jaxpr compilation automatic differentiation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="question to ask")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    return parser.parse_args()


def expand_query(question: str) -> str:
    lowered = question.lower()
    expansions = []
    has_symbol = bool(re.search(r"\bjax(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", question))
    if not has_symbol and re.search(r"\b(how|what|why|overview|explain|broad|general|think about)\b", lowered):
        expansions.append(TOPIC_EXPANSIONS["broad"])
    for trigger, expansion in TOPIC_EXPANSIONS.items():
        if trigger in lowered and expansion not in expansions:
            expansions.append(expansion)
    return question + "\n" + "\n".join(expansions)


def excerpt(text: str, max_chars: int = 1000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def keyword_score(question: str, doc: dict) -> float:
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_:+./-]*", question)
        if len(term) > 2
    }
    if not query_terms:
        return 0.0
    haystack = f"{doc.get('route', '')} {doc.get('heading', '')} {doc.get('text', '')}".lower()
    hits = sum(1 for term in query_terms if term in haystack)
    score = min(hits / max(len(query_terms), 1), 1.0) * 0.08
    for symbol in re.findall(r"\bjax(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", question):
        if symbol.lower() in haystack:
            score += 0.35
            if symbol.lower() in haystack[:1500]:
                score += 0.12
    if doc.get("kind") == "object" and any(term in doc.get("title", "").lower() for term in query_terms):
        score += 0.08
    return score


def is_broad_question(question: str) -> bool:
    lowered = question.lower()
    has_symbol = bool(re.search(r"\bjax(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", question))
    return not has_symbol and bool(
        re.search(r"\b(how|why|overview|explain|broad|general|think about|should i)\b", lowered)
    )


def search(index_path: Path, question: str, top_k: int) -> list[dict]:
    index = joblib.load(index_path)
    expanded = expand_query(question)
    word_query = index["word_vectorizer"].transform([expanded])
    char_query = index["char_vectorizer"].transform([expanded])
    query_matrix = normalize(sparse.hstack([word_query, char_query], format="csr"), copy=False)
    scores = (index["matrix"] @ query_matrix.T).toarray().ravel()

    ranked = []
    for row, base_score in enumerate(scores):
        doc = index["docs"][row]
        score = float(base_score) + keyword_score(question, doc)
        if doc.get("kind") in {"section", "object"} and is_broad_question(question):
            score += 0.012
        elif doc.get("kind") in {"section", "object"}:
            score -= 0.02
        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    results = []
    seen = set()
    for score, doc in ranked:
        key = (doc.get("kind"), doc.get("route"), doc.get("heading"))
        if key in seen:
            continue
        seen.add(key)
        result = dict(doc)
        result["score"] = round(score, 4)
        result["excerpt"] = excerpt(doc.get("text", ""))
        results.append(result)
        if len(results) >= top_k:
            break
    return results


def print_markdown(question: str, results: list[dict]) -> None:
    print(f"# JAX Docs Query\n\n**Question:** {question}\n")
    for rank, result in enumerate(results, start=1):
        heading = result.get("heading") or result.get("title") or "JAX documentation"
        route = result.get("route", "?")
        kind = result.get("kind", "chunk")
        print(f"## {rank}. {heading}")
        print(f"- Score: `{result['score']}`")
        print(f"- Type: `{kind}`")
        print(f"- Route: `{route}`")
        print(f"- URL: `{result.get('url', '')}`")
        print(f"- Source: `{result.get('source', '')}`")
        if result.get("line"):
            print(f"- Line: `{result['line']}`")
        print("")
        print(result["excerpt"])
        print("")


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()
    results = search(args.index, question, args.top_k)
    if args.json:
        print(json.dumps({"question": question, "results": results}, indent=2, ensure_ascii=False))
    else:
        print_markdown(question, results)


if __name__ == "__main__":
    main()
