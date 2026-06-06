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
    "nan": "debug_nans jax.debug_nans jax_debug_nans disable_jit checkify debug flags NaN Inf",
    "nans": "debug_nans jax.debug_nans jax_debug_nans disable_jit checkify debug flags NaN Inf",
    "pallas": "jax.experimental.pallas kernel BlockSpec GridSpec TPU GPU",
    "pytree": "pytrees tree_util tree_map tree_flatten tree_unflatten leaves treedef nested containers transformations",
    "pytrees": "pytrees tree_util tree_map tree_flatten tree_unflatten leaves treedef nested containers transformations",
    "numpy": "jax.numpy ndarray NumPy-compatible functions broadcasting dtype promotion",
    "broad": "JAX transformations jit grad vmap pmap arrays sharding pytrees jaxpr compilation automatic differentiation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="question to ask")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1200, help="maximum excerpt characters per result")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--full-text", action="store_true", help="include full text/search_text fields in JSON output")
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


def clean_display_text(text: str) -> str:
    text = re.sub(r"\[\\?#\]\([^)]*\)", "", text)
    text = re.sub(r"\[¶\]\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    if text.startswith("- []("):
        first_heading = text.find(" # ")
        if 0 <= first_heading < 900:
            text = text[first_heading + 1 :]
    return text.strip()


def query_terms(question: str) -> list[str]:
    terms = []
    for term in re.findall(r"[A-Za-z_][A-Za-z0-9_:+./-]*", question):
        if len(term) > 2 and term.lower() not in {"what", "how", "does", "work", "with", "the", "and"}:
            terms.append(term)
    return terms


def extract_symbols(question: str) -> list[str]:
    return re.findall(r"\bjax(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", question)


def first_relevant_position(text: str, question: str) -> int:
    lowered = text.lower()
    candidates = extract_symbols(question) + query_terms(question)
    if "nan" in question.lower():
        candidates = ["debug_nans", "jax_debug_nans"] + candidates
    positions = [lowered.find(candidate.lower()) for candidate in candidates]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else 0


def excerpt(text: str, question: str, max_chars: int = 1200) -> str:
    text = clean_display_text(text)
    if len(text) <= max_chars:
        return text
    position = first_relevant_position(text, question)
    start = max(position - max_chars // 5, 0)
    end = start + max_chars
    if end > len(text):
        end = len(text)
        start = max(end - max_chars, 0)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def keyword_score(question: str, doc: dict) -> float:
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_:+./-]*", question)
        if len(term) > 2
    }
    if not terms:
        return 0.0
    haystack = f"{doc.get('route', '')} {doc.get('heading', '')} {doc.get('text', '')}".lower()
    hits = sum(1 for term in terms if term in haystack)
    score = min(hits / max(len(terms), 1), 1.0) * 0.08
    for symbol in re.findall(r"\bjax(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", question):
        if symbol.lower() in haystack:
            score += 0.35
            if symbol.lower() in haystack[:1500]:
                score += 0.12
    if doc.get("kind") == "object" and any(term in doc.get("title", "").lower() for term in terms):
        score += 0.08
    return score


def route_matches_symbol(route: str, symbol: str) -> bool:
    route = route.lower().rstrip("/")
    symbol = symbol.lower()
    return route in {symbol, f"_autosummary/{symbol}"}


def is_stub_record(doc: dict) -> bool:
    return doc.get("kind") in {"object", "section"} and len(doc.get("text", "")) < 260


def usability_score(question: str, doc: dict) -> float:
    score = 0.0
    route = doc.get("route", "")
    title = doc.get("title", "")
    text = doc.get("text", "")
    kind = doc.get("kind")
    symbols = extract_symbols(question)
    terms = query_terms(question)

    if kind == "chunk":
        score += 0.04
        route_title = f"{route} {title}".lower()
        for term in terms:
            term = term.lower()
            if term in {route.lower(), title.lower()}:
                score += 0.16
            elif term in route_title:
                score += 0.03
        if "custom" in route_title and "custom" not in {term.lower() for term in terms}:
            score -= 0.08
        if "nan" in question.lower() and "debug_nans" in text.lower():
            score += 0.16
    elif is_stub_record(doc):
        score -= 0.18 if not symbols else 0.08

    for symbol in symbols:
        if route_matches_symbol(route, symbol):
            if kind == "chunk":
                score += 0.55
            elif kind == "section":
                score += 0.02
            elif kind == "object":
                score += 0.01
        elif kind == "chunk" and symbol.lower() in text.lower():
            score += 0.12

    return score


def is_broad_question(question: str) -> bool:
    lowered = question.lower()
    has_symbol = bool(re.search(r"\bjax(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", question))
    return not has_symbol and bool(
        re.search(r"\b(how|why|overview|explain|broad|general|think about|should i)\b", lowered)
    )


def search(index_path: Path, question: str, top_k: int, max_chars: int = 1200) -> list[dict]:
    index = joblib.load(index_path)
    expanded = expand_query(question)
    word_query = index["word_vectorizer"].transform([expanded])
    char_query = index["char_vectorizer"].transform([expanded])
    query_matrix = normalize(sparse.hstack([word_query, char_query], format="csr"), copy=False)
    scores = (index["matrix"] @ query_matrix.T).toarray().ravel()

    ranked = []
    for row, base_score in enumerate(scores):
        doc = index["docs"][row]
        score = float(base_score) + keyword_score(question, doc) + usability_score(question, doc)
        if doc.get("kind") == "section" and is_broad_question(question) and not is_stub_record(doc):
            score += 0.012
        elif doc.get("kind") in {"section", "object"}:
            score -= 0.02
        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    results = []
    seen = set()
    content_routes_seen = set()
    for score, doc in ranked:
        key = (doc.get("kind"), doc.get("route"), doc.get("heading"))
        if key in seen:
            continue
        route = doc.get("route")
        if is_stub_record(doc) and route in content_routes_seen:
            continue
        seen.add(key)
        result = dict(doc)
        result["heading"] = clean_display_text(result.get("heading", ""))
        result["score"] = round(score, 4)
        result["excerpt"] = excerpt(doc.get("text", ""), question, max_chars=max_chars)
        results.append(result)
        if doc.get("kind") == "chunk" and route:
            content_routes_seen.add(route)
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


def json_result(result: dict, full_text: bool) -> dict:
    if full_text:
        return result
    compact = dict(result)
    compact.pop("text", None)
    compact.pop("search_text", None)
    return compact


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()
    results = search(args.index, question, args.top_k, max_chars=args.max_chars)
    if args.json:
        print(
            json.dumps(
                {"question": question, "results": [json_result(result, args.full_text) for result in results]},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print_markdown(question, results)


if __name__ == "__main__":
    main()
