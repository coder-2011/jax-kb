#!/usr/bin/env python3
"""Smoke tests for the JAX docs query index."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_query(question: str) -> str:
    command = [sys.executable, str(ROOT / "scripts" / "query.py"), question, "--top-k", "5"]
    return subprocess.check_output(command, cwd=ROOT, text=True)


def run_query_json(question: str) -> dict:
    command = [sys.executable, str(ROOT / "scripts" / "query.py"), question, "--top-k", "5", "--json"]
    return json.loads(subprocess.check_output(command, cwd=ROOT, text=True))


def run_query_json_full_text(question: str) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "query.py"),
        question,
        "--top-k",
        "1",
        "--json",
        "--full-text",
    ]
    return json.loads(subprocess.check_output(command, cwd=ROOT, text=True))


class QuerySmokeTests(unittest.TestCase):
    def test_jit_tracing_query_finds_relevant_context(self) -> None:
        output = run_query("How should I think about jit and tracing?")
        lowered = output.lower()
        self.assertIn("route:", lowered)
        self.assertTrue(any(term in lowered for term in ("jit", "tracing", "compilation", "jaxpr")), output)

    def test_exact_vmap_query_finds_vmap(self) -> None:
        output = run_query("What does jax.vmap do?")
        lowered = output.lower()
        self.assertIn("jax.vmap", lowered)
        self.assertIn("route:", lowered)

    def test_exact_vmap_query_prefers_explanatory_chunk(self) -> None:
        data = run_query_json("What does jax.vmap do?")
        first = data["results"][0]
        self.assertEqual(first["kind"], "chunk")
        self.assertEqual(first["route"], "_autosummary/jax.vmap")
        self.assertIn("Vectorizing map", first["excerpt"])
        self.assertNotIn("Source repository", first["excerpt"][:250])

    def test_exact_property_query_prefers_explanatory_chunk(self) -> None:
        data = run_query_json("How does jax.Array.addressable_shards work?")
        first = data["results"][0]
        self.assertEqual(first["kind"], "chunk")
        self.assertEqual(first["route"], "_autosummary/jax.Array.addressable_shards")
        self.assertIn("List of addressable shards", first["excerpt"])

    def test_grad_query_finds_autodiff_context(self) -> None:
        output = run_query("What does jax.grad do?")
        lowered = output.lower()
        self.assertIn("jax.grad", lowered)
        self.assertTrue(any(term in lowered for term in ("gradient", "differentiat", "autodiff")), output)

    def test_debug_nan_query_returns_debugging_context(self) -> None:
        data = run_query_json("How do I debug NaNs in jit?")
        top_routes = {result["route"] for result in data["results"][:3]}
        top_text = " ".join(result["excerpt"] for result in data["results"][:3]).lower()
        self.assertTrue({"debugging", "notebooks/thinking_in_jax"} & top_routes, data)
        self.assertIn("debug_nans", top_text)
        self.assertIn("jit", top_text)

    def test_broad_pytree_query_returns_readable_context(self) -> None:
        data = run_query_json("Explain pytrees at a high level")
        first = data["results"][0]
        self.assertEqual(first["kind"], "chunk")
        self.assertIn(first["route"], {"pytrees", "key-concepts"})
        self.assertIn("pytree", first["excerpt"].lower())

    def test_json_is_compact_unless_full_text_is_requested(self) -> None:
        compact = run_query_json("What does jax.vmap do?")["results"][0]
        self.assertNotIn("text", compact)
        self.assertNotIn("search_text", compact)

        full_text = run_query_json_full_text("What does jax.vmap do?")["results"][0]
        self.assertIn("text", full_text)
        self.assertIn("search_text", full_text)


if __name__ == "__main__":
    unittest.main()
