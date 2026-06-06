#!/usr/bin/env python3
"""Smoke tests for the JAX docs query index."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_query(question: str) -> str:
    command = [sys.executable, str(ROOT / "scripts" / "query.py"), question, "--top-k", "5"]
    return subprocess.check_output(command, cwd=ROOT, text=True)


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

    def test_grad_query_finds_autodiff_context(self) -> None:
        output = run_query("What does jax.grad do?")
        lowered = output.lower()
        self.assertIn("jax.grad", lowered)
        self.assertTrue(any(term in lowered for term in ("gradient", "differentiat", "autodiff")), output)


if __name__ == "__main__":
    unittest.main()
