# Agent Setup Prompt

Paste this into the agent that should set up and use the JAX Documentation Knowledge Base.

```text
You are setting up and using the JAX Documentation Knowledge Base:

https://github.com/coder-2011/jax-kb

Make the KB usable in the current environment, then use it to answer JAX questions with evidence from the docs.

Rules:
- Do not create a virtual environment unless explicitly asked.
- Use the Python environment that already fits the machine or project.
- Inspect the repo before acting. Read the README, AGENTS.md, scripts, tests, and requirements as needed.
- If these instructions are incomplete or inaccurate, figure out the correct path from the codebase and take the necessary action.
- Do not add an MCP server. This KB is local files plus scripts.

Setup:
1. Clone or update the repo.

   git clone https://github.com/coder-2011/jax-kb.git
   cd jax-kb

   If it already exists, enter it and pull the latest changes.

2. Install the bundled agent skill.

   If the agent runtime has a known skills directory, symlink the skill there:

   SKILL_HOME="${CODEX_HOME:-$HOME/.codex}/skills"
   mkdir -p "$SKILL_HOME"
   ln -sfn "$PWD/skills/jax-docs" "$SKILL_HOME/jax-docs"

   If symlinks are not supported, copy `skills/jax-docs` into the runtime's skills directory instead. If the runtime can load repo-local skills directly, point it at that folder.

3. Check dependencies before installing anything.

   python3 -c 'import importlib.util; mods=["joblib","numpy","sklearn","scipy"]; print("missing:", [m for m in mods if importlib.util.find_spec(m) is None])'

4. If packages are missing, install `requirements.txt` with the least disruptive tool available in the environment.

   python3 -m pip install -r requirements.txt

5. Verify the KB works.

   python3 scripts/query.py "How should I think about jit and tracing?" --top-k 4
   python3 scripts/query.py "What does jax.vmap do?" --top-k 4
   python3 -m unittest discover -s tests

Use:
- Use `$jax-docs` when the skill is available.
- Query the KB before answering factual JAX questions.
- Read multiple returned results.
- For broad questions, use broad queries with `--top-k 8` or higher.
- For API, class, method, or property questions, include the exact name.
- Use `--json` when structured retrieval output is useful.
- Cite docs URLs/routes from the returned results.
- If the KB is not enough, say what is docs-grounded and what is outside reasoning.

Example queries:

python3 scripts/query.py "What parts of JAX matter most for writing fast numerical code?" --top-k 10
python3 scripts/query.py "What does jax.grad do?" --top-k 8
python3 scripts/query.py "Explain pytrees at a high level" --json --top-k 8
```
