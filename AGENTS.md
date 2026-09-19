# PageIndex

Fork of VectifyAI's vectorless, reasoning-based RAG: builds a hierarchical tree index from long documents (PDF, Markdown, plain text) for LLM tree-search retrieval. This fork adds txt-file support and a pluggable LLM provider layer.

---

## Toolchain: Python, uv & tox

- `requires-python = ">=3.10"`; dev tooling through tox with tox-uv. Run checks via `tox -e <env>`, never bare `ruff`/`mypy`/`pytest`.
- Core deps are only `python-dotenv` and `pyyaml`; PDF parsing, OpenAI, and Anthropic are optional extras (`pdf`, `openai`, `anthropic`, `all` in `pyproject.toml`). Code must degrade gracefully when an extra is missing.

---

## Compiler Checks (CRITICAL)

```bash
tox -e lint    # ruff check + ruff format --check (read-only)
tox -e type    # mypy pageindex
tox -e fix     # ruff check --fix + ruff format (rewrites files; scoped to pageindex/ and tests/)
```

Ruff config: line-length 100, rules `E W F I B UP` (`pyproject.toml [tool.ruff]`). Mypy targets `pageindex` only, `ignore_missing_imports = true`.

---

## Testing

```bash
tox -e test    # pytest tests --cov=pageindex --cov-report=term-missing
```

Tests live in `tests/` (`test_llm_provider.py`, `test_page_index_txt.py`); asyncio mode is `auto`. Fixture PDFs in `tests/pdfs/`, expected tree outputs in `tests/results/`. Tests are offline — mocking the LLM provider is the norm; never hit a live API in tests.

---

## PageIndex — This Project

Layout:

- `pageindex/` — the package. `page_index.py` (PDF pipeline), `page_index_md.py` (Markdown), `page_index_txt.py` (plain text, this fork's addition), `llm_provider.py` (provider abstraction: OpenAI, Anthropic, user callback, Claude Code CLI), `utils.py` (shared LLM/token helpers), `config.yaml` (default options, model `gpt-4o-2024-11-20`)
- `run_pageindex.py` — CLI entry point: exactly one of `--pdf_path`, `--md_path`, `--txt_path`; writes `<name>_structure.json` to `results/`
- `cookbook/`, `tutorials/` — upstream Jupyter notebooks and guides; not part of the package

Domain rules:

- LLM calls go through `get_llm_provider()` in `pageindex/llm_provider.py`, not direct SDK calls. API keys come from `.env`: `CHATGPT_API_KEY` (falls back to `OPENAI_API_KEY`) or `ANTHROPIC_API_KEY`. Never read or print `.env`.
- `ClaudeCodeProvider` shells out to the `claude` CLI; `is_claude_code_available()` gates it.
- The `yes`/`no` option flags (`if_add_node_id`, etc.) are strings, not booleans — preserve that convention end to end.
- Markdown tree building infers hierarchy from `#` heading levels only; converted PDF/HTML markdown loses hierarchy (README warning).
