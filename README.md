# RepoVisor

AI repository onboarding: GitHub + MCP + AST code intelligence + RAG + architecture graph.

## Run
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export GEMINI_API_KEY=...   # free key from aistudio.google.com/apikey
    uvicorn repovisor.app:app --reload        # dashboard at http://localhost:8000
    python -m repovisor.mcp_server            # MCP server (stdio), usable from Claude Desktop

Paste a GitHub URL (or local path) in the dashboard, click Index, then ask questions.
For private repos use a fine-grained, read-only token. It is sent as a git header and never stored or put in prompts.

## Pipeline (maps to synopsis sections 4-5)
1. Connect/clone: `core.clone`
2. AST + symbols + imports + call edges: `core.parse_py` (Python ast), `core.parse_js` (regex, MVP)
3. Knowledge layer: chunks + bge-small embeddings (numpy) + JSON metadata in `data/<repo_id>/`
4. Retrieval: cosine similarity plus keyword boost, then dependency facts for top files
5. Agent: Gemini (or Claude/Groq) answers only from evidence, cites [path:start-end]; citations not in the retrieved set are flagged
6. Architecture view: force-directed dependency graph, answer sources highlighted
7. Docs: `GET /api/repos/{id}/doc`
8. MCP tools (read-only): repo_metadata, search_code, get_file, find_symbol, dependencies, who_calls, commit_history

## Known limits / upgrade path (good viva material)
- JS/TS parsing is regex-based. Upgrade to tree-sitter for real ASTs and more languages.
- Vector store is in-memory numpy. Swap to Chroma/pgvector/Qdrant when repos get large.
- Call graph matches by name, not full type resolution.
- Add a reranker, AST-aware chunking by symbol, and incremental re-indexing by commit diff.

## Evaluation plan to implement (synopsis section 8)
- Build 20-30 questions per sample repo with ground-truth files. Measure Recall@k and MRR for retrieval.
- Grounding: share of cited files present in retrieved evidence (already returned as `unverified_citations`).
- Onboarding efficiency: timed tasks with and without the tool, 5-10 users.
- Performance: log indexing time and query latency.
