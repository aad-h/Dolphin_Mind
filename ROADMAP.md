# DolphinAgent Roadmap

Turning DolphinMind (retrieval for small LMs) into **DolphinAgent**: a deployed, measured,
multi-agent research system over long documents.

**Goal by the end:** a live demo + a benchmark table + a written ablation study, so the
resume bullet can carry real numbers:

> Built a multi-agent long-document research system (planner/worker/verifier, 6 tools)
> achieving X% accuracy on a 400-question benchmark vs Y% for standard RAG at Z× lower
> cost; deployed with tracing and a streaming UI.

---

## Phase 1: Agent core (week 1)
- [x] `LLM` protocol so any backend plugs in (local Qwen, OpenAI/Anthropic API, test fakes)
- [x] Tool registry with typed args: `search_doc`, `grep`, `read_section`, `calculator`
- [x] ReAct loop with JSON tool calls, step budget, error recovery and full trace
- [x] Citations: every answer references chunk ids it actually read
- [x] Unit tests with a scripted fake LLM (no GPU)
- [x] API-model backend (OpenAI-compatible) so the agent runs fast while developing
- [x] `python -m dolphinmind.agent "question" --doc file.txt` CLI

## Phase 2: Evaluation harness (week 2): the part that stands out
- [ ] Benchmark of 200–400 questions over 3–5 long public-domain books:
      single-hop, multi-hop, numeric/aggregation, unanswerable
- [ ] Metrics: accuracy, citation faithfulness (is the claim in the cited chunk?),
      tokens & $ per question, latency, tool calls per question
- [x] Baselines: truncation, plain RAG, DolphinMind retrieval, single ReAct agent
- [x] `run_agent_eval.py` → CSV + summary table with bootstrap confidence intervals

## Phase 3: Multi-agent (week 3)
- [ ] Planner: decomposes question into sub-questions
- [ ] Workers: research sub-questions in parallel (asyncio)
- [ ] Verifier: checks each claim against its cited chunk; retries or flags "not found"
- [ ] Compare single vs multi-agent on the benchmark (accuracy vs cost)

## Phase 4: Product (week 4)
- [ ] FastAPI backend with streaming (SSE) of agent steps
- [ ] Small Next.js UI: upload doc, ask, watch tool calls live, click citations
- [ ] Tracing (OpenTelemetry/Langfuse), response cache, per-query cost budget
- [ ] Deploy (Render/Fly for API, Vercel for UI) + public demo link in README

## Phase 5: Data science write-up (week 5)
- [ ] Ablations: chunk size, top-k, reranker on/off, small vs large model, step budget
- [ ] Error analysis: categorize 100 failures (retrieval miss, reasoning, hallucinated citation…)
- [ ] `REPORT.md` / blog post with charts and honest limitations
- [ ] Final README: architecture diagram, results table, demo GIF, resume-ready summary

---

### Interview talking points this creates
- **SWE:** tool runtime design, async multi-agent orchestration, streaming API, deployment
- **Product:** UX of trust (citations, "I don't know", live steps), cost/latency trade-offs
- **DS:** metric design, ablations, confidence intervals, failure taxonomy

---

## Scheduled backlog (worked top to bottom, 1–2 commits per session)

Each item must ship with GPU-free tests (use `ScriptedLLM` / mocked HTTP; no
API keys or downloads in tests), keep `pytest` green, and update the README if
user-visible. Tick the box in the same commit that finishes it.

- [x] 1. `OpenAICompatLLM` backend (`dolphinmind/agent/llm.py`): any OpenAI-compatible endpoint via `httpx`, key from `DOLPHIN_API_KEY` env var, retries + timeout, token usage tracking; tests with a mocked transport
- [x] 2. CLI: `python -m dolphinmind.agent "question" --doc book.txt [--backend scripted|openai|hf] [--trace]` printing the answer, citations and the step trace
- [x] 3. Eval dataset format + generator: JSONL schema (question, answer, type, evidence chunk text) and a synthetic generator for single-hop, multi-hop, numeric and unanswerable questions over long text
- [x] 4. Metrics (`dolphinmind/eval/metrics.py`): exact/fuzzy accuracy, citation faithfulness (answer evidence present in cited chunks), tool calls, tokens, latency; bootstrap 95% CIs
- [x] 5. `run_agent_eval.py`: compare truncation, plain RAG, DolphinMind retrieval and the ReAct agent on a dataset; CSV + markdown summary table
- [ ] 6. Multi-agent: planner (decompose) → workers (research sub-questions) → verifier (check each claim against its cited chunk, retry or answer "not found")
- [ ] 7. Run workers in parallel with `asyncio` + per-query token/cost budget enforcement
- [ ] 8. FastAPI service: `POST /ask` and `GET /ask/stream` (SSE streaming each agent step), document upload, in-memory doc cache
- [ ] 9. Single-page web UI (static HTML/JS served by FastAPI): upload doc, ask, watch tool calls stream live, click citations to highlight chunks
- [ ] 10. Tracing (OpenTelemetry spans per LLM/tool call), response caching, Dockerfile, architecture diagram in README
