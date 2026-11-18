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
- [ ] API-model backend (OpenAI-compatible) so the agent runs fast while developing
- [ ] `python -m dolphinmind.agent "question" --doc file.txt` CLI

## Phase 2: Evaluation harness (week 2): the part that stands out
- [ ] Benchmark of 200–400 questions over 3–5 long public-domain books:
      single-hop, multi-hop, numeric/aggregation, unanswerable
- [ ] Metrics: accuracy, citation faithfulness (is the claim in the cited chunk?),
      tokens & $ per question, latency, tool calls per question
- [ ] Baselines: truncation, plain RAG, DolphinMind retrieval, single ReAct agent
- [ ] `run_agent_eval.py` → CSV + summary table with bootstrap confidence intervals

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
