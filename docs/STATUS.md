# DeepResearch Playground — Status, Problems, and Gaps

This document inventories: (1) the original research aim and phased plan
you set out with, (2) the LangGraph migration plan I wrote in response,
(3) what actually shipped, (4) what works today and what's still broken,
(5) what's deferred from the original plan, and (6) honest critique of
the gaps between the plan and the thesis aim.

Date: 2026-05-12.

---

## 1. Original Research Aim (your initial message)

> Build a research playground that co-schedules a single local RTX 4090
> (running a ≤20B "minimizer/personalizer" model + private memory + RAG)
> against a 4×RTX 4090 cloud node (running Llama-3.3-70B-Instruct AWQ or
> Qwen-2.5-72B-AWQ via vLLM/SGLang) to traverse a **3-D Pareto frontier
> of latency × privacy-leakage × joint GPU utilization** for deep-research
> workflows.

Three streams to integrate:

1. **PD-disaggregation / agentic schedulers** — DistServe, Llumnix,
   Mooncake, Parrot, ECO-LLM.
2. **Hierarchical memory OSes** — MemOS, Letta/MemGPT, A-Mem, Mem0,
   Zep/Graphiti, ReMe.
3. **Contextual-integrity privacy enforcement** for multi-agent traffic
   — AirGapAgent, PrivacyLens, MAGPIE, AgentLeak.

Novel contributions:

- **ParetoDispatch** — privacy-bounded co-scheduling.
- **Reflection Broadcast Protocol** — CI-filtered reflections that
  propagate to local memory **and** laterally to cloud subagents under
  a mutual-information budget.
- **Hot/warm/cold hierarchical memory placement** across local-GPU /
  local-SSD / cloud tiers.
- **BrowseComp-Hybrid** — BrowseComp + GAIA + LoCoMo extension.

---

## 2. Original Phased Plan (your phrasing, condensed)

| Phase | Goal | Key deliverables |
|---|---|---|
| **1. Baseline DeepResearch with ReMe** | runnable STORM-style loop | planner/searcher/reader/synthesizer/reflector agents; ReMe personal/task/tool memory; Qdrant; FastAPI + Typer CLI; OpenAI-compatible `ModelClient` |
| **2. Evaluation & instrumentation** | measurable research tool | 10–20 golden tasks; metrics (citations, latency, tokens, memory reuse); structured logs; ablations across memory profiles |
| **3. Hybrid local/cloud routing** | local + cloud split | ≤20B local minimizer; 70B AWQ cloud via vLLM/SGLang; per-call telemetry (NVML/DCGM, queue time, bytes-to-cloud) |
| **4. Privacy-bounded ParetoDispatch** | the research contribution | `PrivacyEnvelope` on every message/memory/call; CI labels (subject/sender/recipient/attribute/principle/sensitivity); leakage proxies (sensitive-token exposure, MI budget); ParetoDispatch jointly optimizing latency × privacy × GPU |
| **5. Reflection Broadcast + hierarchical memory** | novel memory contributions | CI-filtered reflection emission; lateral cloud broadcast under MI budget; hot/warm/cold placement policy |
| **6. BrowseComp-Hybrid** | thesis evaluation | task suite extending BrowseComp/GAIA/LoCoMo with private prefs + privacy budget + routing constraints; full ablation table |

Mid-stream pivot you requested: **swap the custom STORM base for
LangChain `open_deep_research`**, keeping ReMe + endpoint serving + the
Phase-4 router seam.

---

## 3. The Phase 1.5 Plan I Wrote

Recorded in `/home/lyc/.claude/plans/i-made-a-dapper-quasar.md`.
Headline:

- **In-place orchestrator swap** (no parallel subfolder).
- **Bundle the ReMe Phase-1.5 wire-up** into the same PR.
- Vendor `langchain-ai/open_deep_research @ 0dd30bd` (MIT, feature-frozen
  Aug 2025) — fork-as-reference rather than upstream-as-dependency.
- Preserve `Router.select(profile, role, envelope, hint)` signature as
  the immutable Phase-4 seam.
- Keep SQLite + `TraceRecorder` as the source of truth for thesis-grade
  metrics; LangSmith optional, env-gated.

Six structural pieces (in order of dependency):

1. `RouterChatModel(BaseChatModel)` + `RouterConfigurableModel` proxy —
   route every LLM call from any LangGraph node through `Router.select`
   into our async `ModelClient.complete()`.
2. `TraceCallbackHandler` mapping LangGraph chain events to `AgentStep`
   rows.
3. `agents/langgraph/runtime.py` — entrypoint that primes memory,
   builds the graph with a reflector node appended, sets the
   active-run contextvar, invokes, then writes reflection + working
   memory.
4. `memory/reme_flows.py` + `memory/reme_adapter.py` rewrite —
   `ReMeApp.async_execute(flow_name, **kwargs)` for personal/task
   retrieve + summary flows.
5. New role set in `ModelProfileConfig`:
   `{supervisor, researcher, compressor, final_report, reflector}`.
   Legacy STORM keys kept Optional.
6. `scripts/demo_e2e.py` — hermetic e2e demo against a fake `ModelClient`.

Planned five-commit sequence: deps → seams → orchestrator swap →
ReMe wire-up → smoke + README.

---

## 4. What Actually Shipped (git history)

### Phase 1.5 commits (my work, in order)

| SHA | Subject | Notes |
|---|---|---|
| `ea81567` | chore(deps): pin langchain 0.3.x stack + vendor open_deep_research | 5 upstream files at pinned SHA, `UPSTREAM_NOTE.md`, license file |
| `82819bf` | feat(agents): RouterChatModel + TraceCallbackHandler for LangGraph | proxy, callback handler, role_map, ModelClient `tools=` kwarg; 15 new tests |
| `385c71c` | feat(agents): swap orchestrator to LangGraph; delete STORM agents | runtime, reflector_node, memory_hooks; orchestrator becomes a shim; 5 STORM agents + prompts/ + _jsonparse deleted |
| `eeac932` | feat(memory): wire up ReMeApp via flowllm flows (Phase 1.5) | reme_flows + reme_adapter rewrite; 12 mocked tests |
| `a189ee1` | test(smoke): bundled smoke gate + README addendum on Phase 1.5 swap | smoke_e2e_bundled.sh; README rewrite |

### Studio integration (also my work)

| SHA | Subject | Notes |
|---|---|---|
| `5adc6fc` | feat(studio): wire up LangGraph Studio frontend | `langgraph.json`, `studio.py` with bootstrap node |
| `54000a9` | fix(studio): use module-level slot, not contextvar, to thread active-run | Pregel runs nodes in fresh Tasks; contextvars don't propagate |
| `6e6cabf` | chore: gitignore .langgraph_api/ checkpoint state | |
| `665a23b` | fix(studio): override upstream Configuration defaults via env vars | env vars beat `configurable.*` in `Configuration.from_runnable_config` |
| `62c3fd0` | fix(router): translate tool_choice='any' -> 'required' for vLLM compat | OpenAI renamed the value in May 2024 |

### Follow-up fixes (your hand, after I went out of context)

| SHA | Subject | What it solved |
|---|---|---|
| `e3382a4` | fix(serve+reme): vLLM tool-call parser + drop bogus reme vector_store override | vLLM needs `--enable-auto-tool-choice --tool-call-parser qwen3_xml`; flowllm CLI parser refuses dict-as-string |
| `6f51d87` | fix(reme): wire LLM credentials via env (DeepSeek, OpenAI-compat) | ReMe init crashed on missing OPENAI_API_KEY; now resolves via REME_LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY |
| `90136e9` | fix(cli+upstream): silence loguru noise + fall back when brief parser returns None | Patch 5: structured-output retry exhaustion → fall back to raw HumanMessage instead of crashing |
| `99c53d0` | fix(reme): pass init_logger=false to ReMeApp so flowllm doesn't reset loguru | |
| `dc0401a` | feat(cli): print agent-step + LLM-call tables after every run | quick post-run trace in the terminal |
| `a8dbfa6` | fix(cli): ModelCallRecord field is started_at not created_at | |

### Test surface

- 36 unit + integration tests across `tests/` — all green at the Phase
  1.5 merge point (`a189ee1`).
- One end-to-end fake-client test (`test_orchestrator_swap.py`) verifies
  local + cloud routing both fire under `co_schedule_v0`.
- `scripts/demo_e2e.py` — hermetic e2e against a fake client.
- `scripts/smoke_e2e_bundled.sh` — three hermetic gates always, plus
  `LIVE_E2E=1` / `REME_E2E=1` opt-ins.

### Pipeline shape today

```
__start__ → clarify_with_user → write_research_brief
         → research_supervisor (supervisor ↔ supervisor_tools loop
                                 spawning researcher subgraphs)
         → final_report_generation → reflector → END
```

`clarify_with_user` short-circuits when `ALLOW_CLARIFICATION=false`
(default for Studio and the CLI). The `reflector` node is ours, not
upstream's.

Every LLM call flows through `RouterChatModel` →
`Router.select(profile, role, envelope, hint)` → `ModelClient.complete`
→ OpenAI-compatible endpoint. The Phase-4 seam is intact.

---

## 5. Status — What Works Today

| Capability | Status | Notes |
|---|---|---|
| Hermetic demo (`scripts/demo_e2e.py`) | ✅ | Fake LLM, shows local + cloud routing + token counts + AgentStep trace |
| 36 unit + integration tests | ✅ | `uv run pytest` |
| CLI happy path (`uv run deepresearch run "..."`) | 🟡 | Works against a real vLLM; sensitive to model quality (see open issues) |
| FastAPI service | ✅ | `/research_runs` POST + GET + /trace; no UI |
| LangGraph Studio UI | 🟡 | Graph loads, bootstrap + clarify nodes pass; `write_research_brief` reached real vLLM (after `tool_choice` fix). Has a None-response fallback. Deeper stages untested in the browser path |
| Working memory (Qdrant embedded) | ✅ | Report persisted per run; embedded mode needs no Docker |
| ReMe personal/task retrieve + summary | 🟡 | Adapter wired; init resolves DeepSeek/OpenAI credentials; embedding endpoint open question still applies |
| Trace persistence (SQLite) | ✅ | `runtime.run_research` path. Studio path bypasses it. |
| Vendor patch documentation | ✅ | `UPSTREAM_NOTE.md` records all 5 patches incl. re-sync procedure |

---

## 6. Known Problems / Open Questions

### A. Self-driven debug loop is missing

I can't open the Studio tab; every iteration costs you a query + paste.
**Fix proposal:** a `scripts/studio_e2e.py` that drives `studio_graph.ainvoke(...)`
the same way `langgraph dev` does (real config, real ModelClient pointed at
your vLLM, fake-stub option for offline iteration). Tightens the debug
cycle to seconds.

### B. Small-model + structured-output reliability

Qwen3-8B-AWQ stumbles on `bind_tools([...]).with_structured_output(...)`
chains. Patch 5 catches it for `write_research_brief` (falls back to the
raw user message). But the **supervisor** and **researcher** subgraphs
use `bind_tools` with multiple tools (`ConductResearch`,
`ResearchComplete`, `think_tool`) and the same retry-then-None failure
mode applies. Likely failure modes still uncovered:

- Supervisor returns no `tool_calls` → exits cleanly to END (best case).
- Supervisor returns malformed `tool_calls` → JSON parsing error.
- Researcher loops without ever emitting `ResearchComplete` → hits
  `MAX_REACT_TOOL_CALLS=4` and exits with an empty `compressed_research`.

**Fix proposals (cheapest first):**

1. Add Patch-5-style fallbacks to supervisor and researcher loops, so
   None responses become a deterministic "ResearchComplete with empty
   notes" path rather than crashes.
2. Route `supervisor` to the 70B cloud model in `phase1_default` too
   (today only `co_schedule_v0` does this). Big models handle these
   schemas reliably.
3. As a last resort, add a regex/JSON fallback parser that extracts
   `{"name": ..., "args": ...}` from free-text when the proper
   tool-calls format is missing.

### C. ReMe embedding endpoint

DeepSeek doesn't serve `/v1/embeddings`, so ReMe summary writes can't
embed and store. Until this is resolved:

- Personal/task **reads** still work if memories were written via a
  process where embeddings did work (e.g. OpenAI as the embedding
  endpoint).
- Summary **writes** will silently fail on the embedding step.

**Fix proposals:**

1. Point `REME_EMBEDDING_API_BASE` at OpenAI `text-embedding-3-small`
   (cheap, reliable; ~\$0.02 / 1M tokens).
2. Or run a second vLLM serving BGE-M3 on `:8003`.
3. Or implement a sentence-transformers CPU fallback inside the
   adapter (drops retrieval quality but unblocks the smoke gate).

### D. Studio runs don't persist to our SQLite

`studio_bootstrap` doesn't inject the `TraceCallbackHandler`. The
graph runs but no `AgentStep` rows are written; only Studio's own
trace view captures the call graph. Fine for visual debugging,
not fine for thesis metrics.

**Fix proposal:** have `studio_bootstrap` attach
`TraceCallbackHandler(deps.recorder, run_id, SeqAllocator())` to the
runtime config it returns. One node-end emission per chain.

### E. Studio runs skip memory priming + reflection writeback

Bootstrap only allocates the run; it doesn't read primes or run the
reflector's writes through `MemoryService`. Means Studio sessions
contribute nothing to ReMe state.

**Fix proposal:** call `memory_hooks.prime_brief_messages` in
bootstrap and prepend the system message to state.messages; have
the reflector node also call `write_reflection` + `write_working_report`
when run via Studio.

### F. Researcher subgraph spawn cost

Even with `MAX_CONCURRENT_RESEARCH_UNITS=2`, two cold researcher tasks
each running 4 ReAct iterations is heavy on the 8B local model.
Latency goes up; per-step token cost goes up; structured-output
failure modes multiply.

**Fix proposal:** ship a `--profile minimal` that sets
`MAX_RESEARCHER_ITERATIONS=1` + `MAX_CONCURRENT_RESEARCH_UNITS=1`
for thesis-baseline runs where we want determinism.

---

## 7. Not Yet Implemented from Your Original Plan

### Phase 2 — Evaluation & instrumentation

- Golden task suite scaffold exists (`eval/runner.py`), but only runs
  `default` vs `none` memory profile ablations. No model-profile or
  search-on/off ablations. No latency or token-cost reporting per
  ablation cell.
- No NVML/DCGM-based GPU utilization metrics. The router has the
  `hint` parameter for these but nothing populates it.
- No queue-time / bytes-to-cloud accounting.

### Phase 3 — Hybrid routing

- The router is set up for this (`co_schedule_v0` profile routes
  researcher + final_report to cloud). But the **cloud endpoint isn't
  configured live** — sjtu vLLM serving is still on the operator's
  todo. No measurements have been taken.
- The "local minimizer/personalizer" role hasn't actually been used
  as a privacy-minimizer in any code path — it's just another endpoint
  the router can pick.

### Phase 4 — Privacy-bounded ParetoDispatch

Not implemented. `PrivacyEnvelope` exists as a schema and flows
through `Router.select`, but `Router.select` ignores it. No CI labels
are attached anywhere. No leakage proxies. No multi-objective
optimization.

**This is the actual thesis contribution** and is the next major
chunk of work.

### Phase 5 — Reflection Broadcast + hierarchical memory

Not implemented. The reflector node *emits* a `ReflectionUpdate` and
the runtime *writes* it to ReMe locally. There is no broadcast to
cloud subagents. No CI filter on broadcasts. No MI budget.
`scheduling/`, `broadcast/`, `tiering/` directories exist as empty
placeholders.

### Phase 6 — BrowseComp-Hybrid

Not started.

---

## 8. Gaps in the Original Plan vs. the Thesis Aim

These are honest critiques of where the original phased plan would
need to be sharpened to actually fulfill the 3-D Pareto frontier aim.

### 8.1 Pareto frontier sweep design is missing

A Pareto frontier across (latency, privacy-leakage, joint GPU
utilization) requires **multiple operating points**. The plan lists
the metrics but never specifies:

- The sweep dimensions (privacy budget ε in {0, 0.5, 1, 2, ∞}? Routing
  aggressiveness in {0%, 25%, 50%, 100% cloud}? Memory budget in
  {0, 4k, 16k, 64k tokens}?).
- The number of operating points per axis.
- How to handle the curse of dimensionality (3 axes × N points each =
  N³ runs against a 10–20 task suite × multiple seeds — that's a real
  compute budget).

Without a sweep design, "Pareto frontier" is rhetoric, not an artifact.

### 8.2 PD-disaggregation stream is barely engaged

DistServe, Llumnix, Mooncake, Parrot, ECO-LLM all operate **inside a
serving system** (prefill/decode disaggregation, batch coalescing,
KV-cache routing). Our setup is two separate vLLM servers with
endpoint-level routing — there's no actual disaggregation across them.

If the thesis claims to integrate this stream, the implementation
needs to either:

1. Narrow the stream to "agentic-step routing" (which is what we have)
   and reposition the contribution honestly, or
2. Pick one paper (e.g. Parrot's "semantic variable" tagging) and
   implement a slim version of it inside our pipeline, or
3. Use a unified vLLM cluster with chunked-prefill enabled and route
   to it — gives a real disaggregation surface to compare against.

Right now (1) is the de-facto position but the plan reads like (2)/(3).

### 8.3 Contextual-integrity labels need an actual classifier

The plan says "add CI labels: subject, sender, recipient, attribute
type, transmission principle, sensitivity" without saying *who
attaches them*. Real options:

- User-supplied at query time (unrealistic for 10–20 golden tasks).
- LLM classifier on each message (another model call per check —
  compounds latency, which is one of the axes you're optimizing on).
- Heuristic / regex based on entity types (PII detection only;
  doesn't capture transmission principle).
- Pre-labeled in the BrowseComp-Hybrid suite (only works if Phase 6
  is done first).

Phase 4 starts before Phase 6, so the classifier path is the default.
The latency cost of CI checks needs to be in the Pareto budget itself,
which the plan doesn't yet acknowledge.

### 8.4 Mutual-information "budget proxy" is undefined

The plan mentions a mutual-information budget but doesn't define
the proxy. Real MI estimation between (private prefs) and (cloud-visible
tokens) is hard and requires either:

- A reference distribution (which we don't have for the user's
  preferences).
- A bounded surrogate (count of sensitive-token leaks, count of
  rare-entity mentions, KL-to-public-baseline).

The thesis needs to pick one surrogate and defend it. Recommendation:
**sensitive-token exposure count + entity exposure count**, both
bounded above by a knob. Calling this "MI" is a stretch — call it
what it is: a leakage proxy.

### 8.5 Reflection Broadcast — operationally undefined

When does broadcast fire? On every reflection? Once per session? On
high-quality reflections only? How does it propagate — by
modifying cloud subagent system prompts? Pushing memories into a
shared store? Streaming via a side channel?

The plan needs a concrete protocol diagram. Otherwise the contribution
is a name, not a mechanism.

### 8.6 Hierarchical memory placement policy is unspecified

The plan lists the inputs ("recency, utility, sensitivity, retrieval
frequency, expected token savings") but not the function. Realistic
options:

- Static thresholds: "hot if accessed in last N hours AND sensitivity ≤ k".
- Learned policy: bandit or contextual-bandit trained on retrieval logs.
- LP/optimization: minimize cost subject to capacity constraints.

Pick one. Each has different evaluation requirements.

### 8.7 BrowseComp-Hybrid construction methodology is missing

"Each task includes: research question, private preference or
constraint, public web component, expected evidence requirements,
privacy budget, local/cloud routing constraints."

That's a *schema*. The *generation rule* is missing: how do you
synthesize the private preference such that it (a) doesn't trivially
leak through the research question, (b) actually changes the optimal
answer, and (c) is verifiable? Templates? Crowd-sourcing? LLM
synthesis with manual review?

Realistically, this is its own paper's worth of work; the thesis
needs to either descope to "use BrowseComp + add private preferences
manually for 10 tasks" or commit a compute budget to synthesis +
review.

### 8.8 No baseline comparison

Pareto curves are only useful relative to a baseline. The plan
doesn't specify what we're comparing against. Candidates:

- "Cloud-only" baseline (everything on the 70B; max quality, max
  privacy leakage, lowest joint utilization).
- "Local-only" baseline (everything on the 8B; min quality, min
  leakage, lowest cost — but maybe lowest joint utilization too).
- Open Deep Research vanilla (the very thing we forked) on the same
  task suite.
- Naive routers (random; round-robin; first-call-cloud).

Pick three. Without them the frontier story has no contrast.

### 8.9 Eval cost / compute budget is unspecified

Running BrowseComp-style evaluations against frontier models is
expensive. With 10–20 tasks × multiple ablations × multiple seeds ×
LLM-as-judge scoring, you're easily into the \$50–\$500/run range.
The plan doesn't budget for this.

### 8.10 The user / personalization model is shallow

The "minimizer/personalizer" gets named in the aim but never
elaborated. Realistic minimization needs at least:

- A user-preference schema (what's stored in personal memory).
- A minimizer prompt that rewrites/redacts queries before they hit
  the cloud.
- A way to score "did minimization preserve the answer's quality?"

ReMe's `personal` memory type is a vehicle; the minimization
*function* is what the thesis needs to specify.

---

## 9. Recommended Immediate Next Steps

In priority order, smallest-blocking-thing first:

1. **`scripts/studio_e2e.py` debug harness** (problem A) — unblocks
   the rest. Single afternoon of work.
2. **Patch 5–style fallbacks at supervisor + researcher** (problem B)
   — keeps the local-only pipeline stable.
3. **OpenAI embedding endpoint for ReMe** (problem C) — flip a config
   field, set `REME_EMBEDDING_API_BASE`. Five minutes.
4. **Wire `TraceCallbackHandler` into the Studio bootstrap** (problem D).
5. **Define the Pareto sweep design** (gap 8.1) before writing more
   code — committing a sweep design forces the rest of the design
   questions (gaps 8.3, 8.4, 8.8, 8.9) onto the page.
6. **Sjtu vLLM cloud endpoint** — already in the operator todo. Once
   live, run `co_schedule_v0` end-to-end and start collecting
   latency / token-cost / GPU-utilization numbers.

Phase 4 (ParetoDispatch) lands after step 5 — the sweep design tells
us what the dispatcher needs to optimize for and what counts as a
working Phase-4.

---

## 10. Files Worth Re-reading

- `/home/lyc/.claude/plans/i-made-a-dapper-quasar.md` — the original
  Phase 1.5 migration plan.
- `src/deepresearch/agents/langgraph/upstream/UPSTREAM_NOTE.md` —
  vendored-code patch list (5 patches as of today).
- `src/deepresearch/agents/langgraph/runtime.py` — the canonical
  entrypoint that exercises the full pipeline with memory + tracing.
- `src/deepresearch/agents/langgraph/studio.py` — the Studio
  variant; the source of most of the open issues.
- `src/deepresearch/models/router.py` — the Phase-4 seam.
  Signature is sacred; body is profile-only today.
- `scripts/demo_e2e.py` — hermetic e2e, the cleanest existence proof
  that the architecture works.
