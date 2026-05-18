# DeepResearch Playground — Status, Problems, and Gaps

This document inventories: (1) the original research aim and phased plan
you set out with, (2) the LangGraph migration plan I wrote in response,
(3) what actually shipped, (4) what works today, (5) what's deferred
from the original plan, and (6) honest critique of the gaps between the
plan and the thesis aim.

Date: 2026-05-12 (afternoon).

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

Six structural pieces (in order of dependency), with their landed paths:

1. `RouterChatModel(BaseChatModel)` + `RouterConfigurableModel` proxy —
   route every LLM call from any LangGraph node through `Router.select`
   into our async `ModelClient.complete()`.
   Landed in `agents/langgraph/router_chat_model.py`.
2. `TraceCallbackHandler` mapping LangGraph chain events to `AgentStep`
   rows. Landed in `agents/langgraph/callbacks.py`.
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

Plus, after the original landing: `agents/langgraph/reflection_node.py`
(our injected reflector), `agents/langgraph/memory_hooks.py`
(prime + write helpers), `agents/langgraph/state.py` (our extension
of upstream `AgentState` to declare the `reflection` field upstream's
TypedDict doesn't), and `scripts/studio_e2e.py` (live-endpoint Studio
debug harness).

Planned five-commit sequence: deps → seams → orchestrator swap →
ReMe wire-up → smoke + README.

---

## 4. What Actually Shipped (git history)

### Phase 1.5 commits (my work, in order)

| SHA | Subject | Notes |
|---|---|---|
| `ea81567` | chore(deps): pin langchain 0.3.x stack + vendor open_deep_research | 5 upstream files at pinned SHA, `UPSTREAM_NOTE.md`, license file |
| `82819bf` | feat(agents): RouterChatModel + TraceCallbackHandler for LangGraph | proxy, callback handler, role_map, ModelClient `tools=` kwarg; 15 new tests |
| `385c71c` | feat(agents): swap orchestrator to LangGraph; delete STORM agents | runtime, reflection_node, memory_hooks; orchestrator becomes a shim; 5 STORM agents + prompts/ + _jsonparse deleted |
| `eeac932` | feat(memory): wire up ReMeApp via flowllm flows (Phase 1.5) | reme_flows + reme_adapter rewrite; 12 mocked tests |
| `a189ee1` | test(smoke): bundled smoke gate + README addendum on Phase 1.5 swap | smoke_e2e_bundled.sh; README rewrite |

### Studio integration (also my work)

| SHA | Subject | Notes |
|---|---|---|
| `5adc6fc` | feat(studio): wire up LangGraph Studio frontend | `langgraph.json`, `studio.py` with bootstrap node |
| `54000a9` | fix(studio): use module-level slot for Studio active-run propagation | Studio re-enters Pregel in fresh asyncio Tasks where contextvars don't propagate; the slot is Studio-only. `runtime.run_research` keeps the contextvar path. |
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
| `dc0473a` | feat(cli): print agent-step + LLM-call tables after every run | quick post-run trace in the terminal |
| `a8dbfa6` | fix(cli): ModelCallRecord field is started_at not created_at | |

### Bug-fix sweep landed 2026-05-12 (afternoon)

Patches and code changes shipped together against the failure modes
catalogued in the prior §6 (now deleted — see §C below):

- **Patch 6** in `agents/langgraph/upstream/deep_researcher.py`: remove
  the unconditional `or True` in `supervisor_tools`' exception handler.
  Non-token-limit exceptions now propagate to `runtime.run_research`,
  which logs and persists the real error on `ResearchRun.error`.
- **Patch 7** in same file: defensive None / empty-response fallback at
  the end of both `supervisor` and `researcher` nodes. When the model
  returns None or an AIMessage with neither content nor tool_calls
  (small-model structured-output failure mode), synthesize a
  deterministic `ResearchComplete` tool call so the loops exit with
  intent (visible in the trace) instead of crashing or silently looping.
- New `agents/langgraph/state.py`: subclass of upstream `AgentState`
  that declares the `reflection: Optional[dict]` field. Without this
  the reflector's output was silently dropped by LangGraph's reducer
  (a latent bug affecting both CLI and Studio paths — runtime's
  `write_reflection` was always being called with a default empty
  `ReflectionUpdate`).
- `studio.py` rewrite: `studio_bootstrap` now resolves the memory
  profile, calls `prime_brief_messages`, builds a `TraceCallbackHandler`
  + `SeqAllocator`, stashes them on `_ActiveRun`, manually emits a
  bootstrap `AgentStep`, and best-effort injects the callback into the
  live `RunnableConfig`. A new `reflector_writer_node` replaces the
  bare reflector in the Studio graph — it runs the reflector, then
  calls `write_reflection` + `write_working_report` and finalizes the
  run row.
- `RouterChatModel._agenerate`: when the Studio active-run slot has
  `prime_msgs` set, prepend them to the messages sent to the model
  (Studio's `add_messages` reducer can't cleanly prepend system
  messages from a bootstrap node, so we inject at the LLM boundary).
- `.env.example` + `README.md`: document the `REME_EMBEDDING_API_BASE`
  / `REME_EMBEDDING_API_KEY` env vars (resolved via OpenAI
  `text-embedding-3-small`).
- `RunRequest.max_concurrent_units` + CLI `--minimal` preset: caps
  researcher iterations and concurrency to 1 for thesis-baseline runs.
- `scripts/studio_e2e.py` (new): live-endpoint Studio debug harness.
  Hermetic-fake mode for offline iteration; live mode against the
  configured vLLM.
- `UPSTREAM_NOTE.md`: documents Patches 6 + 7, with a note on the
  out-of-order patch numbering (1, 2, 3, 5, 4, 6, 7 reflect creation
  order, not file order).

### Test surface

- 36 unit + integration tests across `tests/` — all green at every
  step of the bug-fix sweep.
- One end-to-end fake-client test (`test_orchestrator_swap.py`) verifies
  local + cloud routing both fire under `co_schedule_v0`.
- `scripts/demo_e2e.py` — hermetic e2e against a fake client.
- `scripts/studio_e2e.py` — hermetic + live e2e for the Studio code
  path.
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
upstream's. The Studio graph prepends a `studio_bootstrap` node before
`clarify_with_user` and swaps the bare `reflector` for
`reflector_writer_node`, which additionally calls `write_reflection`
and `write_working_report`.

Every LLM call flows through `RouterChatModel` →
`Router.select(profile, role, envelope, hint)` → `ModelClient.complete`
→ OpenAI-compatible endpoint. The Phase-4 seam is intact.

---

## 5. Status — What Works Today

| Capability | Status | Notes |
|---|---|---|
| Hermetic demo (`scripts/demo_e2e.py`) | ✅ | Fake LLM, shows local + cloud routing + token counts + AgentStep trace + reflection persisted to working memory |
| Hermetic Studio demo (`scripts/studio_e2e.py --fake`) | ✅ | Mirrors demo_e2e but goes through `studio_graph`; bootstrap + reflector_writer steps recorded |
| 36 unit + integration tests | ✅ | `uv run pytest` |
| CLI happy path (`uv run deepresearch run "..."`) | ✅ | Works against a real vLLM; Patch 7 fallbacks keep small-model failure modes visible-and-recoverable |
| `--minimal` CLI preset | ✅ | `--minimal` sets max_searches=1 + max_concurrent_units=1 for thesis-baseline runs |
| FastAPI service | ✅ | `/research_runs` POST + GET + /trace; no UI |
| LangGraph Studio UI | ✅ | Graph loads, bootstrap primes memory + attaches callback handler; reflector_writer persists reflection + working report. Per-node chain callbacks may not propagate to all sibling nodes via Pregel's config-patching, so the AgentStep trace is bootstrap + reflector markers + per-LLM ModelCallRecord rows (which is what thesis metrics actually need) |
| Working memory (Qdrant embedded) | ✅ | Report persisted per run; embedded mode needs no Docker |
| ReMe personal/task retrieve + summary | ✅ | Adapter wired; LLM creds resolved via DeepSeek/OpenAI; embedding endpoint resolves via `REME_EMBEDDING_API_BASE` (recommended: OpenAI `text-embedding-3-small`) |
| Trace persistence (SQLite) | ✅ | Both `runtime.run_research` (full per-node chain) and Studio (bootstrap + reflector_writer markers + per-LLM ModelCallRecord rows) write to SQLite. |
| Reflection persisted into ReMe + working memory | ✅ | Both CLI and Studio paths now call `write_reflection` + `write_working_report` (latent reflection-key-dropped bug fixed via `agents/langgraph/state.py`) |
| Vendor patch documentation | ✅ | `UPSTREAM_NOTE.md` records all 7 patches incl. re-sync procedure |

---

## 6. Not Yet Implemented from Your Original Plan

### Phase 2 — Evaluation & instrumentation

- Golden task suite scaffold exists (`eval/runner.py`), but only runs
  `default` vs `none` memory profile ablations. No model-profile or
  search-on/off ablations. No latency or token-cost reporting per
  ablation cell.
- No NVML/DCGM-based GPU utilization metrics. The router has the
  `hint` parameter for these but nothing populates it.
- No queue-time / bytes-to-cloud accounting.

### Phase 3 — Hybrid routing

The Phase-4 router **seam** is set up: `Router.select(profile, role,
envelope, hint)` is wired into every LangGraph LLM call. The router
**body** ignores `envelope` and `hint` (router.py:45
`_ = envelope, hint`) and dispatches purely by `(profile, role)` — so
"set up for this" means "the signature is right and the wiring is
complete"; ParetoDispatch itself is Phase 4 work.

`co_schedule_v0` is defined in `config/config.example.yaml`
(researcher + final_report → cloud) but `config.local.yaml` still
uses legacy STORM keys (the example file is the authoritative
Phase-1.5 mapping). The **cloud endpoint isn't configured live** —
sjtu vLLM serving is still on the operator's todo. No measurements
have been taken.

The "local minimizer/personalizer" role hasn't actually been used
as a privacy-minimizer in any code path — it's just another endpoint
the router can pick.

**Phase-3.5 experiment** (decided 2026-05-12, was §6.B proposal 2):
once sjtu vLLM is live, add a model profile that routes `supervisor`
to the cloud 70B while keeping `compressor` + `reflector` local.
Compare structured-output stability and end-to-end latency against
`phase1_default` (everything-local) and `co_schedule_v0`
(researcher + final_report cloud). Config-only; no new code.

### Phase 4 — Privacy-bounded ParetoDispatch

Not implemented. `PrivacyEnvelope` exists as a schema and flows
through `Router.select`, but `Router.select` ignores it. No CI labels
are attached anywhere. No leakage proxies. No multi-objective
optimization.

**This is the actual thesis contribution** and is the next major
chunk of work, gated on the design decisions in §7.

### Phase 5 — Reflection Broadcast + hierarchical memory

Not implemented. The reflector node *emits* a `ReflectionUpdate` and
the runtime *writes* it to ReMe locally. There is no broadcast to
cloud subagents. No CI filter on broadcasts. No MI budget.
`scheduling/`, `broadcast/`, `tiering/` directories exist as empty
placeholders.

### Phase 6 — BrowseComp-Hybrid

Not started.

---

## 7. Gaps in the Original Plan vs. the Thesis Aim

These are honest critiques of where the original phased plan needs to
be sharpened to actually fulfill the 3-D Pareto frontier aim. They are
research-design questions, not engineering — every Phase-4/5/6 item in
§6 above depends on these being settled first.

### 7.1 Pareto frontier sweep design is missing

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

### 7.2 PD-disaggregation stream is barely engaged

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

### 7.3 Contextual-integrity labels need an actual classifier

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

### 7.4 Mutual-information "budget proxy" is undefined

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

### 7.5 Reflection Broadcast — operationally undefined

When does broadcast fire? On every reflection? Once per session? On
high-quality reflections only? How does it propagate — by
modifying cloud subagent system prompts? Pushing memories into a
shared store? Streaming via a side channel?

The plan needs a concrete protocol diagram. Otherwise the contribution
is a name, not a mechanism.

### 7.6 Hierarchical memory placement policy is unspecified

The plan lists the inputs ("recency, utility, sensitivity, retrieval
frequency, expected token savings") but not the function. Realistic
options:

- Static thresholds: "hot if accessed in last N hours AND sensitivity ≤ k".
- Learned policy: bandit or contextual-bandit trained on retrieval logs.
- LP/optimization: minimize cost subject to capacity constraints.

Pick one. Each has different evaluation requirements.

### 7.7 BrowseComp-Hybrid construction methodology is missing

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

### 7.8 No baseline comparison

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

### 7.9 Eval cost / compute budget is unspecified

Running BrowseComp-style evaluations against frontier models is
expensive. With 10–20 tasks × multiple ablations × multiple seeds ×
LLM-as-judge scoring, you're easily into the \$50–\$500/run range.
The plan doesn't budget for this.

### 7.10 The user / personalization model is shallow

The "minimizer/personalizer" gets named in the aim but never
elaborated. Realistic minimization needs at least:

- A user-preference schema (what's stored in personal memory).
- A minimizer prompt that rewrites/redacts queries before they hit
  the cloud.
- A way to score "did minimization preserve the answer's quality?"

ReMe's `personal` memory type is a vehicle; the minimization
*function* is what the thesis needs to specify.

---

## 8. Recommended Immediate Next Steps

The bug-fix sweep landed 2026-05-12 closed out everything in the
previous §6 ("Known Problems / Open Questions") plus a latent bug in
state propagation. The pipeline now runs end-to-end through both the
CLI and Studio paths. The forward agenda is:

### Track A — Sjtu cloud endpoint (blocks Phase 3 measurement)

1. Finish the Qwen3.6-35B-A3B-FP8 / Llama-3.3-70B-AWQ download on sjtu
   (already in operator todo).
2. `bash scripts/serve_cloud.sh` on sjtu + `bash scripts/tunnel_sjtu.sh`
   locally.
3. Smoke `co_schedule_v0` end-to-end: confirm cloud endpoint hits in
   the routing trace and the supervisor's structured outputs survive
   on the 70B.
4. Run the Phase-3.5 experiment (route `supervisor` to cloud) for the
   stability comparison.

### Track B — Phase 2 instrumentation (parallel with Track A)

1. Expand `eval/runner.py` to ablate model_profile × memory_profile ×
   search_on/off with per-cell latency + token + citation counts.
2. Add NVML/DCGM polling worker; populate `SchedulingHint` with live
   utilization samples for future Phase-4 use.
3. Add a bytes-to-cloud counter on each remote `ModelClient.complete`.

### Track C — Phase 4 design (must precede Phase 4 code)

The §7 gaps are gating questions. In priority order:

1. **§7.1** Define the Pareto sweep design — this forces commitment on
   §7.3, §7.4, §7.8, §7.9.
2. **§7.4** Pick the leakage-proxy surrogate (recommended:
   sensitive-token + entity exposure counts).
3. **§7.3** Pick the CI-label source (recommended: a small classifier
   on the minimizer endpoint, counted as a cost in the latency axis).
4. **§7.10** Specify the minimizer / personalization function.

Once §7.1 / §7.3 / §7.4 / §7.10 are settled, implement the new
`Router.select` body in `models/router.py` and the `scheduling/`
package contents.

### Track D — Phase 5 + 6 (research questions still open)

Phase 5 (Reflection Broadcast, hierarchical memory) and Phase 6
(BrowseComp-Hybrid) are downstream of Tracks A–C; their gating §7
items (§7.5, §7.6, §7.7) need design decisions, not code, as the
next move.

---

## 9. Files Worth Re-reading

- `/home/lyc/.claude/plans/i-made-a-dapper-quasar.md` — the original
  Phase 1.5 migration plan.
- `/home/lyc/.claude/plans/1-read-docs-status-md-2-luminous-toucan.md`
  — the audit / refine plan that produced the bug-fix sweep landed
  2026-05-12 (afternoon).
- `src/deepresearch/agents/langgraph/upstream/UPSTREAM_NOTE.md` —
  vendored-code patch list (7 patches as of today).
- `src/deepresearch/agents/langgraph/runtime.py` — the canonical
  entrypoint that exercises the full pipeline with memory + tracing.
- `src/deepresearch/agents/langgraph/studio.py` — the Studio variant;
  studio_bootstrap + reflector_writer_node mirror runtime's effects.
- `src/deepresearch/agents/langgraph/state.py` — our `AgentState`
  extension that declares the `reflection` key.
- `src/deepresearch/agents/langgraph/callbacks.py` — the
  `TraceCallbackHandler` + `SeqAllocator`.
- `src/deepresearch/agents/langgraph/reflection_node.py` — our
  injected reflector that emits `ReflectionUpdate`.
- `src/deepresearch/agents/langgraph/router_chat_model.py` — the
  `RouterChatModel` + `RouterConfigurableModel` + active-run slot.
- `src/deepresearch/models/router.py` — the Phase-4 seam.
  Signature is sacred; body is profile-only today.
- `scripts/demo_e2e.py` — hermetic e2e against a fake client.
- `scripts/studio_e2e.py` — hermetic + live e2e for the Studio path.
