# Root-Cause Confirmation & Fix Design — Gemini Orchestrator Over-Management

**Branch:** `gemini-port` · **Date:** 2026-06-23 · **Status:** PLAN-FIRST (no code changed; awaiting approval)

**Verdict:** Root cause **confirmed from source** (adversarially verified). The instability is harness-induced, as hypothesized — with three corrections to the original framing that *strengthen* the case and one that *changes the fix*. The naive version of candidate (i) has concrete holes that re-introduce O15; the design below is the hardened version.

---

## Phase 1.1 — Root cause: CONFIRMED

The leading hypothesis is correct. Mechanism, traced end-to-end in source:

1. **Auto-arm.** First `spawn_agent` on a GeminiNative root orchestrator with no existing goal arms an **Active** thread goal (objective = user's task, `token_budget = 250_000`). `multi_agents_v2/spawn.rs:232` (gate `is_gemini_native`) → `:258-296` `maybe_auto_arm_gemini_orchestration_goal` → `:25,280`. This is the *only* automatic `create_thread_goal` callsite.

2. **Re-engagement is unconditional on `status==Active`.** Every turn-end raises `MaybeContinueIfIdle` (`tasks/mod.rs:797-802`) → `maybe_continue_goal_if_idle_runtime` (`goals.rs:1243`) → `maybe_start_goal_continuation_turn` (`:1248`) → `goal_continuation_candidate_if_active` (`:1333`). That candidate gates **only** on: Goals on, not plan-mode, no active turn, no `trigger_turn` mailbox items, and `goal.status==Active` (`:1336-1388`). **It has zero awareness of child state and zero awareness of whether a final answer was already produced** — grep for child/pending/spawn/synthes/last_agent in that path returns nothing. So it fires **again after the parent has synthesized a final answer with all children terminal** (`fires_after_synthesis = true`, verified).

3. **The self-loop.** A continuation turn ends → `on_task_finished` re-raises `MaybeContinueIfIdle` → re-arms another continuation. It re-fires every idle turn until the goal leaves Active.

4. **The injected prompt is a SOLO-WORKER prompt** — the smoking gun. `goals.rs:1393` builds the continuation items from `continuation_prompt(&goal)` → `prompts/templates/goals/continuation.md`. That template is written for *one agent doing the work itself* and contains exactly the directives that produce the observed pathology:
   - `:20` *"inspect the current state before relying on it. Improve, replace, or remove existing work"* → **re-investigation of delegated domains.**
   - `:23` *"If update_plan is available… use it to show a concise plan… Keep the plan current"* → **the 83 "Updated Plan" re-plans.**
   - `:30-41` a stringent multi-paragraph **Completion audit** that treats completion as *"unproven,"* demands *"gather stronger evidence or continue the work"* → **re-verification / the 36 web searches.**
   - `:51` *"Do not call update_goal unless the goal is complete or the strict blocked audit… is satisfied"* → **removes the easy exit.**

5. **Loop guards = exactly the two named, both weak:** model calls `update_goal complete` (`goal/update_goal.rs:53-79`), or the 250K budget flips Active→BudgetLimited (`state/runtime/goals.rs:434-452`).

**On the empirical count:** confirmed *structurally* rather than by a live instrumented dogfood. The candidate function provably has **no synthesis guard and no child-state guard**, so it *must* fire post-synthesis whenever the session is idle with the goal Active — not a sampled tendency, an unconditional code path. A live "N continuation turns after first final answer" number can still be obtained via temporary `tracing` at `goals.rs:1248`/`1393` and `tasks/mod.rs:797`, but the structural proof already answers confirm-or-refute.

---

## Phase 1.2 — Reference vs ours: structural vs model-driven termination

| | **References (upstream / non-Gemini)** | **Ours (GeminiNative root)** |
|---|---|---|
| Orchestrator "done" trigger | **Structural** — turn ends with no more tool calls → thread idle → done | **Model-driven** — parent must call `update_goal complete` (or 250K trips) |
| Backstop after idle | None — `MaybeContinueIfIdle` is a **no-op** (no goal exists) | Goal-continuation re-engages every idle turn while Active |
| Why | Auto-arm is Gemini-gated; no `create_thread_goal` on non-Gemini | Auto-arm installs an Active goal on first delegation |

**Source-verified (this repo):** non-Gemini delegation auto-arms **no** goal (`spawn.rs:232` gate; test `multi_agents_tests.rs:807-825` asserts `get_thread_goal()==None` after a Bedrock spawn). With no Active goal, `goal_continuation_candidate_if_active` returns None (`goals.rs:1379-1382`) and the orchestrator terminates at turn-end — structurally identical to upstream. **Exactly where ours depends on flash remembering:** `goal/update_goal.rs:53-79` is the *only* model-driven exit, and `continuation.md:30-51` actively discourages taking it.

**Antigravity / Claude Code:** **INFERENCE, not sourced** — both believed to terminate structurally at the turn boundary (subagent results return synchronously as the parent's tool result; no persistent goal a model must close). **The design does not ride on these** — it rides on the source-verified upstream/non-Gemini structural model, which is the one being matched.

> Note: the goals-continuation machinery itself is *not* Gemini-gated. A manually-created Active goal on any wire API would loop the same way. The non-Gemini "stability" is purely "no auto-goal is armed," not a Gemini-specific suppression.

---

## Phase 1.3 — Other harness-induced over-management drivers

**(a) Orchestrator tool surface — unbounded, confirmed.** The root Gemini orchestrator keeps its **full solo toolset** after delegating: client-side **Tavily `web_search` + `web_fetch`** (`spec_plan.rs:558-587`, `add_client_web_tools`), plus shell/exec, `apply_patch`, `update_plan`, and `spawn_agent`. The web tools gate **only** on `wire_api==GeminiNative` + search-mode + tool-mode — **no role / orchestrator / depth gating whatsoever**. So 36 parent web searches is fully possible and nothing bounds the count (`web_search` even allows `limit` up to 20). The tools' own descriptions push *"You must fetch a page before claiming to have read or verified it"* (`client_web.rs:419-425`), compounding the continuation prompt's "gather stronger evidence."

**(b) The hint gap — real.** `GEMINI_MULTI_AGENT_V2_USAGE_HINT` (`spec_plan.rs:104`, default-on) tells the parent to *wait for all children* and *"integrate every child's result into one synthesized answer."* It has **no STOP clause** ("once all children report, stop — don't start new work") and **no anti-redo clause** ("treat child reports as authoritative; do not re-investigate/re-search delegated domains"). Half of it is actually addressed to the *child* role, diluting orchestrator guidance.

**(c) Child-result presentation invites redo.** Results reach the parent in **two** channels, both carrying the raw full report body with **no "authoritative, do-not-re-derive" framing**: `wait_agent`'s `gemini_full_delivery` agent_statuses (`wait.rs:300-309`) and a `trigger_turn=true` `<subagent_notification>` JSON blob (`session/mod.rs:1771-1827`, `subagent_notification.rs:55-63`).

**(d) The dominant driver is structural, not the hint:** the auto-armed Active goal + the solo `continuation.md` prompt. The hint/tool-surface gaps are *amplifiers* that give the re-engaged parent both the means and the lack of a stop-signal.

---

## Contradictions with the original framing (flagged)

1. **The 250K budget may never even trip.** `goal.tokens_used` counts **non-cached-input + output only** (`goals.rs:1550-1554`), excluding the cached input that dominates a large multi-agent context — while the observed **289–296K are raw session totals**. So in the worst runs the goal budget reads <250K and *never flips*, leaving **model self-completion as the only real exit.** This makes a structural fix *more* necessary, not less. (Also: BudgetLimited is a *soft* turn-boundary steer, not a hard interrupt — a wrap-up turn is billed on top, explaining the overshoot past 250K.)

2. **Empty completions are now fully suppressed on Gemini**, not merely non-waking. The trigger_turn-only-on-Gemini note (commit b09c46202) is right, but the *current* branch additionally **early-returns `Completed(None)` before any mailbox enqueue** (`session/mod.rs:1784-1788`, commit ecb970c87). So the **empty/suppressed *last* child completion** is the **unique O15 gap that only the goal backstop can cover** — the pending-work wake structurally cannot fire for it. Load-bearing for the fix.

3. **Web tools come from the *client-side* path, not hosted `create_web_search_tool`.** The 36 searches are client-side Tavily via `add_client_web_tools`, not the hosted `ToolSpec::WebSearch`.

4. **The 83 plans / 36 searches are model-behavior, not derivable from source.** Code proves the parent *can* and *is re-prompted to* re-plan/re-search unboundedly; the magnitude is flash's response to that pressure — the harness-structure-times-model instability.

---

## Phase 2 — Fix design (structural termination, hardened)

**Principle:** push the control decision into the system. Make orchestrator termination **structural** — stop depending on flash remembering `update_goal complete`. Match the reference model: *all delegated work terminal + a consolidated answer emitted ⇒ done.*

### Layer 1 (primary — the load-bearing fix): hardened structural auto-complete

**Hook:** `tasks/mod.rs:797`, in `on_task_finished` immediately **before** `MaybeContinueIfIdle`. At that point the harness has the just-cleared idle state, `last_agent_message` in scope, and `self.services.agent_control` + `self.thread_id` to enumerate children. Complete the auto-armed goal (same `set_thread_goal(status=Complete)` path `update_goal` uses) **iff ALL of:**

1. **Gemini-native + root-orchestrator source** (reuse `is_gemini_native` + `is_root_orchestrator_source` from `spawn.rs`).
2. **The goal is the tracked auto-armed goal** and still Active — *not* a user-created goal. → **New requirement the probe's hook proposal omitted:** record `auto_armed_orchestration_goal_id` in `goal_runtime` state at arm time (mirrors `budget_limit_reported_goal_id`; no schema change). A user can create a goal *before* the first spawn (auto-arm then skips, `spawn.rs:263-264`); without this marker we'd wrongly auto-complete *their* goal.
3. **This turn emitted a final-channel answer** (`last_agent_message.is_some()`).
4. **This turn ran none of `{spawn_agent, followup_task, send_message}`** (the re-engagement tools). `wait_agent` *is* allowed — it's how the parent collects, and the synthesis turn legitimately calls it.
5. **No `trigger_turn` mailbox items pending** (no uncollected non-empty completion).
6. **Every child is conservatively terminal:** `Completed | Errored | Shutdown` only. **Treat `NotFound`, `PendingInit`, `Running`, `Interrupted` as not-done** (stricter than the existing `is_final`, which counts `NotFound` terminal). Enumerate via `open_thread_spawn_children(self.thread_id)` + `get_status` — the exact pattern `wait_agent` already uses (`wait.rs:171-232`).

Completing here deterministically suppresses the very next continuation (`goals.rs:1379` short-circuits on `status!=Active`) and emits the GOAL_COMPLETED metric — closing the loop the moment the orchestrator is genuinely done, without waiting on the model or the (possibly-inert) budget.

### Layer 2 (makes termination *happen sooner* + kills re-search): Gemini-orchestration continuation prompt

When a continuation *legitimately* fires (children still pending, pre-synthesis), it currently injects the solo `continuation.md`. Branch at `goals.rs:1393`: for the tracked auto-armed goal, inject an **orchestration-specific** prompt instead — *"You are an orchestrator. Treat each child's report as authoritative for the part you delegated; do NOT re-investigate, re-search, or re-run delegated work. Collect any newly-reported results and integrate them. If all delegated children have reported and you've delivered the consolidated answer, call `update_goal complete`. Do not re-plan or web-search delegated domains."* This counters the re-plan/re-verify drivers **and inverts the completion-audit pressure**, so the parent reaches a clean Layer-1 synthesis turn fast instead of grinding toward 250K. Gemini-gated; byte-identical elsewhere.

### Layer 3 (cheap reinforcement): STOP + anti-redo clauses in the usage hint

Append to `GEMINI_MULTI_AGENT_V2_USAGE_HINT` (`spec_plan.rs:104`, already Gemini-only): *"Once every spawned child has reported, integrate and STOP — deliver the consolidated answer and start no new work. Treat each child's report as authoritative for the part you delegated; do not re-investigate or independently re-verify a domain you handed to a child."* Additive, low-risk.

### Layer 4 (optional / future): bound the orchestrator's post-delegation tool surface

Drop `web_search`/`web_fetch` on continuation turns of an auto-armed orchestration goal. Heaviest and riskiest (could block legitimate orchestrator needs) — **not** recommended for the first cut; Layers 2–3 discourage re-search without removing capability.

**Recommendation:** ship **Layers 1 + 2 + 3** together. Layer 1 is the structural stop; Layer 2 ensures the parent *reaches* the stop quickly and stops re-searching; Layer 3 is belt-and-suspenders.

---

## Anti-stall (O15) preservation — explicit argument

The stall state is: *parent idle, no user input, and either a child non-terminal OR a child completion uncollected.* In every such state the design keeps the backstop alive:

- **Child still running / not yet synthesized** → condition 6 (or 3) fails → goal stays **Active** → goal-continuation **still fires**, re-engaging the parent exactly as today.
- **The unique O15 gap (empty/suppressed *last* completion):** while that child was running, condition 6 was false, so we never completed; the goal stayed Active and continuation kept re-waking the parent — so the parent re-engages, observes the child terminal (via `wait_agent`/`list_agents`), and synthesizes. Auto-complete only fires *after* that clean synthesis turn.
- **We only ever complete on a pure synthesis turn** (conditions 3–6): a final answer was emitted, no child was re-engaged this turn, nothing is pending, all children are conservatively done. That is precisely the "genuinely done" state where continuing is pure over-management — and the state where the references terminate too.

### Adversarial holes (verifier) in the naive (i), and how the hardening closes them

- **Hole A — followup re-open race (would break O15):** parent emits a summary *and* `followup_task`s an empty child in the same turn; followup is in-flight, child not yet Running. Naive (i) completes → child later re-completes *empty* (suppressed) → no wake → permanent stall. **Closed by condition 4** (followup ran this turn → no auto-complete).
- **Hole B — interim message false positive:** `last_agent_message.is_some()` ≠ "final report." **Closed by condition 6** — an interim "waiting for results" message only coincides with *all children terminal* if they actually are done.
- **Hole C — transient `NotFound`/fast-error:** **Closed by condition 6's conservative terminal set** (NotFound ⇒ not-done) + condition 4 (no spawn this turn → children registered in a prior turn).
- **Hole E — spawn+synthesize in one turn:** **Closed by condition 4** (no `spawn_agent` this turn).
- **Hole D — "uncollected results" isn't representable:** correct — that's why the design uses **variant (i) auto-complete with the conservative predicate**, not variant (ii) keyed on a non-existent "uncollected" signal. (`wait_delivery_snapshots` only updates on `wait_agent` calls; it can't distinguish synthesized from ignored.)

---

## Invariant checklist

- **O15 (15-min stall):** preserved — argument above; continuation fires whenever any child is non-terminal or pre-synthesis.
- **O11 (wait floor):** untouched — `wait_agent` runs inside an *active* turn; the turn-end hook cannot fire mid-wait (verifier-confirmed).
- **O23 / O23b / O25d (null/empty-completion cluster):** untouched — suppression path (`session/mod.rs:1784-1788`) unchanged; condition 4 guarantees we never complete on a turn that re-engaged a child, so the empty-recompletion-after-complete stall (Hole A) cannot occur.
- **Gemini-gated / non-Gemini byte-identical:** all new logic behind `is_gemini_native` + `is_root_orchestrator_source` + the tracked auto-armed-goal-id; no non-Gemini code path changes.

---

## Open items before implementation

1. **Per-turn tool-call signal for condition 4** ("did this turn run spawn/followup/send?") — needs a small confirmation of where turn-scoped tool usage is recorded (turn metadata vs. a session flag set in those handlers). The only piece that may need minor new plumbing; everything else reuses existing APIs.
2. **Optional live empirical probe** — if a measured continuation-turn-after-synthesis count is wanted, add temporary tracing and run the 4-way dogfood. (Structural proof already confirms it; this is just for a number.)
3. **New tests to add:** completes on clean synthesis turn; does NOT complete while a child Running; does NOT complete on a followup turn (Hole A); does NOT complete on interim message with children running; does NOT complete a user-created goal. Existing tests `multi_agents_tests.rs:807-825/874-911/913-954` stay valid.

---

## Key source references

| Concern | Location |
|---|---|
| Auto-arm goal (Gemini-gated) | `core/src/tools/handlers/multi_agents_v2/spawn.rs:232,258-296,25,280,298-312` |
| Turn-end → MaybeContinueIfIdle | `core/src/tasks/mod.rs:782-803,797-802` |
| Continuation candidate (no child/synthesis guard) | `core/src/goals.rs:1243-1246,1248-1331,1333-1395,1379-1382` |
| Continuation prompt build | `core/src/goals.rs:1393` → `prompts/templates/goals/continuation.md:20,23,30-41,51` |
| Pending-work wake (trigger_turn) | `core/src/tasks/mod.rs:466-487`; `core/src/session/input_queue.rs:93-99` |
| Child→parent completion + empty suppression | `core/src/session/mod.rs:1771-1827,1784-1788,1811` |
| Budget accounting / BudgetLimited flip | `core/src/goals.rs:936-1054,1550-1554`; `state/src/runtime/goals.rs:428-452` |
| Model exit (update_goal complete) | `core/src/tools/handlers/goal/update_goal.rs:53-85`; `goal.rs:89` |
| Child enumeration / status | `core/src/agent/control.rs:1231-1239,869-878`; `core/src/agent/status.rs:23-28`; `protocol/src/protocol.rs:1589-1605` |
| Existing wait precedent | `core/src/tools/handlers/multi_agents_v2/wait.rs:171-232,241-248,266,300-309` |
| Orchestrator web tools (no role gating) | `core/src/tools/spec_plan.rs:558-587`; `core/src/tools/handlers/client_web.rs:34,193-200,419-425` |
| Usage hint | `core/src/tools/spec_plan.rs:104,731-744,754`; `core/src/config/mod.rs:1067` |
| Non-Gemini no-auto-arm test | `core/src/tools/handlers/multi_agents_tests.rs:807-825` |
