# Codex CLI → Gemini Port Specification

**Goal:** Fork OpenAI Codex CLI (Rust core in `codex-rs/`) so the agent runs on **Google Gemini** as the brain while keeping Codex's harness intact: agent loop, subagents, sandbox, `apply_patch`, MCP, plugins, skills, hooks, file-search, TUI.

**Audience:** Coding agents (Claude Code / Codex / Antigravity) and the author. Every fix below is field-level and references the exact crate/file. Verify all API behavior against the linked Google docs before implementing — Gemini's API surface moves fast.

**Scope decision (locked):** Voice / `realtime-webrtc` is **out of scope**. It is independent of the coding core and would require a separate port to Gemini Live API. Disable it for this build.

---

## 0. Source-of-truth references

Cite and re-verify these while implementing:

- Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Thought signatures (critical): https://ai.google.dev/gemini-api/docs/thought-signatures
- Gemini 3 developer guide: https://ai.google.dev/gemini-api/docs/gemini-3
- Thinking / reasoning: https://ai.google.dev/gemini-api/docs/thinking
- Function calling + modes (AUTO/ANY/NONE): https://ai.google.dev/gemini-api/docs/function-calling
- What's new in Gemini 3.5 Flash: https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5
- Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Vertex AI auth / ADC: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys
- Vertex OpenAI-compat endpoint: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/openai
- Gemini 3.1 Pro (Vertex): https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro
- Reference bug — Codex multi-turn tool calls fail on Gemini (thought_signature): https://github.com/openai/codex/issues/7519
- Reference logic — gemini-cli 429 fallback handling: https://github.com/google-gemini/gemini-cli/issues/9248
- Reference impl — gemini-cli OpenAI-compat provider (SSE accumulation, factory detection): https://github.com/google-gemini/gemini-cli/issues/23385
- LiteLLM Gemini 3 thought-signature preservation: https://docs.litellm.ai/blog/gemini_3

---

## 1. Architecture decision: native `generateContent` vs OpenAI-compat

This is the most consequential choice. Pick **native `generateContent`** for the agentic path.

| | OpenAI-compat (`/v1beta/openai/`) | Native `generateContent` |
|---|---|---|
| Translation effort | Low (chat-completions shaped) | Higher (Gemini-native parts/contents) |
| `thought_signature` fidelity | **Often stripped by the compat layer** → multi-turn tool calls 400 | Full, returned exactly as needed |
| Tool schema | OpenAI `tools[]` | `functionDeclarations` |
| Verdict | OK for simple chat | **Required for agentic loop / subagents** |

**Rationale:** Codex's value is multi-step tool calling. Gemini is stateless and requires `thought_signature` round-trip on every function-call turn (see §P1). The OpenAI-compat layer frequently drops the `extra_content.google.thought_signature` field, which is exactly the failure in openai/codex#7519. The native API never loses it.

**Implementation:** Build the request/response translators against native `generateContent` first. Keep an OpenAI-compat fallback behind a flag for debugging only.

- Native endpoint (AI Studio): `https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent`
- Native endpoint (Vertex): `https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{model}:streamGenerateContent`

---

## 2. Target models & thinking

GA model IDs (verified May 2026):

- **`gemini-3.5-flash`** — primary. Strongest current model for agentic/coding; stable GA ID (no preview suffix). Leads `gemini-3.1-pro-preview` on Terminal-Bench 2.1, MCP Atlas, Finance Agent v2; trails it on Humanity's Last Exam, ARC-AGI-2, 128K long-context. Inputs: text/image/audio/video/PDF, 1M context, 64K output cap.
- **`gemini-3.1-pro-preview`** — deeper reasoning fallback. Has an additional `MEDIUM` thinking level. There is also a `gemini-3.1-pro-preview-customtools` endpoint variant (see §P6).

**Thinking levels:** `minimal`, `low`, `medium` (default), `high`.
- The default dropped from `high` → `medium` in the 3.5 generation. **Always set the level explicitly** to avoid a silent quality regression.
- `thinking_level` and the legacy `thinking_budget` are mutually exclusive — sending both returns 400.
- Reasoning **cannot be disabled** on Gemini 3.x models (no "off"); minimum is `low`/`minimal`.
- Via OpenAI-compat, OpenAI's `reasoning_effort` is auto-mapped to `thinking_level`. On the native path, set `generationConfig.thinkingConfig.thinkingLevel`.

Docs: https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5 , https://ai.google.dev/gemini-api/docs/thinking

---

## 3. Authentication

Two modes. Add both to the `AuthMode` enum (`login/src/auth/manager.rs`; existing variants: `ApiKey`, `Chatgpt`, `ChatgptAuthTokens`, `AgentIdentity`).

### 3a. AI Studio (`AuthMode::GeminiApiKey`)
- Reuse the existing `ApiKeyAuth` struct.
- Header: `Authorization: Bearer <GEMINI_API_KEY>` (native API also accepts `x-goog-api-key`).
- Simplest path; good for local dev.

### 3b. Vertex AI (`AuthMode::GeminiVertexAdc`) — production
- **Vertex rejects API keys** ("API keys are not supported by this API. Expected OAuth2 access token"). Must use ADC.
- Env: `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.
- ADC source: `gcloud auth application-default login` (user) or `GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json` (service account, "Vertex AI User" role).
- Implement an **ADC token provider**: fetch OAuth2 access token, cache it, refresh ~5 min before its ~1h expiry. In Rust use `gcp_auth` or `google-cloud-auth`, or shell out to `gcloud auth application-default print-access-token`.
- Header: `Authorization: Bearer <ADC access token>`.

### 3c. 401 refresh path
`core/src/client.rs:~1914` currently refreshes ChatGPT tokens once on 401. Add a branch: when auth mode is `GeminiVertexAdc`, refresh the **ADC token** (not a ChatGPT `refresh_token`). Skip PKCE/device-code (`login/src/pkce.rs`, `device_code_auth.rs`) for Gemini.

Docs: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys , https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/openai

---

## 4. Coupling map: keep vs change

**Model-agnostic — inherited for free (do NOT touch):**
- Subagents: `agent-graph-store` ("storage-neutral parent/child topology"), `core/src/codex_delegate.rs`, `core/src/thread_manager.rs`.
- Tool definitions: `tools/src/tool_definition.rs`, `json_schema.rs`, `tool_spec.rs`.
- Sandbox: `bwrap`, `linux-sandbox`, `windows-sandbox-rs`. Exec: `exec`, `execpolicy`.
- MCP: `rmcp-client`, `mcp-server`. Plus `plugin`, `skills`, `hooks`, `file-search`, `tui`.
- Thread store is **local** (`LocalThreadStore`/`InMemoryThreadStore`) → history is not lost.

**OpenAI/Responses-coupled — must change (the 24 problems below):**
1. `model-provider-info/src/lib.rs` — `WireApi` enum.
2. `core/src/client.rs` — request build + SSE parse against `/responses`.
3. `tools/src/responses_api.rs` + `tools/src/tool_spec.rs` — tool serialization, `FreeformTool`, `ToolSpec::WebSearch`.
4. `protocol/src/models.rs` — `ResponseItem`, `ContentItem`, reasoning `encrypted_content`.
5. Hosted endpoints: `/responses/compact`, `/v1/memories/trace_summarize`, `/backend-api/files`.
6. Auth/login, headers, usage, rate-limit, model config.

---

## 5. Crate structure: `gemini-adapter`

Add one new crate. OpenAI path stays untouched (factory detection, per gemini-cli#23385: branch on env vars).

```
codex-rs/gemini-adapter/
  src/
    lib.rs                 # GeminiProvider, factory detection
    auth.rs                # AuthProvider: VertexAdc { token cache+refresh } | AiStudio { api_key }
    request_translator.rs  # Codex Prompt (Responses-shaped) -> Gemini GenerateContentRequest
    response_translator.rs # Gemini SSE -> Codex ResponseItem stream (incl. thought_signature, usage)
    tool_translator.rs     # ToolSpec -> functionDeclarations + JSON-schema sanitizer
    signature_store.rs     # per-tool-call thought_signature capture & round-trip
    error.rs               # Google RPC error (RetryInfo/QuotaFailure) -> Codex retry decision
    compaction.rs          # client-side history compaction (replaces /responses/compact)
    model_config.rs        # ModelInfo entries for gemini-3.5-flash / gemini-3.1-pro-preview
```

Wire it in at `core/src/client.rs` where the turn stream is built: if provider is Gemini, route through `gemini-adapter` instead of the `/responses` builder.

---

## 6. The 24 problems and detailed fixes

Severity: 🔴🔴 = breaks agentic loop · 🔴 = breaks a major feature · 🟡 = required, contained · 🟢 = mechanical · 🔵 = deferred/behavioral.

### 🔴🔴 P1. thought_signature round-trip (the single most important fix)
**Symptom:** second or later tool call returns `400 INVALID_ARGUMENT: ... is missing a thought_signature` (openai/codex#7519).
**Root cause:** Gemini is stateless and requires the encrypted `thought_signature` from each function-call part to be returned on subsequent turns. Codex has `Reasoning { encrypted_content }` (Responses format) at `protocol/src/models.rs` but no Gemini-signature concept.
**Fix:**
- In `response_translator.rs`, on every function-call part, capture the signature. Native: `candidates[].content.parts[].thoughtSignature`. Compat: `extra_content.google.thought_signature` / `provider_specific_fields.thought_signature`.
- Store it keyed by tool-call id in `signature_store.rs`; attach it to the corresponding `ResponseItem::FunctionCall` (reuse/extend a field so it survives in history).
- In `request_translator.rs`, when serializing history back, re-attach each signature to its function-call part **exactly as received**.
- Rules: required for function-calling parts even at `minimal` thinking on 3.x Flash; must NOT be cleared within the active turn; from Gemini 3.5 Flash, reasoning context from all prior turns is used when signatures are present.
- Streaming caveat: the signature may arrive in a part with empty text content — parse the whole stream up to `finish_reason` (§P13).
Docs: https://ai.google.dev/gemini-api/docs/thought-signatures , https://docs.litellm.ai/blog/gemini_3

### 🔴🔴 P2. Compat layer strips signatures
**Fix:** use native `generateContent` (see §1). If a compat gateway must be used and strips signatures, last-resort dummy-signature injection exists but degrades quality — do not rely on it.

### 🔴 P3. Server-side compaction lost (`/responses/compact`)
**Root cause:** `core/src/client.rs` `compact_conversation_history` calls OpenAI's hosted compaction; Gemini has no equivalent.
**Fix:** implement client-side compaction in `compaction.rs`. Reuse existing helpers in `core/src/compact_remote.rs`: `process_compacted_history`, `trim_function_call_history_to_fit_context_window`, `should_keep_compacted_history_item`. Replace the remote call with a local summarization request to Gemini (a normal `generateContent` call returning a condensed transcript). Trigger at `auto_compact_token_limit` (§P19).

### 🔴 P4. web_search / web_fetch were OpenAI hosted tools
**Root cause:** `ToolSpec::WebSearch` (`tools/src/tool_spec.rs`) and `ResponseItem::WebSearchCall` (`protocol/src/models.rs`) are OpenAI server-side tools. They vanish on Gemini.
**Fix:** remove `ToolSpec::WebSearch`. Add two **client-side** `ToolSpec::Function` tools, executed by Codex in the agent loop:
- `web_search(query)` → call a search backend (Brave / Tavily / SearXNG) via `reqwest`.
- `web_fetch(url)` → fetch + readability-extract via `reqwest`.
Prefer this over Gemini's Google Search grounding: grounding over OpenAI-compat is limited and non-deterministic; client-side gives full control and is model-agnostic.

### 🟡 P5. Hosted memory summarization lost (`/v1/memories/trace_summarize`)
**Fix:** disable, or reimplement as a local `generateContent` summarization. Thread history itself is local and unaffected.

### 🟢 P6. apply_patch is a freeform/custom-grammar tool
**Root cause:** `create_apply_patch_freeform_tool` → `ToolSpec::Freeform(FreeformTool{ format: { syntax, definition } })`. Chat-style function calling has no custom grammar.
**Fix (preferred):** drop the `Freeform` variant for Gemini; let the model invoke `apply_patch` **via shell/exec** — `core/src/tools/handlers/shell.rs` already has `intercept_apply_patch`, and `StreamingPatchParser` handles parsing. No custom grammar, no special endpoint.
**Alternatives:** (b) `gemini-3.1-pro-preview-customtools` endpoint preserves custom tools; (c) wrap as a JSON `Function` with a `{patch: string}` param (risk: JSON escaping corruption).

### 🟡 P7. Tool JSON-schema rejected by Gemini
**Symptom:** `Unknown name "type" at 'tools[0].function': Cannot find field`; ANY mode rejects large/deeply-nested schemas.
**Fix:** in `tool_translator.rs` add a sanitizer pass over every `JsonSchema`: strip unsupported keys (`additionalProperties`, complex `$ref`, exotic `format`), flatten deep nesting, shorten property names. Gemini supports only a subset of OpenAPI schema.
Docs: https://ai.google.dev/gemini-api/docs/function-calling

### 🟢 P8. `parallel_tool_calls` param rejected
**Root cause:** `core/src/client.rs:~779` sends `parallel_tool_calls`; Gemini rejects the param even though it supports parallel calls natively.
**Fix:** strip the param for Gemini. Parallel still happens; **match function results to calls by id, not by position** (model may return calls in any order).

### 🟡 P9. `instructions` is a top-level Responses field
**Root cause:** `core/src/client.rs:~775` sends `base_instructions.text` as top-level `instructions`.
**Fix:** map to Gemini `systemInstruction` (native) / system message (compat). Also **trim it**: Gemini 3 prefers concise, direct instructions and over-analyzes verbose prompt-engineering written for older models.
Docs: https://ai.google.dev/gemini-api/docs/gemini-3

### 🟢 P10. Low temperature breaks Gemini 3
**Fix:** override `temperature = 1.0` for Gemini 3.x in `request_translator.rs`. Lower temperatures cause degraded/looping behavior.

### 🟢 P11. `reasoning_effort` vs `thinking_level`/`thinking_budget` conflict
**Fix:** send exactly one. On compat, send only `reasoning_effort`. On native, send only `thinkingConfig.thinkingLevel`. Never combine with `thinking_budget`.

### 🔵 P12. Plan-mode may not stop to ask
**Root cause:** plan mode relies on the model proactively calling `request_user_input` (`core/src/codex_delegate.rs` — `RequestUserInputEvent`). Gemini AUTO mode tends to "fill the function" rather than pause.
**Fix (behavioral, needs eval):** strengthen the system instruction for the plan phase ("always call `request_user_input` to confirm scope before editing files"); or use Gemini `ANY` mode with `allowed_function_names` restricted to `{request_user_input, propose_plan}` during the plan phase. **Do not** combine forced-tool (ANY) with structured output → 400 (§P18).

### 🟡 P13. SSE: signature/finish arrive in unpredictable chunks
**Fix:** in `response_translator.rs`, accumulate `tool_calls` deltas across chunks and parse the entire stream until `finish_reason`. The signature can appear in an empty-text part. Reference SSE accumulation: gemini-cli#23385.

### 🔴 P14. Rate-limit / retry format is completely different
**Root cause:** Codex parses OpenAI-style (`Retry-After` header + `{type, code}`). Gemini returns `429 RESOURCE_EXHAUSTED` with Google RPC details in the body.
**Fix:** in `error.rs`, parse `error.details[]`:
- `google.rpc.RetryInfo.retryDelay` (e.g. `"52s"`) → backoff duration.
- `google.rpc.QuotaFailure.quotaId` → classify the limit.
Decision logic (per gemini-cli#9248):
- `retryDelay` < 5 min OR `quotaId` contains "PerMinute" → retry silently (transient).
- `retryDelay` > 5 min OR `quotaId` contains "PerDay" → long lockout → fallback model (Flash↔Pro).
- token/context errors → do NOT retry.
Docs: https://github.com/google-gemini/gemini-cli/issues/9248 , https://ai.google.dev/gemini-api/docs/rate-limits

### 🟡 P15. Token-usage field mapping
**Root cause:** `protocol/src/protocol.rs:1919` `TokenUsage { input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens, total_tokens }`.
**Fix:** map in `response_translator.rs`:

| `TokenUsage` | Gemini compat `usage` | Gemini native `usageMetadata` |
|---|---|---|
| `input_tokens` | `prompt_tokens` | `promptTokenCount` |
| `output_tokens` | `completion_tokens` | `candidatesTokenCount` |
| `reasoning_output_tokens` | `completion_tokens_details.reasoning_tokens` | `thoughtsTokenCount` |
| `cached_input_tokens` | `prompt_tokens_details.cached_tokens` | `cachedContentTokenCount` |
| `total_tokens` | `total_tokens` | `totalTokenCount` |

Note: `thoughtsTokenCount` is **billed** — map it correctly.

### 🟡 P16. Prompt caching
**Root cause:** Codex sends OpenAI `prompt_cache_key` (`core/src/client.rs:222`).
**Fix:** strip it for Gemini. Rely on Gemini **implicit caching** (≈90% discount on cached input, automatic). Keep the request prefix stable across turns (`systemInstruction` + tool declarations + old history first, no timestamps/random at the front) so the cache hits. Optional: create an explicit `cachedContent` resource for the large base-instructions block.

### 🟢 P17. `x-codex-*` headers
**Root cause:** `core/src/client.rs:135+` sends `x-codex-installation-id`, `x-codex-turn-state` (sticky routing), `x-codex-turn-metadata`, `x-codex-parent-thread-id`, `x-codex-window-id`, `x-codex-beta-features`.
**Fix:** strip all `x-codex-*` for Gemini. Replace auth header with the Gemini bearer.

### 🟡 P18. Guardian uses a structured-output review call
**Root cause:** `core/src/guardian/mod.rs:62` — `GuardianAssessment { risk_level, user_authorization, outcome, rationale }`, a separate model call with structured output + `prompt_cache_key_override_for_review_session`.
**Fix:** route the guardian session through `gemini-adapter`. Use `response_format: json_schema` (compat) / `responseSchema` (native). The guardian call has **no tools**, so it avoids the structured-output + forced-tool 400. Strip the cache-key override. Keep `temperature = 1.0` and let `responseSchema` enforce the format instead of lowering temperature.

### 🟡 P19. Model capabilities config
**Root cause:** `ModelInfo` (`protocol/src/openai_models.rs:284`) is config-driven with fallback metadata.
**Fix:** add entries:
```
gemini-3.5-flash:     context_window=1_000_000, max_output_tokens=64_000,
                      supports_parallel_tool_calls=false, supports_reasoning_summaries=true,
                      auto_compact_token_limit=900_000, effective_context_window_percent=90
gemini-3.1-pro-preview: context_window=1_000_000  (deeper reasoning; slower)
```
`auto_compact_token_limit` drives when §P3 compaction fires — set it carefully.

### 🟡 P20. Auth/login flow
Covered in §3. Add `GeminiVertexAdc` + `GeminiApiKey` to `AuthMode`; add ADC token provider with refresh; add the 401-refresh branch for ADC.

### 🔵 P21. Voice / realtime — DEFERRED
`realtime-webrtc` (SDP offer/answer, OpenAI Realtime API). **Disable for this build.** Independent of coding core. If needed later, port separately to Gemini Live API.

### 🔵 P22. code-mode (TypeScript pragma)
**Root cause:** `code-mode/src/description.rs:121` — model writes TS with `// @exec:` pragma to call tools by namespace.
**Fix (eval, then decide):** keep code-mode and evaluate with Flash. Gemini leans toward plain function calling over free-form code. If pragma adherence is poor, disable code-mode for Gemini and fall back to JSON function calling (toggle via `is_code_mode_nested_tool`).

### 🟡 P23. MCP file upload was hosted
**Root cause:** `core/src/mcp_openai_file.rs` uploads via `/backend-api/files` (OpenAI hosted storage).
**Fix:** use Gemini Files API (`media.upload`) or handle files locally (read directly into a content part). Local is usually sufficient for a coding agent.

### 🟢 P24. Hardcoded base URL
**Root cause:** `core/src/config/mod.rs:~3526` defaults `chatgpt_base_url = "https://chatgpt.com/backend-api/"`.
**Fix:** make base URL provider-driven (factory detection §5). Gemini path → Vertex/AI Studio endpoint; OpenAI path unchanged.

---

## 7. Multimodal (low risk)
`ContentItem::InputImage { image_url }` (`protocol/src/models.rs`) uses a data URL via `image.into_data_url()`. OpenAI chat completions and Gemini multimodal both accept `image_url`/inline data. Verify Gemini accepts base64 data URLs on the chosen endpoint; native API uses `inlineData { mimeType, data }` — map accordingly in `request_translator.rs`.

---

## 8. Build order (risk-first)

**Phase 1 — spine, prove signatures work:**
1. `gemini-adapter` crate skeleton + factory detection (§5, P24).
2. `auth.rs`: ADC + API key, token refresh (§3, P20).
3. `model_config.rs`: ModelInfo entries (P19).
4. `request_translator.rs` + `response_translator.rs`: minimal `generateContent` round-trip, no tools yet. Map usage (P15), temperature (P10), thinking (P11).
5. **P1 + P13 thought_signature capture/round-trip** with a single function tool. This is the make-or-break milestone — get a 3-step tool loop working without 400s before anything else.

**Phase 2 — tools & web:**
6. `tool_translator.rs` + sanitizer (P7), strip `parallel_tool_calls` (P8), id-based result matching.
7. apply_patch via shell-intercept (P6).
8. Client-side `web_search`/`web_fetch` (P4).

**Phase 3 — robustness:**
9. `error.rs` Google RPC retry/fallback (P14).
10. `compaction.rs` client-side (P3).
11. Strip `x-codex-*` + `prompt_cache_key`, stable prefix (P16, P17).
12. `instructions` → systemInstruction + trim (P9).
13. Guardian structured output (P18). Disable hosted memory (P5) + map MCP files (P23).

**Phase 4 — behavioral eval:**
14. Plan-mode prompting / ANY-mode (P12).
15. code-mode eval, fallback to function calling if needed (P22).
16. Subagents end-to-end (Flash is well-suited to many parallel subagents).

---

## 9. Test checklist
- [ ] 3+ step tool loop with no `thought_signature` 400 (the core gate).
- [ ] Parallel tool calls resolve by id, not position.
- [ ] `apply_patch` via shell creates/edits/deletes files correctly.
- [ ] `web_search` + `web_fetch` execute client-side and feed results back.
- [ ] 429 with short `retryDelay` retries silently; long lockout triggers Flash↔Pro fallback.
- [ ] Compaction fires at ~900K tokens and preserves function-call history coherence.
- [ ] Token usage (incl. `thoughtsTokenCount`) reported accurately.
- [ ] Guardian returns valid `GuardianAssessment` JSON.
- [ ] Subagents spawn and complete under Flash.
- [ ] Plan mode pauses to ask before editing (eval, tune prompt/ANY mode).

---

## 10. Quick "gotcha" reference
- Use **native `generateContent`**, not OpenAI-compat, for the agentic path (signatures).
- `thought_signature` is mandatory on every function-call turn — never drop it mid-turn.
- `temperature = 1.0` on Gemini 3.x.
- Set `thinking_level` explicitly (default fell to `medium`); never send `thinking_budget` alongside it.
- Strip `parallel_tool_calls`, `prompt_cache_key`, and all `x-codex-*` headers.
- Match parallel tool results by id.
- Gemini accepts only a subset of OpenAPI schema — sanitize tool params.
- Parse 429 from Google RPC `RetryInfo`/`QuotaFailure`, not `Retry-After`.
- Vertex = ADC OAuth2 token (no API keys); AI Studio = API key.

---

# Appendix A — Concrete shapes & examples

All shapes below match the native `generateContent` wire format verified against
https://ai.google.dev/gemini-api/docs/thought-signatures and
https://ai.google.dev/gemini-api/docs/function-calling (last verified 2026-06).
Examples use Codex-flavored tools (`shell`, `apply_patch`) but are structurally
identical to the docs. Re-verify field names before implementing.

## A.1 Native request skeleton (`GenerateContentRequest`)

A `Part` in Gemini is a flat object: `functionCall`/`text`/`functionResponse`/
`inlineData` and `thoughtSignature` are **sibling fields of the same part object**.

```json
{
  "systemInstruction": {
    "role": "user",
    "parts": [{ "text": "<Codex base_instructions, TRIMMED for Gemini 3>" }]
  },
  "contents": [
    { "role": "user", "parts": [{ "text": "Add a retry to fetch_user()." }] }
  ],
  "tools": [
    {
      "functionDeclarations": [
        {
          "name": "shell",
          "description": "Run a shell command (apply_patch is intercepted here).",
          "parameters": {
            "type": "object",
            "properties": {
              "command": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["command"]
          }
        }
      ]
    }
  ],
  "toolConfig": {
    "functionCallingConfig": { "mode": "AUTO" }
  },
  "generationConfig": {
    "temperature": 1.0,
    "maxOutputTokens": 64000,
    "thinkingConfig": { "thinkingLevel": "high", "includeThoughts": true }
  }
}
```

Notes: `temperature` MUST be `1.0` for Gemini 3 (P10). Set `thinkingLevel`
explicitly (P11) — never also send `thinkingBudget`. `toolConfig.mode` is
`AUTO` normally; switch to `ANY` with `allowedFunctionNames` in plan phase (P12).
`role` is only `"user"` or `"model"` (no `"system"`/`"developer"` — use
`systemInstruction`).

## A.2 thought_signature round-trip (P1) — THE critical pattern

### Native — sequential (multi-step), one tool per step
The signature comes back on the `functionCall` part. Echo it back **in the exact
part** on the next request.

Model response (step 1):
```json
{
  "candidates": [{
    "content": {
      "role": "model",
      "parts": [{
        "functionCall": { "name": "shell", "args": { "command": ["cat", "src/api.rs"] } },
        "thoughtSignature": "<SIG_1>"
      }]
    },
    "finishReason": "STOP"
  }]
}
```

Next request `contents` (you append model FC **with** `thoughtSignature`, then the tool result as a `user` part):
```json
[
  { "role": "user",  "parts": [{ "text": "Add a retry to fetch_user()." }] },
  { "role": "model", "parts": [{
      "functionCall": { "name": "shell", "args": { "command": ["cat", "src/api.rs"] } },
      "thoughtSignature": "<SIG_1>"
  }]},
  { "role": "user",  "parts": [{
      "functionResponse": { "name": "shell", "response": { "stdout": "...file contents..." } }
  }]}
]
```
Step 2 returns a new `functionCall` with `<SIG_2>`; step 3 request must carry
**both** `<SIG_1>` and `<SIG_2>` in their original parts.

### Native — parallel calls in one response
The signature is attached to the **first** `functionCall` part only; subsequent
parallel calls have **no** signature. Echo back exactly that way (first has it,
rest don't). Validation rule: when the model returns `FC1+sig, FC2`, you must send
back `FC1+sig, FC2, FR1, FR2` — **do NOT interleave** as `FC1+sig, FR1, FC2, FR2`
or you get a 400.

### OpenAI-compat shape (if used)
The signature rides in `tool_calls[].extra_content.google.thought_signature`:
```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "function-call-1",
    "type": "function",
    "function": { "name": "shell", "arguments": "{\"command\":[\"cat\",\"src/api.rs\"]}" },
    "extra_content": { "google": { "thought_signature": "<SIG_1>" } }
  }]
}
```
Compat gateways frequently strip `extra_content` → use native (§1).

### Validation rules to encode in `signature_store.rs`
- Required & validated only for the **current turn** (newest `user` text msg →
  now). Previous turns are not re-validated.
- The **first** `functionCall` part in **each step** of the current turn must
  carry its signature; omitting it → `400 "... is missing a thought_signature"`.
- Required even at `thinkingLevel: "minimal"` on Gemini 3 Flash.
- Non-function text parts may also return a signature in the **last** part;
  echoing it is recommended (not validated) for reasoning quality.
- **Streaming:** when no function call, the signature can arrive in an
  empty-text part — parse the stream until `finishReason` (P13).
- **Escape hatch** (for FCs from another model / deterministic client injection
  with no real signature): set the dummy value
  `"context_engineering_is_the_way_to_go"` or `"skip_thought_signature_validator"`
  to skip validation (quality may degrade). Use only when unavoidable.

## A.3 `ResponseItem` ↔ Gemini `Part` mapping

Translator must convert both directions. `protocol/src/models.rs` variants:

| Codex `ResponseItem` / `ContentItem` | Gemini part (native) | Notes |
|---|---|---|
| `Message { role: user, content: InputText }` | `{ role:"user", parts:[{text}] }` | role → `user` |
| `Message { role: assistant, content }` | `{ role:"model", parts:[{text}] }` | role → `model` |
| `ContentItem::InputImage { image_url (data URL) }` | `{ inlineData:{ mimeType, data(base64) } }` | split data URL into mime+base64 |
| `Reasoning { encrypted_content }` | `text` part `thoughtSignature` (model turn) | map encrypted_content ↔ thoughtSignature |
| `FunctionCall { name, arguments(string) }` | `{ functionCall:{ name, args(object) }, thoughtSignature }` | parse args string → JSON object; attach sig |
| `FunctionCallOutput { output }` | `{ role:"user", parts:[{ functionResponse:{ name, response } }] }` | tool result becomes a `user` part |
| `CustomToolCall` (apply_patch freeform) | — (drop; route via `shell`) | P6 |
| `WebSearchCall` (hosted) | — (drop; client-side `web_search` fn) | P4 |
| `LocalShellCall` (hosted) | `functionCall` to client-side `shell` | execute locally |

Key gotchas: Gemini `args` is a JSON **object**, but Codex keeps function-call
arguments as a **raw string** — parse on the way out, stringify on the way back.
Tool results are `functionResponse` parts under a **`user`** role turn.

## A.4 SSE chunk shape (`streamGenerateContent?alt=sse`)

Each SSE line is `data: <partial GenerateContentResponse>`. Accumulate across
chunks until `finishReason` appears.

```
data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Loo"}]}}]}
data: {"candidates":[{"content":{"role":"model","parts":[{"text":"king at the file"}]}}]}
data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"shell","args":{"command":["cat","src/api.rs"]}},"thoughtSignature":"<SIG_1>"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":812,"candidatesTokenCount":47,"thoughtsTokenCount":260,"totalTokenCount":1119}}
```

Accumulator rules:
- Concatenate `text` deltas in order.
- A `functionCall` may arrive whole in one chunk (native) — capture its
  `thoughtSignature` from the same part.
- `usageMetadata` typically arrives on the final chunk → map per P15.
- Stop only on `finishReason` (`STOP` / `MAX_TOKENS` / `SAFETY` / `MALFORMED_FUNCTION_CALL`).

## A.5 Rust type skeletons (`gemini-adapter`)

Model `Part` with optional sibling fields (matches the flat JSON exactly).

```rust
// ---- request ----
#[derive(Serialize)]
struct GenerateContentRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
    generation_config: GenerationConfig,
}

#[derive(Serialize, Deserialize)]
struct Content { role: String, parts: Vec<Part> } // role: "user" | "model"

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct Part {
    #[serde(skip_serializing_if = "Option::is_none")] text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")] function_response: Option<FunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")] inline_data: Option<InlineData>,
    // sibling of the above; THE signature field (P1)
    #[serde(skip_serializing_if = "Option::is_none")] thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] thought: Option<bool>,
}

#[derive(Serialize, Deserialize)]
struct FunctionCall { name: String, args: serde_json::Value }   // args is OBJECT
#[derive(Serialize, Deserialize)]
struct FunctionResponse { name: String, response: serde_json::Value }
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InlineData { mime_type: String, data: String }            // base64

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Tool { function_declarations: Vec<FunctionDeclaration> }
#[derive(Serialize)]
struct FunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value, // SANITIZED JSON schema (P7)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolConfig { function_calling_config: FunctionCallingConfig }
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FunctionCallingConfig {
    mode: String,                                   // "AUTO" | "ANY" | "NONE"
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,    // for plan-phase ANY (P12)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    temperature: f32,                               // 1.0 for Gemini 3 (P10)
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i64>,
    thinking_config: ThinkingConfig,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ThinkingConfig {
    thinking_level: String,                         // "minimal"|"low"|"medium"|"high"
    include_thoughts: bool,
}

// ---- response (streaming chunk) ----
#[derive(Deserialize)]
struct GenerateContentResponse {
    #[serde(default)] candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")] usage_metadata: Option<UsageMetadata>,
}
#[derive(Deserialize)]
struct Candidate {
    content: Content,
    #[serde(rename = "finishReason")] finish_reason: Option<String>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    prompt_token_count: Option<i64>,        // -> input_tokens
    candidates_token_count: Option<i64>,    // -> output_tokens
    thoughts_token_count: Option<i64>,      // -> reasoning_output_tokens (BILLED)
    cached_content_token_count: Option<i64>,// -> cached_input_tokens
    total_token_count: Option<i64>,         // -> total_tokens
}

// ---- Google RPC error (P14) ----
#[derive(Deserialize)]
struct GoogleApiError { error: GoogleApiErrorBody }
#[derive(Deserialize)]
struct GoogleApiErrorBody { code: i64, status: String, message: String, #[serde(default)] details: Vec<serde_json::Value> }
// details[] may contain "@type": ".../RetryInfo" { retryDelay: "52s" }
//                       and "@type": ".../QuotaFailure" { violations:[{ quotaId, quotaMetric }] }
```

## A.6 Auth header quick map

| Mode | base_url | Authorization |
|---|---|---|
| AI Studio | `https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse` | `Bearer <GEMINI_API_KEY>` (or `x-goog-api-key`) |
| Vertex (ADC) | `https://{LOC}-aiplatform.googleapis.com/v1/projects/{PROJ}/locations/{LOC}/publishers/google/models/{model}:streamGenerateContent?alt=sse` | `Bearer <ADC OAuth2 token>` (refresh ~55 min) |

Strip all `x-codex-*` headers (P17) and `prompt_cache_key` (P16) on the Gemini path.
