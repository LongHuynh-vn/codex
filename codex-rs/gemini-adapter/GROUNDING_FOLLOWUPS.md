# Gemini grounding — follow-up surfaces (DESIGN / RESEARCH ONLY)

Status: **not implemented.** This document scopes the richer Google Search grounding
surfaces beyond the shipped CHANGE 1 (which renders `groundingMetadata.webSearchQueries`
as a `WebSearchCall` "Searched the web for …" history cell, reusing the existing
Responses-path rendering).

**Every wire-shape detail below is UNVERIFIED** until a live capture confirms it against
our model. Do not add any struct, protocol field, or TUI surface on the basis of this
doc alone.

---

## Step 0 — REQUIRED FIRST (blocks every follow-up)

Capture a real **`gemini-3.5-flash`** grounded response and record exactly which
`groundingMetadata` fields are present and their shapes:
`groundingChunks`, `groundingSupports`, `searchEntryPoint`, `retrievalMetadata`.

- **Open question:** recent Flash models reportedly **omit `groundingChunks`** from
  `groundingMetadata`. Since our model is `gemini-3.5-flash`, it is unknown whether
  citation chunks even arrive for us. If they don't, Follow-ups A and B are moot until a
  model/config that returns them is in use.
- **How:** reuse the live-capture harness in `gemini-adapter/src/lib_tests.rs`
  (`collect_live_response`, ~lines 636–703) which already `eprintln!`s each raw chunk;
  run a grounded prompt with `gemini_search_mode` set and save the chunks as a fixture.
- **Exit criterion:** a committed fixture file showing the real field set. No protocol or
  TUI surface in this doc is "final" until that fixture exists and the structs are
  derived from it.

---

## Follow-up A — Source citations (`groundingChunks`)

- Shape (UNVERIFIED): `groundingChunks: [{ web: { uri, title } }]`. `uri` is typically a
  `vertexaisearch.cloud.google.com/grounding-api-redirect/…` redirect, not the raw source.
- Value: show the actual sources that grounded the answer (the real user payoff beyond
  "what was searched").
- Cost / blocker: **new surface.** `ContentItem` has no citation/annotation field today
  (verified at `protocol/src/models.rs:703-716`, variants `InputText` / `InputImage` /
  `OutputText` only). Surfacing sources needs either a new `ResponseItem` variant + event
  or a citation field on the assistant message — both touch the **shared protocol crate**
  used by the OpenAI/Responses path. Must be its own gated phase with the
  "Responses path stays byte-identical" invariant re-verified.
- Not parsed today: the `gemini-adapter` `GroundingMetadata` struct only has
  `web_search_queries`; `groundingChunks` is dropped.

## Follow-up B — Inline citation markers (`groundingSupports`)

- Shape (UNVERIFIED): `groundingSupports: [{ segment: { startIndex, endIndex },
  groundingChunkIndices: [...], confidenceScores: [...] }]`, mapping answer spans to the
  `groundingChunks` indices.
- Value: inline "[1]"-style markers tied to sources, with confidence.
- Cost / blocker: needs a **per-segment annotation surface on `ContentItem`** that does
  not exist. Strictly depends on Follow-up A (the chunks it indexes into). Most invasive;
  gate behind A. Do not scope as a rider on A.

## Follow-up C — Google "Search Suggestions" compliance (`searchEntryPoint`)

- Shape (UNVERIFIED): `searchEntryPoint.renderedContent` = HTML/CSS that Google's
  "Grounding with Google Search" terms require be displayed to the user.
- Constraint: a terminal TUI cannot render HTML/CSS. The shipped `WebSearchCall` queries
  cell is a partial, good-faith surrogate but is **not** the required rendered chip.
- Action: treat as a known compliance gap; decide policy (and whether a non-TUI surface
  is needed) before any GA of Gemini Search grounding.

---

## Invariant for all follow-ups

Gemini-only by construction; the OpenAI/Responses path must remain byte-identical; and
**no protocol edit lands without the Step 0 live capture** justifying the exact fields
being added.
