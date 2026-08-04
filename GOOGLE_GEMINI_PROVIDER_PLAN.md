# Plan: Add Google (Gemini) as a TolleCode provider via the Gemini Interactions API

**Status:** Plan only — no code changes made yet.
**Source of truth:** <https://ai.google.dev/gemini-api/docs/interactions-overview> and
<https://ai.google.dev/gemini-api/docs/migrate-to-interactions>

---

## 1. Decisions (locked)

| Decision | Value | Rationale |
|---|---|---|
| Integration surface | **Gemini Interactions API** (not the OpenAI-compatible endpoint) | Recommended/GA surface for all new work; OpenAI-compat causes model errors and loses Gemini-native features |
| State mode | **Stateful** (`store=true`, `previous_interaction_id` chaining) | Enables server-side history, implicit caching, lower token costs; data retained server-side (55 days paid / 1 day free) — accepted |
| Default model | **`gemini-3.5-flash`** | Current fast workhorse model (Gemini 3.x family) |
| Go SDK | **None** — call REST directly (`net/http` + SSE parsing) | Official Go SDK for Interactions API does not exist yet (Python `google-genai` ≥2.3.0, JS `@google/genai` ≥2.3.0 only) |
| Provider type id | `google` | Mirrors existing `anthropic` / `openai` / `ollama` / `custom` types |

---

## 2. The Gemini Interactions API (research summary)

- **Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta2/interactions`
- **Auth:** `x-goog-api-key: $GEMINI_API_KEY` header (not Bearer)
- **Request body (key fields):**
  ```json
  {
    "model": "gemini-3.5-flash",
    "input": "text" | [ { "type": "text", "text": "..." },
                        { "type": "image", "mime_type": "image/jpeg", "data": "<base64>" } ],
    "previous_interaction_id": "int_123",
    "tools": [ { "type": "google_search" },
               { "type": "function", "name": "...", "description": "...", "parameters": { ... } } ],
    "system_instruction": "...",
    "generation_config": { "thinking_level": "...", "temperature": 0.7, ... },
    "response_format": [ { "type": "text", "mime_type": "application/json", "schema": { ... } } ],
    "store": true,
    "background": false,
    "stream": true
  }
  ```
- **Response (non-streaming):** a stored `Interaction` resource:
  ```json
  {
    "id": "int_123",
    "status": "completed" | "requires_action",
    "steps": [
      { "type": "user_input", "status": "done", "content": [ { "type": "text", "text": "..." } ] },
      { "type": "thought", "status": "done", "content": [ { "type": "text", "text": "..." } ] },
      { "type": "function_call", "id": "fc_1", "name": "get_weather", "arguments": { "location": "Boston, MA" }, "status": "waiting" },
      { "type": "function_result", "call_id": "fc_1", "name": "get_weather", "result": [ { "type": "text", "text": "52°F, rain" } ] },
      { "type": "model_output", "status": "done", "content": [ { "type": "text", "text": "It's 52°F and rainy in Boston." } ] }
    ]
  }
  ```
- **Streaming (SSE):** same endpoint with `"stream": true`. Events (JSON in `data:` lines):
  - `interaction.created` — `{ interaction: { id, status: "created" } }`
  - `interaction.in_progress`
  - `step.start` — `{ index, step: { type: "thought" | "function_call" | "model_output" | ... }, id?, name? }`
  - `step.delta` — `{ index, delta: { type: "thought"|"text", text } | { type: "arguments", partial_arguments: "<json>" } }`
  - `step.stop` — `{ index, status }`
  - `interaction.requires_action` — tool use pause (agent must submit results)
  - `interaction.completed` — `{ interaction: { id, status: "completed", usage: { prompt_tokens, completion_tokens, total_tokens } } }`
- **Multi-turn / tool loop:** continue with `previous_interaction_id`; submit tool output as a new interaction with
  `input: { "type": "function_result", "call_id": "fc_1", "name": "...", "result": [ { "type": "text", "text": "..." } ] }`.
- **Current model lineup (per docs):** `gemini-3.6-flash`, `gemini-3.5-flash` **(default)**, `gemini-3.5-flash-lite`,
  `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-pro`,
  `gemini-2.5-flash`, `gemini-2.5-flash-lite` — plus image/TTS models and agents (Deep Research, Antigravity).
- **Limitations to respect:** safety settings and the Batch API are not available in the Interactions API; Gemini 3 does not support remote MCP (not relevant here).

---

## 3. Architecture

New **`internal/ai/google.go`** — a `GoogleProvider` implementing the existing `ai.Provider` interface
(`Stream(ctx, StreamRequest) (<-chan StreamEvent, error)` + `DiscoverModels(ctx) ([]ModelInfo, error)`),
registered as a new provider **type `google`** in `buildRawAdapter()`.

```
TolleCode agent loop  ──StreamRequest──▶  GoogleProvider  ──POST /v1beta2/interactions──▶  Gemini API
                       ◀─StreamEvent───   (REST + SSE)      ◀────────── SSE stream ───────
```

### 3.1 Request mapping (`StreamRequest` → `interactions.create`)

| TolleCode `StreamRequest` | Interactions API field |
|---|---|
| `System` | top-level `system_instruction` (required for 2.5+/3.x, not a chat role) |
| last user `Content` + `Images` | `input` array: `{type:"text", text}` + `{type:"image", mime_type, data}` |
| `Tools` (`[]ToolDef`) | `tools`: `[{type:"function", name, description, parameters}]` (schema passed verbatim) |
| `ThinkLevel` / `ThinkingBudget` | `generation_config.thinking_level` (map; enum values to verify — see §7) |
| `MaxTokens` | `generation_config` max output tokens |
| — | `store: true` (always, stateful mode), `stream: true` |

### 3.2 Response mapping (SSE → `StreamEvent`)

| Interactions SSE event | TolleCode `StreamEvent` |
|---|---|
| `step.start{type:"thought"}` / `step.delta{type:"thought"}` | `{Type: "thinking", Text}` |
| `step.delta{type:"text"}` | `{Type: "token", Text}` |
| `step.start{type:"function_call", id, name}` + `step.delta{type:"arguments", partial_arguments}` | accumulate JSON → `{Type: "tool_call", ToolID, ToolName, ToolInput}` |
| `interaction.completed` usage | `{Type: "done", InputTokens, OutputTokens, FinishReason}` |
| `interaction.requires_action` | end of turn (no `done`; agent executes tools then calls `Stream` again) |

### 3.3 Tool loop (stateful chaining)

- The provider keeps a mutex-guarded map of `conversationKey → last interaction ID`.
- `conversationKey` = hash of the message history *before* tool results (stable across tool round-trips of one agent run).
- **Turn N** (no tool results): create interaction, store returned `id` under the key.
- **Turn N+1** (messages carry `ToolResults`): post `previous_interaction_id` + `input` array of
  `function_result` parts (`call_id` from the stored `function_call` step id, `result` = `[{type:"text", text: content}]`).
- Parallel tool results map to multiple parts in the input array (schema to verify — see §7).
- Retry/backoff on 429/5xx for the initial POST, mirroring `OpenAIProvider.createStreamWithRetry`.

### 3.4 Model discovery

`DiscoverModels` → `GET https://generativelanguage.googleapis.com/v1beta/models` with `x-goog-api-key`,
filter to text-capable `gemini-*` model ids, map into `ModelInfo` (capabilities per family table).

---

## 4. Implementation steps (file by file)

1. **`internal/ai/google.go` (new)** — `GoogleProvider`:
   - `NewGoogleProvider(apiKey, endpoint string) *GoogleProvider` — endpoint defaults to
     `https://generativelanguage.googleapis.com/v1beta2`.
   - `Stream()` — build request (§3.1), POST with `stream:true`, parse SSE with `bufio.Scanner`
     over `data:` lines, translate events (§3.2), manage conversation state (§3.3).
   - `DiscoverModels()` (§3.4).
   - `openAI`-style model metadata table for the `gemini-*` family (context window, max output,
     vision, thinking, function calling).
2. **`internal/ai/manager.go`** —
   - `buildRawAdapter()`: `case "google": return NewGoogleProvider(cfg.APIKey, cfg.Endpoint)`.
   - `BestProvider()` `tierOf`: add `case "google"` (suggest tier after `openai`).
3. **`internal/cli/commands.go`** —
   - Provider picker: add `{"google", "Google Gemini (Interactions API)"}`.
   - `needsKey`: add `"google": true` (endpoint optional → do **not** add to `needsEndpoint`).
   - Prefill model list on add: current lineup with `gemini-3.5-flash` marked default.
4. **`internal/cli/repl.go`** — type→label map: add `"google": "google gemini"`.
5. **`cmd/tollecode/main.go`** — `specToType`: add `"google": "google"` so
   `tollecode launch google [--model gemini-3.5-flash]` works.
6. **`README.md`** — add Google row to the provider table + usage example.
7. **Server mode** — no changes required: `tollecode.yaml` providers flow through
   `httpserver.ConvertProviders` into the same `buildRawAdapter` switch, so
   `type: google` with `api_key: ${GEMINI_API_KEY}` works out of the box.

---

## 5. Usage & setup

```sh
# CLI
tollecode configure add                          # pick "Google Gemini", paste key (stored in ~/.tollecode/config.json)
tollecode launch google --model gemini-3.5-flash --task "explain this repo"
tollecode google  # (later: REPL provider selection)

# Server mode (tollecode.yaml)
# providers:
#   - id: google
#     type: google
#     api_key: ${GEMINI_API_KEY}
#     default_model: gemini-3.5-flash
#     models:
#       - gemini-3.5-flash
#       - gemini-3.6-flash
#       - gemini-2.5-flash
```

**Setup needed:** a Gemini API key from Google AI Studio (`GEMINI_API_KEY`, free tier available).
Keys are stored in `~/.tollecode/config.json` (matching existing providers), or via YAML
`${GEMINI_API_KEY}` expansion in server mode / `~/.tollecode/.env`.

**Note on retention:** stateful mode stores interactions server-side (55 days paid tier,
1 day free tier). This is a deliberate product decision — conversations sent to the Gemini API
are already external, but *retention* is new and worth documenting to users in the README.

---

## 6. Validation & testing

1. `go build ./...` — compiles cleanly.
2. `go test ./internal/ai/...` — new unit tests:
   - `internal/ai/google_test.go` with an `httptest` server replaying recorded Interactions
     fixtures: plain text, streaming text, thinking (`thought` steps), tool round-trip
     (`function_call` → `function_result` with `previous_interaction_id`), vision input, usage reporting.
   - Manager test: `buildRawAdapter` returns a `GoogleProvider` for `type: google`.
3. Live smoke test (with a real key): one-shot task, streaming REPL, tool-calling agent run, image input.
4. `git status` — only intended files changed.

---

## 7. Open items to verify against the REST reference during implementation

- Exact `generation_config.thinking_level` enum values (map from TolleCode `ThinkLevel`/`ThinkingBudget`).
- Whether multiple `function_result` parts are accepted in a single `input` array (parallel tool calls).
- Exact `tools` schema for function declarations (parameter casing: `type`/`properties`/`required`).
- Whether `interaction.completed` is also emitted on `requires_action` streams (finish-reason handling).
- Model listing endpoint/response shape on `v1beta/models` for `DiscoverModels`.

---

## 8. Out of scope (for now)

- Native grounding / `google_search` server-side tool exposure through the agent loop.
- `background=true` long-running execution (Deep Think / Deep Research agents).
- Vertex AI endpoint support (`endpoint` override is supported by design, but only
  `generativelanguage.googleapis.com` is targeted in this phase).
