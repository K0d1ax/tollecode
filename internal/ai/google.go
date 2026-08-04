package ai

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// GoogleProvider implements Provider for Google's Gemini Interactions API
// (https://ai.google.dev/gemini-api/docs/interactions-overview). There is no
// official Go SDK for the Interactions API yet (Python google-genai >= 2.3.0
// and JS @google/genai >= 2.3.0 only), so we call the REST endpoint directly
// and parse the SSE stream.
//
// Endpoint version: the Interactions endpoint is live on /v1beta (verified
// against the deployed API; the docs' v1beta2 example 404s). An endpoint
// override in the provider config still wins.
//
// State model: stateful mode (store=true) with previous_interaction_id
// chaining. Each create() call references the previous interaction's id so the
// server keeps the conversation history — the client only sends the new input
// (the user's message, or function_result parts after a tool round trip).
// Interactions are retained server-side (55 days paid / 1 day free tier).
type GoogleProvider struct {
	// base is the API root, e.g. https://generativelanguage.googleapis.com/v1beta.
	base string
	// host is the scheme+host used for the v1beta models catalog.
	host string
	// apiKey is sent as the x-goog-api-key header.
	apiKey string

	mu           sync.Mutex
	interactions map[string]string // conversationKey -> last interaction id
}

func NewGoogleProvider(apiKey, endpoint string) *GoogleProvider {
	base := strings.TrimRight(strings.TrimSpace(endpoint), "/")
	if base == "" {
		base = "https://generativelanguage.googleapis.com/v1beta"
	}
	// Derive the scheme+host for the v1beta models catalog. base may carry a path
	// (…/v1beta), so parsing the URL is required — a naive Index("/") would
	// truncate at the "//" inside the scheme.
	host := "https://generativelanguage.googleapis.com"
	if u, err := url.Parse(base); err == nil && u.Host != "" {
		scheme := u.Scheme
		if scheme == "" {
			scheme = "https"
		}
		host = scheme + "://" + u.Host
	}
	return &GoogleProvider{
		base:         base,
		host:         host,
		apiKey:       apiKey,
		interactions: map[string]string{},
	}
}

func (p *GoogleProvider) Stream(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error) {
	// Build the wire payload (separated so unit tests can inspect it).
	payload, conversationKey, err := p.buildPayload(req)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	resp, err := p.postWithRetry(ctx, "/interactions", body)
	if err != nil {
		return nil, err
	}

	ch := make(chan StreamEvent, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		var (
			interactionID     string
			lastStep          string // "thought" | "text" | "function_call"
			finishReason      string
			usageIn, usageOut int
		)
		// Accumulated text is emitted at sentence boundaries for parity with the
		// other providers.
		var textBuf strings.Builder
		// Tool call argument accumulation.
		type tcAcc struct {
			id   string
			name string
			buf  strings.Builder
		}
		var tc *tcAcc

		emitUpToSentence := func(force bool) {
			s := textBuf.String()
			if len(s) == 0 {
				return
			}
			if force {
				ch <- StreamEvent{Type: "token", Text: s}
				textBuf.Reset()
				return
			}
			lastCut := -1
			for i := 0; i < len(s)-1; i++ {
				c, next := s[i], s[i+1]
				if (c == '.' || c == '!' || c == '?') && (next == ' ' || next == '\n') {
					lastCut = i + 1
				} else if c == '\n' && next == '\n' {
					lastCut = i + 2
				}
			}
			if lastCut > 0 {
				ch <- StreamEvent{Type: "token", Text: s[:lastCut]}
				textBuf.Reset()
				textBuf.WriteString(s[lastCut:])
			}
		}

		// Inactivity watchdog: a silently hung connection must not block the
		// executor forever (mirrors Ollama).
		const inactivityTimeout = 2 * time.Minute
		watchdog := time.AfterFunc(inactivityTimeout, func() { resp.Body.Close() })
		defer watchdog.Stop()

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 1<<20), 1<<20)
		for scanner.Scan() {
			watchdog.Reset(inactivityTimeout)
			if ctx.Err() != nil {
				return
			}
			line := strings.TrimSpace(scanner.Text())
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "" || data == "[DONE]" {
				continue
			}
			var ev struct {
				Type  string `json:"type"`
				Event string `json:"event_type"`
				Step  struct {
					Type string `json:"type"`
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"step"`
				Index int `json:"index"`
				Delta struct {
					Type             string `json:"type"`
					Text             string `json:"text"`
					PartialArguments string `json:"partial_arguments"`
					Arguments        string `json:"arguments"`
					Content          struct {
						Text string `json:"text"`
					} `json:"content"`
				} `json:"delta"`
				Interaction struct {
					ID     string `json:"id"`
					Status string `json:"status"`
					Usage  struct {
						PromptTokens      int `json:"prompt_tokens"`
						CompletionTokens  int `json:"completion_tokens"`
						TotalInputTokens  int `json:"total_input_tokens"`
						TotalOutputTokens int `json:"total_output_tokens"`
					} `json:"usage"`
				} `json:"interaction"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				continue
			}
			evType := ev.Type
			if evType == "" {
				evType = ev.Event
			}

			switch evType {
			case "interaction.created", "interaction.in_progress":
				if ev.Interaction.ID != "" {
					interactionID = ev.Interaction.ID
				}
			case "step.start":
				switch ev.Step.Type {
				case "thought":
					lastStep = "thought"
				case "function_call":
					lastStep = "function_call"
					emitUpToSentence(true)
					tc = &tcAcc{id: ev.Step.ID, name: ev.Step.Name}
				case "model_output":
					lastStep = "text"
				}
			case "step.delta":
				switch ev.Delta.Type {
				case "thought", "thought_summary":
					// Thinking text may arrive under delta.text or under
					// delta.content.text (the thought_summary shape).
					if ev.Delta.Text != "" {
						ch <- StreamEvent{Type: "thinking", Text: ev.Delta.Text}
					} else if ev.Delta.Content.Text != "" {
						ch <- StreamEvent{Type: "thinking", Text: ev.Delta.Content.Text}
					}
				case "text":
					textBuf.WriteString(ev.Delta.Text)
					emitUpToSentence(false)
				case "arguments", "arguments_delta":
					// The live API sends function-call arguments as
					// {"type":"arguments_delta","arguments":"..."} (the docs show
					// "arguments"/"partial_arguments"); accept all three shapes.
					if tc != nil {
						tc.buf.WriteString(ev.Delta.PartialArguments)
						tc.buf.WriteString(ev.Delta.Arguments)
					}
				}
			case "step.stop":
				if lastStep == "function_call" && tc != nil {
					var input map[string]any
					_ = json.Unmarshal([]byte(tc.buf.String()), &input)
					if input == nil {
						input = map[string]any{} // no-arg tool
					}
					ch <- StreamEvent{
						Type:      "tool_call",
						ToolID:    tc.id,
						ToolName:  tc.name,
						ToolInput: input,
					}
					tc = nil
				}
			case "interaction.requires_action":
				// Tool pause: no interaction.completed follows. The agent executes
				// the tools and calls Stream again with previous_interaction_id.
				finishReason = "tool_use"
			case "interaction.completed":
				if ev.Interaction.Usage.PromptTokens > 0 || ev.Interaction.Usage.CompletionTokens > 0 {
					usageIn = ev.Interaction.Usage.PromptTokens
					usageOut = ev.Interaction.Usage.CompletionTokens
				} else {
					usageIn = ev.Interaction.Usage.TotalInputTokens
					usageOut = ev.Interaction.Usage.TotalOutputTokens
				}
				// The live API signals a tool pause as interaction.completed with
				// status "requires_action" (not a separate event as the docs show).
				if ev.Interaction.Status == "requires_action" {
					finishReason = "tool_use"
				} else {
					finishReason = "end_turn"
				}
			}
		}
		if err := scanner.Err(); err != nil && ctx.Err() == nil {
			ch <- StreamEvent{Type: "error", Err: fmt.Errorf("gemini stream: %w", err)}
			return
		}

		// Record the interaction id so the next call in this conversation can
		// chain with previous_interaction_id.
		if interactionID != "" && conversationKey != "" {
			p.mu.Lock()
			p.interactions[conversationKey] = interactionID
			p.mu.Unlock()
		}

		// Flush any buffered text.
		emitUpToSentence(true)

		if finishReason == "" {
			finishReason = "end_turn"
		}
		ch <- StreamEvent{
			Type:         "done",
			FinishReason: finishReason,
			InputTokens:  usageIn,
			OutputTokens: usageOut,
		}
	}()

	return ch, nil
}

// buildPayload maps a StreamRequest to the interactions.create request body and
// returns the conversation key used to chain tool round trips. The conversation
// key is the hash of the message history with tool-round-trip turns (assistant
// tool calls + user tool results) stripped — stable across all the calls of one
// agent turn. When the exact key misses and the trailing turns are a completed
// exchange (plain assistant turn + plain user turn), a fallback key derived by
// also stripping those is tried so a new user turn chains onto the previous
// turn's last interaction.
func (p *GoogleProvider) buildPayload(req StreamRequest) (map[string]any, string, error) {
	payload := map[string]any{
		"model":  req.Model,
		"store":  true,
		"stream": true,
	}
	if req.System != "" {
		payload["system_instruction"] = req.System
	}
	if len(req.Tools) > 0 {
		payload["tools"] = buildGoogleTools(req.Tools)
	}

	genCfg := map[string]any{}
	if level := googleThinkLevel(req); level != "" {
		genCfg["thinking_level"] = level
	}
	if req.MaxTokens > 0 {
		// Clamp to the model's real output ceiling when known — the configured
		// budget is provider-agnostic and can exceed what a small-output model
		// (e.g. gemini-2.5-flash-lite) allows, which the API rejects. Mirrors
		// OpenAIProvider.
		maxOut := req.MaxTokens
		if mo := googleModelInfo(req.Model).MaxOutputTokens; mo > 0 && maxOut > mo {
			maxOut = mo
		}
		genCfg["max_output_tokens"] = maxOut
	}
	// Stream visible thought summaries: without this the API only sends opaque
	// thought signatures, leaving the thinking pane empty. "auto" emits summary
	// content (delta.content.text) whenever the model reasons.
	genCfg["thinking_summaries"] = "auto"
	payload["generation_config"] = genCfg

	core := stripToolTurns(req.Messages)

	// Tool round trip: the last turn carries tool results.
	if n := len(req.Messages); n > 0 && req.Messages[n-1].Role == "user" && len(req.Messages[n-1].ToolResults) > 0 {
		key := conversationKey(core)
		prev := p.lookupInteraction(key)
		if prev != "" {
			payload["previous_interaction_id"] = prev
		}
		input, err := buildGoogleFunctionResults(req.Messages[n-1].ToolResults)
		if err != nil {
			return nil, "", err
		}
		payload["input"] = input
		return payload, key, nil
	}

	// Plain user turn: input is the last user message (text + images).
	lastUser, ok := lastPlainUserTurn(core)
	if !ok {
		return nil, "", fmt.Errorf("gemini: no user message to send")
	}
	key := conversationKey(core)
	prev := p.lookupInteraction(key)
	if prev == "" {
		// New user turn following a completed exchange — chain onto the previous
		// turn's last interaction. Tool turns are stripped from the fallback too:
		// if the previous turn ended with tool calls, its last interaction was
		// stored under the key of the tool-stripped history.
		prev = p.lookupInteraction(conversationKey(stripToolTurns(stripCompletedExchange(core))))
	}
	if prev != "" {
		payload["previous_interaction_id"] = prev
	}
	payload["input"] = buildGoogleUserInput(lastUser)
	return payload, key, nil
}

// lookupInteraction returns the last interaction id stored under key.
func (p *GoogleProvider) lookupInteraction(key string) string {
	if key == "" {
		return ""
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.interactions[key]
}

// googleThinkLevel maps TolleCode's ThinkLevel / ThinkingBudget onto the
// Interactions API thinking_level enum (minimal|low|medium|high). There is no
// "off" for most models — "minimal" is the closest to disabled. Unset returns ""
// so the model's default applies.
func googleThinkLevel(req StreamRequest) string {
	switch req.ThinkLevel {
	case "low", "medium", "high":
		return req.ThinkLevel
	case "true":
		return "high"
	case "false":
		return "minimal"
	}
	if req.ThinkingBudget > 0 {
		return "high"
	}
	return ""
}

// stripToolTurns removes trailing tool-round-trip turns (assistant turns with
// tool calls and user turns with tool results) from a message history, leaving
// the stable conversation core shared by every call of one agent turn.
func stripToolTurns(msgs []ChatMessage) []ChatMessage {
	end := len(msgs)
	for end > 0 {
		m := msgs[end-1]
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			end--
			continue
		}
		if m.Role == "user" && len(m.ToolResults) > 0 {
			end--
			continue
		}
		break
	}
	return msgs[:end]
}

// stripCompletedExchange removes a trailing plain user turn and the plain
// assistant turn that precedes it — i.e. the completed exchange of the previous
// turn — so the fallback conversation key can chain onto the prior interaction.
func stripCompletedExchange(core []ChatMessage) []ChatMessage {
	end := len(core)
	if end > 0 && core[end-1].Role == "user" && len(core[end-1].ToolResults) == 0 {
		end--
	}
	if end > 0 && core[end-1].Role == "assistant" && len(core[end-1].ToolCalls) == 0 {
		end--
	}
	return core[:end]
}

// lastPlainUserTurn returns the last user turn that is not a tool-result turn.
func lastPlainUserTurn(core []ChatMessage) (ChatMessage, bool) {
	for i := len(core) - 1; i >= 0; i-- {
		m := core[i]
		if m.Role == "user" && len(m.ToolResults) == 0 {
			return m, true
		}
	}
	return ChatMessage{}, false
}

// conversationKey hashes a message history into a stable conversation key.
func conversationKey(msgs []ChatMessage) string {
	b, _ := json.Marshal(msgs)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// buildGoogleUserInput renders a plain user turn as the interactions input: a
// bare text string when there are no images, otherwise an array of typed parts.
func buildGoogleUserInput(m ChatMessage) any {
	if m.Content == "" && len(m.Images) == 0 {
		return ""
	}
	if m.Content != "" && len(m.Images) == 0 {
		return m.Content
	}
	parts := make([]map[string]any, 0, len(m.Images)+1)
	// Images first, then text — the order the API docs use for multimodal input.
	for _, img := range m.Images {
		parts = append(parts, map[string]any{
			"type":      "image",
			"mime_type": DetectBase64MediaType(img),
			"data":      img,
		})
	}
	if m.Content != "" {
		parts = append(parts, map[string]any{"type": "text", "text": m.Content})
	}
	return parts
}

// buildGoogleFunctionResults renders tool results as function_result input parts.
// Parallel tool calls map to multiple parts in the input array.
func buildGoogleFunctionResults(results []ToolResult) ([]map[string]any, error) {
	parts := make([]map[string]any, 0, len(results))
	for _, tr := range results {
		if tr.ToolUseID == "" || tr.Name == "" {
			return nil, fmt.Errorf("gemini: tool result missing call id or name")
		}
		result := make([]map[string]any, 0, 2)
		if tr.Content != "" {
			result = append(result, map[string]any{"type": "text", "text": tr.Content})
		}
		if tr.ImageData != "" {
			mediaType := tr.ImageMediaType
			if mediaType == "" {
				mediaType = "image/jpeg"
			}
			result = append(result, map[string]any{"type": "image", "mime_type": mediaType, "data": tr.ImageData})
		}
		if len(result) == 0 {
			result = append(result, map[string]any{"type": "text", "text": "(no output)"})
		}
		parts = append(parts, map[string]any{
			"type":    "function_result",
			"call_id": tr.ToolUseID,
			"name":    tr.Name,
			"result":  result,
		})
	}
	return parts, nil
}

func buildGoogleTools(tools []ToolDef) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		schema := t.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, map[string]any{
			"type":        "function",
			"name":        t.Name,
			"description": t.Description,
			"parameters":  schema,
		})
	}
	return out
}

// postWithRetry POSTs JSON to endpoint+path, retrying 429 and 5xx with
// exponential backoff (up to 5 attempts), mirroring OpenAIProvider.
func (p *GoogleProvider) postWithRetry(ctx context.Context, path string, body []byte) (*http.Response, error) {
	const maxAttempts = 5
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		resp, err := p.post(ctx, path, body)
		if err != nil {
			// Transport errors (DNS, refused, …) are not retried — they surface
			// the misconfiguration immediately instead of hanging for ~30s of
			// backoff. Only 429/5xx responses are transient.
			return nil, err
		}
		if resp.StatusCode == 200 {
			return resp, nil
		}
		// Close explicitly (not deferred) so retried attempts don't pile up
		// unclosed response bodies.
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		lastErr = fmt.Errorf("gemini returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(detail)))
		if resp.StatusCode != 429 && resp.StatusCode < 500 {
			return nil, lastErr
		}
		wait := time.Duration(1<<attempt) * time.Second
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
	}
	return nil, lastErr
}

func (p *GoogleProvider) post(ctx context.Context, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.base+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", p.apiKey)
	return http.DefaultClient.Do(req)
}

// DiscoverModels lists text-capable Gemini models via the v1beta models catalog.
func (p *GoogleProvider) DiscoverModels(ctx context.Context) ([]ModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.host+"/v1beta/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-goog-api-key", p.apiKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("gemini models list returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(detail)))
	}
	var body struct {
		Models []struct {
			Name                       string   `json:"name"`
			DisplayName                string   `json:"displayName"`
			InputTokenLimit            int      `json:"inputTokenLimit"`
			OutputTokenLimit           int      `json:"outputTokenLimit"`
			SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
			Thinking                   bool     `json:"thinking"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, err
	}
	out := make([]ModelInfo, 0, len(body.Models))
	for _, m := range body.Models {
		id := strings.TrimPrefix(m.Name, "models/")
		if !strings.HasPrefix(id, "gemini-") {
			continue
		}
		info := googleModelInfo(id)
		if m.InputTokenLimit > 0 {
			info.ContextWindow = m.InputTokenLimit
		}
		if m.OutputTokenLimit > 0 {
			info.MaxOutputTokens = m.OutputTokenLimit
		}
		if m.DisplayName != "" {
			info.Name = m.DisplayName
		}
		// A model that doesn't advertise generateContent is unlikely to serve
		// chat (e.g. image/TTS-only variants); keep only text-capable ones.
		if len(m.SupportedGenerationMethods) > 0 {
			textCapable := false
			for _, method := range m.SupportedGenerationMethods {
				if method == "generateContent" || strings.Contains(strings.ToLower(method), "interactions") {
					textCapable = true
					break
				}
			}
			if !textCapable {
				continue
			}
		}
		out = append(out, info)
	}
	return out, nil
}

// GoogleModelInfo maps a Gemini model ID to known capability metadata
// (exported for use in handlers).
func GoogleModelInfo(id string) ModelInfo { return googleModelInfo(id) }

// googleModelInfo maps a Gemini model ID to known capability metadata. Context
// windows and output ceilings are approximate per family; unknown (future)
// models inherit safe defaults.
func googleModelInfo(id string) ModelInfo {
	type entry struct {
		prefix           string
		ctx, maxOut      int
		vision, thinking bool
	}
	// Longer prefixes first: "gemini-3.5-flash-lite" must not be captured by the
	// "gemini-3.5-flash" entry (and likewise 2.5-flash-lite vs 2.5-flash).
	table := []entry{
		{"gemini-3.6-flash", 1_000_000, 65_536, true, true},
		{"gemini-3.5-flash-lite", 1_000_000, 65_536, true, true},
		{"gemini-3.5-flash", 1_000_000, 65_536, true, true},
		{"gemini-3.1-pro-preview", 1_000_000, 65_536, true, true},
		{"gemini-3.1-flash-lite", 1_000_000, 65_536, true, true},
		{"gemini-3-flash-preview", 1_000_000, 65_536, true, true},
		{"gemini-2.5-pro", 1_000_000, 65_536, true, true},
		{"gemini-2.5-flash-lite", 1_000_000, 8_192, true, true},
		{"gemini-2.5-flash", 1_000_000, 65_536, true, true},
	}
	for _, e := range table {
		if strings.HasPrefix(id, e.prefix) {
			return ModelInfo{
				ID:                   id,
				Name:                 id,
				ContextWindow:        e.ctx,
				MaxOutputTokens:      e.maxOut,
				SupportsStreaming:    true,
				SupportsFunctionCall: true,
				SupportsVision:       e.vision,
				SupportsThinking:     e.thinking,
			}
		}
	}
	return ModelInfo{
		ID:                   id,
		Name:                 id,
		ContextWindow:        1_000_000,
		MaxOutputTokens:      65_536,
		SupportsStreaming:    true,
		SupportsFunctionCall: true,
		SupportsVision:       true,
		SupportsThinking:     true,
	}
}
