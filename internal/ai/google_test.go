package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// sse sends SSE events over a response writer.
func sse(w http.ResponseWriter, events []string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, e := range events {
		_, _ = w.Write([]byte("event: x\ndata: " + e + "\n\n"))
	}
}

// bodyCapture decodes the request body and hands it to the test via a buffered
// channel — race-free synchronization between the httptest handler goroutine
// and the test goroutine.
func bodyCapture(t *testing.T, ch chan<- map[string]any, w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decode request body: %v", err)
		return
	}
	ch <- body
}

func TestGoogleStreamTextAndThinking(t *testing.T) {
	bodyCh := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyCapture(t, bodyCh, w, r)
		sse(w, []string{
			`{"type":"interaction.created","interaction":{"id":"int_xyz","status":"created"}}`,
			`{"type":"interaction.in_progress","interaction":{"id":"int_xyz","status":"in_progress"}}`,
			`{"type":"step.start","index":0,"step":{"type":"thought"}}`,
			`{"type":"step.delta","index":0,"delta":{"type":"thought","text":"User wants an explanation."}}`,
			`{"type":"step.stop","index":0,"status":"done"}`,
			`{"type":"step.start","index":1,"step":{"type":"model_output"}}`,
			`{"type":"step.delta","index":1,"delta":{"type":"text","text":"Hello"}}`,
			`{"type":"step.delta","index":1,"delta":{"type":"text","text":" world"}}`,
			`{"type":"step.stop","index":1,"status":"done"}`,
			`{"type":"interaction.completed","interaction":{"id":"int_xyz","status":"completed","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}}`,
		})
	}))
	defer srv.Close()

	p := NewGoogleProvider("test-key", srv.URL)
	ch, err := p.Stream(context.Background(), StreamRequest{
		Model:    "gemini-3.5-flash",
		System:   "You are helpful.",
		Messages: []ChatMessage{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var text, thinking string
	var done StreamEvent
	for ev := range ch {
		switch ev.Type {
		case "token":
			text += ev.Text
		case "thinking":
			thinking += ev.Text
		case "done":
			done = ev
		case "error":
			t.Fatalf("unexpected error event: %v", ev.Err)
		}
	}

	gotBody := <-bodyCh
	if thinking != "User wants an explanation." {
		t.Errorf("thinking = %q", thinking)
	}
	if !strings.Contains(text, "Hello world") {
		t.Errorf("text = %q", text)
	}
	if done.FinishReason != "end_turn" {
		t.Errorf("finish reason = %q", done.FinishReason)
	}
	if done.InputTokens != 10 || done.OutputTokens != 5 {
		t.Errorf("usage = %d/%d", done.InputTokens, done.OutputTokens)
	}
	if gotBody["model"] != "gemini-3.5-flash" {
		t.Errorf("model = %v", gotBody["model"])
	}
	if gotBody["system_instruction"] != "You are helpful." {
		t.Errorf("system_instruction = %v", gotBody["system_instruction"])
	}
	if gotBody["store"] != true || gotBody["stream"] != true {
		t.Errorf("store/stream = %v/%v", gotBody["store"], gotBody["stream"])
	}
	if gotBody["input"] != "Hi" {
		t.Errorf("input = %v", gotBody["input"])
	}
}

func TestGoogleThoughtSummaryContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w, []string{
			`{"type":"step.start","index":0,"step":{"type":"thought"}}`,
			`{"type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"text":"Reasoning about the clues."}}}`,
			`{"type":"step.stop","index":0,"status":"done"}`,
			`{"type":"interaction.completed","interaction":{"id":"i","status":"completed"}}`,
		})
	}))
	defer srv.Close()

	p := NewGoogleProvider("test-key", srv.URL)
	ch, err := p.Stream(context.Background(), StreamRequest{
		Model:    "gemini-3.5-flash",
		Messages: []ChatMessage{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var thinking string
	for ev := range ch {
		if ev.Type == "thinking" {
			thinking += ev.Text
		}
		if ev.Type == "error" {
			t.Fatalf("error event: %v", ev.Err)
		}
	}
	if thinking != "Reasoning about the clues." {
		t.Errorf("thinking = %q", thinking)
	}
}

func TestGoogleThinkLevelMapping(t *testing.T) {
	req := func(level string, budget int) StreamRequest {
		return StreamRequest{Model: "gemini-3.5-flash", ThinkLevel: level, ThinkingBudget: budget,
			Messages: []ChatMessage{{Role: "user", Content: "Hi"}}}
	}
	cases := []struct {
		level  string
		budget int
		want   string
	}{
		{"", 0, ""},
		{"low", 0, "low"},
		{"medium", 0, "medium"},
		{"high", 0, "high"},
		{"true", 0, "high"},
		{"false", 0, "minimal"},
		{"", 4096, "high"},
	}
	for _, c := range cases {
		bodyCh := make(chan map[string]any, 1)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			bodyCapture(t, bodyCh, w, r)
			sse(w, []string{`{"type":"interaction.completed","interaction":{"id":"i","status":"completed"}}`})
		}))
		p := NewGoogleProvider("k", srv.URL)
		ch, err := p.Stream(context.Background(), req(c.level, c.budget))
		if err != nil {
			srv.Close()
			t.Fatalf("Stream(%q): %v", c.level, err)
		}
		for range ch {
		}
		body := <-bodyCh
		srv.Close()
		gc, _ := body["generation_config"].(map[string]any)
		var got string
		if gc != nil {
			got, _ = gc["thinking_level"].(string)
		}
		if got != c.want {
			t.Errorf("ThinkLevel=%q budget=%d: thinking_level=%q, want %q", c.level, c.budget, got, c.want)
		}
		// thinking_summaries=auto must always be sent so the API streams
		// visible thought summaries instead of opaque signatures.
		if gc == nil || gc["thinking_summaries"] != "auto" {
			t.Errorf("ThinkLevel=%q budget=%d: thinking_summaries = %v, want auto", c.level, c.budget, gc["thinking_summaries"])
		}
	}
}

// TestGoogleMaxOutputClamp verifies a large configured budget is clamped to the
// model's known output ceiling (gemini-2.5-flash-lite caps at 8192).
func TestGoogleMaxOutputClamp(t *testing.T) {
	bodyCh := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyCapture(t, bodyCh, w, r)
		sse(w, []string{`{"type":"interaction.completed","interaction":{"id":"i","status":"completed"}}`})
	}))
	defer srv.Close()

	p := NewGoogleProvider("k", srv.URL)
	ch, err := p.Stream(context.Background(), StreamRequest{
		Model:     "gemini-2.5-flash-lite",
		Messages:  []ChatMessage{{Role: "user", Content: "Hi"}},
		MaxTokens: 32000,
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for range ch {
	}
	body := <-bodyCh
	gc, _ := body["generation_config"].(map[string]any)
	if gc["max_output_tokens"] != float64(8192) {
		t.Errorf("max_output_tokens = %v, want 8192", gc["max_output_tokens"])
	}
}

func TestGoogleToolRoundTrip(t *testing.T) {
	var calls int
	bodyCh := make(chan map[string]any, 2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			// Real live-API shape: arguments arrive as "arguments_delta" with an
			// "arguments" field, and the tool pause is interaction.completed with
			// status "requires_action".
			sse(w, []string{
				`{"event_type":"interaction.created","interaction":{"id":"int_001","status":"created"}}`,
				`{"event_type":"step.start","index":0,"step":{"type":"function_call","id":"fc_1","name":"get_weather"}}`,
				`{"event_type":"step.delta","index":0,"delta":{"arguments":"{\"location\": \"Boston, MA\"}","type":"arguments_delta"}}`,
				`{"event_type":"step.stop","index":0}`,
				`{"event_type":"interaction.completed","interaction":{"id":"int_001","status":"requires_action","usage":{"total_input_tokens":65,"total_output_tokens":16}}}`,
			})
			return
		}
		bodyCapture(t, bodyCh, w, r)
		sse(w, []string{
			`{"type":"step.start","index":0,"step":{"type":"model_output"}}`,
			`{"type":"step.delta","index":0,"delta":{"type":"text","text":"It's 52F and rainy in Boston."}}`,
			`{"type":"interaction.completed","interaction":{"id":"int_002","status":"completed"}}`,
		})
	}))
	defer srv.Close()

	p := NewGoogleProvider("test-key", srv.URL)

	// Turn 1: model calls get_weather.
	ch1, err := p.Stream(context.Background(), StreamRequest{
		Model:    "gemini-3.5-flash",
		Messages: []ChatMessage{{Role: "user", Content: "Weather in Boston?"}},
		Tools: []ToolDef{{Name: "get_weather", Description: "Get weather",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{"location": map[string]any{"type": "string"}}}}},
	})
	if err != nil {
		t.Fatalf("Stream 1: %v", err)
	}
	var tc, done StreamEvent
	for ev := range ch1 {
		switch ev.Type {
		case "tool_call":
			tc = ev
		case "done":
			done = ev
		case "error":
			t.Fatalf("error event: %v", ev.Err)
		}
	}
	if tc.ToolID != "fc_1" || tc.ToolName != "get_weather" {
		t.Fatalf("tool_call = %+v", tc)
	}
	if tc.ToolInput["location"] != "Boston, MA" {
		t.Errorf("tool input = %v", tc.ToolInput)
	}
	// The requires_action completion must surface as a tool_use finish reason.
	if done.FinishReason != "tool_use" {
		t.Errorf("finish reason = %q, want tool_use", done.FinishReason)
	}
	if done.InputTokens != 65 || done.OutputTokens != 16 {
		t.Errorf("usage = %d/%d, want 65/16", done.InputTokens, done.OutputTokens)
	}

	// Turn 2: submit the result; must chain via previous_interaction_id.
	ch2, err := p.Stream(context.Background(), StreamRequest{
		Model: "gemini-3.5-flash",
		Messages: []ChatMessage{
			{Role: "user", Content: "Weather in Boston?"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "fc_1", Name: "get_weather", Input: tc.ToolInput}}},
			{Role: "user", ToolResults: []ToolResult{{ToolUseID: "fc_1", Name: "get_weather", Content: "52F, rain"}}},
		},
		Tools: []ToolDef{{Name: "get_weather", Description: "Get weather",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}},
	})
	if err != nil {
		t.Fatalf("Stream 2: %v", err)
	}
	var final string
	for ev := range ch2 {
		if ev.Type == "token" {
			final += ev.Text
		}
		if ev.Type == "error" {
			t.Fatalf("error event: %v", ev.Err)
		}
	}
	secondBody := <-bodyCh
	if !strings.Contains(final, "52F") {
		t.Errorf("final = %q", final)
	}
	if secondBody["previous_interaction_id"] != "int_001" {
		t.Errorf("previous_interaction_id = %v", secondBody["previous_interaction_id"])
	}
	in, ok := secondBody["input"].([]any)
	if !ok || len(in) != 1 {
		t.Fatalf("input = %v", secondBody["input"])
	}
	part, _ := in[0].(map[string]any)
	if part["type"] != "function_result" || part["call_id"] != "fc_1" || part["name"] != "get_weather" {
		t.Errorf("function_result part = %v", part)
	}
}

// TestGoogleCrossTurnChaining verifies a new user turn after a tool-using turn
// chains onto the previous turn's last interaction.
func TestGoogleCrossTurnChaining(t *testing.T) {
	var calls int
	bodyCh := make(chan map[string]any, 3)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyCapture(t, bodyCh, w, r)
		calls++
		if calls == 1 {
			sse(w, []string{
				`{"type":"interaction.created","interaction":{"id":"int_001","status":"created"}}`,
				`{"type":"step.start","index":0,"step":{"type":"function_call","id":"fc_1","name":"read_file"}}`,
				`{"type":"step.delta","index":0,"delta":{"type":"arguments","partial_arguments":"{\"path\":\"a.go\"}"}}`,
				`{"type":"step.stop","index":0,"status":"waiting"}`,
				`{"type":"interaction.requires_action","interaction":{"id":"int_001","status":"requires_action"}}`,
			})
			return
		}
		sse(w, []string{
			`{"type":"step.start","index":0,"step":{"type":"model_output"}}`,
			`{"type":"step.delta","index":0,"delta":{"type":"text","text":"done"}}`,
			`{"type":"interaction.completed","interaction":{"id":"int_00","status":"completed"}}`,
		})
	}))
	defer srv.Close()

	p := NewGoogleProvider("test-key", srv.URL)

	// Turn 1, call 1: model calls read_file.
	ch1, _ := p.Stream(context.Background(), StreamRequest{
		Model:    "gemini-3.5-flash",
		Messages: []ChatMessage{{Role: "user", Content: "Read a.go"}},
		Tools:    []ToolDef{{Name: "read_file", Description: "Read a file"}},
	})
	for range ch1 {
	}
	// Turn 1, call 2: submit the tool result (final text response).
	ch2, _ := p.Stream(context.Background(), StreamRequest{
		Model: "gemini-3.5-flash",
		Messages: []ChatMessage{
			{Role: "user", Content: "Read a.go"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "fc_1", Name: "read_file", Input: map[string]any{"path": "a.go"}}}},
			{Role: "user", ToolResults: []ToolResult{{ToolUseID: "fc_1", Name: "read_file", Content: "package main"}}},
		},
		Tools: []ToolDef{{Name: "read_file", Description: "Read a file"}},
	})
	for range ch2 {
	}

	// Turn 2: new user message; must chain to the previous turn's interaction.
	ch3, _ := p.Stream(context.Background(), StreamRequest{
		Model: "gemini-3.5-flash",
		Messages: []ChatMessage{
			{Role: "user", Content: "Read a.go"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "fc_1", Name: "read_file", Input: map[string]any{"path": "a.go"}}}},
			{Role: "user", ToolResults: []ToolResult{{ToolUseID: "fc_1", Name: "read_file", Content: "package main"}}},
			{Role: "assistant", Content: "done"},
			{Role: "user", Content: "Now read b.go"},
		},
		Tools: []ToolDef{{Name: "read_file", Description: "Read a file"}},
	})
	for range ch3 {
	}

	<-bodyCh // turn 1, call 1
	<-bodyCh // turn 1, call 2
	third := <-bodyCh
	if third["previous_interaction_id"] == "" {
		t.Errorf("turn 2 did not chain: previous_interaction_id = %v", third["previous_interaction_id"])
	}
	if third["input"] != "Now read b.go" {
		t.Errorf("turn 2 input = %v", third["input"])
	}
}

func TestGoogleVisionInput(t *testing.T) {
	bodyCh := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyCapture(t, bodyCh, w, r)
		sse(w, []string{`{"type":"interaction.completed","interaction":{"id":"i","status":"completed"}}`})
	}))
	defer srv.Close()

	p := NewGoogleProvider("test-key", srv.URL)
	ch, err := p.Stream(context.Background(), StreamRequest{
		Model: "gemini-2.5-flash",
		Messages: []ChatMessage{{Role: "user", Content: "What is this?",
			Images: []string{"iVBORw0KGgoAAAANSUhEUg=="}}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	for range ch {
	}
	body := <-bodyCh
	in, ok := body["input"].([]any)
	if !ok || len(in) != 2 {
		t.Fatalf("input = %v", body["input"])
	}
	img, _ := in[0].(map[string]any)
	if img["type"] != "image" || img["mime_type"] != "image/png" || img["data"] != "iVBORw0KGgoAAAANSUhEUg==" {
		t.Errorf("image part = %v", img)
	}
}

func TestGoogleDiscoverModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("api key header = %q", r.Header.Get("x-goog-api-key"))
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "models/gemini-3.5-flash", "displayName": "Gemini 3.5 Flash", "supportedGenerationMethods": []string{"generateContent"}, "thinking": true},
				{"name": "models/gemini-3.1-flash-image", "displayName": "Gemini 3.1 Flash Image", "supportedGenerationMethods": []string{"createImage"}},
				{"name": "models/text-embedding-004", "displayName": "Embedding", "supportedGenerationMethods": []string{"embedContent"}},
			},
		})
	}))
	defer srv.Close()

	p := NewGoogleProvider("test-key", srv.URL)
	models, err := p.DiscoverModels(context.Background())
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	if len(models) != 1 || models[0].ID != "gemini-3.5-flash" {
		t.Errorf("models = %+v", models)
	}
	if models[0].Name != "Gemini 3.5 Flash" || !models[0].SupportsThinking {
		t.Errorf("model info = %+v", models[0])
	}
}

func TestBuildRawAdapterGoogle(t *testing.T) {
	p := buildRawAdapter(ProviderConfig{Type: "google", APIKey: "k"})
	if p == nil {
		t.Fatal("buildRawAdapter returned nil for type google")
	}
	gp, ok := p.(*GoogleProvider)
	if !ok {
		t.Fatalf("adapter type = %T", p)
	}
	if gp.base != "https://generativelanguage.googleapis.com/v1beta" {
		t.Errorf("default base = %q", gp.base)
	}
	if gp.host != "https://generativelanguage.googleapis.com" {
		t.Errorf("default host = %q", gp.host)
	}
}
