package server

import (
	"encoding/json"
	"testing"
)

// callToolsCall is a test helper that dispatches a tools/call RPC request.
func callToolsCall(t *testing.T, s *MCPServer, toolName string, args map[string]any) (any, *rpcError) {
	t.Helper()
	params, err := json.Marshal(map[string]any{
		"name":      toolName,
		"arguments": args,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return s.handleToolsCall(json.RawMessage(params))
}

func TestGroupedToolsList_DefaultOff(t *testing.T) {
	s := newMemoryTestServer(t)

	result, rErr := s.handleToolsList(nil)
	if rErr != nil {
		t.Fatalf("handleToolsList error: %v", rErr.Message)
	}
	toolsMap := result.(map[string]any)
	tools := toolsMap["tools"].([]tool)

	// Should return individual tools, not grouped
	names := make(map[string]bool, len(tools))
	for _, tl := range tools {
		names[tl.Name] = true
	}

	// Individual tools must be present
	if !names["store_memory"] {
		t.Error("expected store_memory in flat tool list")
	}
	if !names["repo_list"] {
		t.Error("expected repo_list in flat tool list")
	}

	// Grouped names must NOT be present
	if names["memory"] {
		t.Error("did not expect grouped 'memory' tool in flat list")
	}
	if names["repo"] {
		t.Error("did not expect grouped 'repo' tool in flat list")
	}
}

func TestGroupedToolsList_Enabled(t *testing.T) {
	s := newMemoryTestServer(t)
	s.config.ToolGrouping = true

	result, rErr := s.handleToolsList(nil)
	if rErr != nil {
		t.Fatalf("handleToolsList error: %v", rErr.Message)
	}
	toolsMap := result.(map[string]any)
	tools := toolsMap["tools"].([]tool)

	names := make(map[string]bool, len(tools))
	for _, tl := range tools {
		names[tl.Name] = true
	}

	// Grouped names must be present
	for _, expected := range []string{"repo", "memory", "memory_admin", "engineering", "session", "project_bank_view"} {
		if !names[expected] {
			t.Errorf("expected %q in grouped tool list", expected)
		}
	}

	// Individual tool names must NOT be present
	for _, absent := range []string{"store_memory", "recall_memory", "repo_list", "close_session"} {
		if names[absent] {
			t.Errorf("did not expect individual tool %q in grouped list", absent)
		}
	}
}

func TestGroupedDispatch_RepoList(t *testing.T) {
	s := newTestServer(t, "")

	// Dispatch via grouped tool name
	result, rErr := callToolsCall(t, s, "repo", map[string]any{
		"action": "list",
	})
	if rErr != nil {
		t.Fatalf("dispatch error: %v", rErr.Message)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestGroupedDispatch_MissingAction(t *testing.T) {
	s := newTestServer(t, "")

	_, rErr := callToolsCall(t, s, "repo", map[string]any{})
	if rErr == nil {
		t.Fatal("expected error for missing action")
	}
	if rErr.Code != rpcErrInvalidParams {
		t.Errorf("expected error code %d, got %d", rpcErrInvalidParams, rErr.Code)
	}
}

func TestGroupedDispatch_UnknownAction(t *testing.T) {
	s := newTestServer(t, "")

	_, rErr := callToolsCall(t, s, "repo", map[string]any{
		"action": "nonexistent",
	})
	if rErr == nil {
		t.Fatal("expected error for unknown action")
	}
	if rErr.Code != rpcErrInvalidParams {
		t.Errorf("expected error code %d, got %d", rpcErrInvalidParams, rErr.Code)
	}
}

func TestGroupedDispatch_MemoryStore(t *testing.T) {
	s := newMemoryTestServer(t)

	// Store via grouped
	result, rErr := callToolsCall(t, s, "memory", map[string]any{
		"action":  "store",
		"content": "test memory from grouped dispatch",
	})
	if rErr != nil {
		t.Fatalf("store via grouped dispatch error: %v", rErr.Message)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestGroupedDispatch_LegacyStillWorks(t *testing.T) {
	s := newMemoryTestServer(t)
	s.config.ToolGrouping = true // Even with grouping on, legacy names work

	result, rErr := callToolsCall(t, s, "store_memory", map[string]any{
		"content": "test memory via legacy name",
	})
	if rErr != nil {
		t.Fatalf("legacy dispatch error: %v", rErr.Message)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestResolveGroupedToolName(t *testing.T) {
	s := newTestServer(t, "")
	groups := s.buildToolGroupDefs()

	tests := []struct {
		name     string
		args     map[string]any
		expected string
	}{
		{"repo", map[string]any{"action": "list"}, "repo_list"},
		{"repo", map[string]any{"action": "read"}, "repo_read"},
		{"repo", map[string]any{"action": "search"}, "repo_search"},
		{"repo", map[string]any{}, "repo"},                                  // missing action → original name
		{"repo", map[string]any{"action": "nonexistent"}, "repo"},           // unknown action → original name
		{"repo_list", map[string]any{}, "repo_list"},                        // non-grouped name → pass through
		{"store_memory", map[string]any{"action": "store"}, "store_memory"}, // non-grouped name → pass through
	}

	for _, tt := range tests {
		t.Run(tt.name+"_"+tt.expected, func(t *testing.T) {
			got := resolveGroupedToolName(tt.name, tt.args, groups)
			if got != tt.expected {
				t.Errorf("resolveGroupedToolName(%q, %v) = %q, want %q", tt.name, tt.args, got, tt.expected)
			}
		})
	}
}

func TestGroupedToolsList_NoMemoryStore(t *testing.T) {
	s := newTestServer(t, "") // no memory store
	s.config.ToolGrouping = true

	result, rErr := s.handleToolsList(nil)
	if rErr != nil {
		t.Fatalf("handleToolsList error: %v", rErr.Message)
	}
	toolsMap := result.(map[string]any)
	tools := toolsMap["tools"].([]tool)

	names := make(map[string]bool, len(tools))
	for _, tl := range tools {
		names[tl.Name] = true
	}

	// repo should be present (no backend required)
	if !names["repo"] {
		t.Error("expected 'repo' in grouped list without memory store")
	}

	// memory groups should be absent
	for _, absent := range []string{"memory", "memory_admin", "engineering", "session", "project_bank_view"} {
		if names[absent] {
			t.Errorf("did not expect %q without memory store", absent)
		}
	}
}
