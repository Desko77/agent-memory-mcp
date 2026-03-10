package server

import (
	"fmt"
	"strings"
)

// groupedAction maps an action name to its legacy tool handler and original tool name.
type groupedAction struct {
	legacyName string
	handler    toolHandler
}

// toolGroupDef defines a grouped meta-tool with an action discriminator.
type toolGroupDef struct {
	name        string
	description string
	actions     map[string]groupedAction
	schema      map[string]any
	// requireMemory indicates this group requires a memory store.
	requireMemory bool
	// requireRAG indicates this group requires a RAG engine.
	requireRAG bool
	// requireHybrid indicates this group requires at least one of memory or RAG.
	requireHybrid bool
}

// buildGroupDispatcher returns a handler that dispatches on the "action" parameter.
func (s *MCPServer) buildGroupDispatcher(group toolGroupDef) toolHandler {
	return func(args map[string]any) (any, *rpcError) {
		action, ok := getString(args, "action")
		if !ok || strings.TrimSpace(action) == "" {
			actions := make([]string, 0, len(group.actions))
			for a := range group.actions {
				actions = append(actions, a)
			}
			return nil, &rpcError{
				Code:    rpcErrInvalidParams,
				Message: fmt.Sprintf("action parameter is required; available actions: %s", strings.Join(actions, ", ")),
			}
		}
		ga, ok := group.actions[action]
		if !ok {
			actions := make([]string, 0, len(group.actions))
			for a := range group.actions {
				actions = append(actions, a)
			}
			return nil, &rpcError{
				Code:    rpcErrInvalidParams,
				Message: fmt.Sprintf("unknown action %q for tool %q; available actions: %s", action, group.name, strings.Join(actions, ", ")),
			}
		}
		return ga.handler(args)
	}
}

// resolveGroupedToolName returns the legacy tool name if the tool name is a grouped
// meta-tool. Otherwise returns the original name unchanged.
func resolveGroupedToolName(name string, args map[string]any, groups []toolGroupDef) string {
	for _, g := range groups {
		if g.name != name {
			continue
		}
		action, ok := getString(args, "action")
		if !ok {
			return name
		}
		if ga, found := g.actions[action]; found {
			return ga.legacyName
		}
		return name
	}
	return name
}

// buildToolGroupDefs returns all group definitions. Called once during server init.
func (s *MCPServer) buildToolGroupDefs() []toolGroupDef {
	return []toolGroupDef{
		s.repoGroupDef(),
		s.memoryGroupDef(),
		s.memoryAdminGroupDef(),
		s.engineeringGroupDef(),
		s.searchGroupDef(),
		s.sessionGroupDef(),
	}
}

// ─── repo ───────────────────────────────────────────────────────────────────

func (s *MCPServer) repoGroupDef() toolGroupDef {
	return toolGroupDef{
		name:        "repo",
		description: "Repository file operations: list, read, or search files in allowlisted paths",
		actions: map[string]groupedAction{
			"list":   {legacyName: "repo_list", handler: s.callRepoList},
			"read":   {legacyName: "repo_read", handler: s.callRepoRead},
			"search": {legacyName: "repo_search", handler: s.callRepoSearch},
		},
		schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"list", "read", "search"},
					"description": "Operation to perform.\n" +
						"- list: list files/folders. Optional: path, max_depth\n" +
						"- read: read a file. Requires: path. Optional: offset, max_bytes\n" +
						"- search: search for text. Requires: query. Optional: path, max_results",
				},
				"path":        map[string]any{"type": "string", "description": "Relative path (empty for root in list/search)"},
				"max_depth":   map[string]any{"type": "integer", "minimum": 0, "description": "Max directory depth (list only)"},
				"offset":      map[string]any{"type": "integer", "minimum": 0, "description": "Byte offset to start reading (read only)"},
				"max_bytes":   map[string]any{"type": "integer", "minimum": 1, "description": "Max bytes to read (read only, default: 2MB)"},
				"query":       map[string]any{"type": "string", "description": "Search query string (search only)"},
				"max_results": map[string]any{"type": "integer", "minimum": 1, "description": "Max search results (search only, default: 200)"},
			},
			"required": []string{"action"},
		},
	}
}

// ─── memory ─────────────────────────────────────────────────────────────────

func (s *MCPServer) memoryGroupDef() toolGroupDef {
	return toolGroupDef{
		name:          "memory",
		description:   "Long-term memory operations: store, recall, update, delete, list, or get stats",
		requireMemory: true,
		actions: map[string]groupedAction{
			"store":  {legacyName: "store_memory", handler: s.callStoreMemory},
			"recall": {legacyName: "recall_memory", handler: s.callRecallMemory},
			"update": {legacyName: "update_memory", handler: s.callUpdateMemory},
			"delete": {legacyName: "delete_memory", handler: s.callDeleteMemory},
			"list":   {legacyName: "list_memories", handler: s.callListMemories},
			"stats":  {legacyName: "memory_stats", handler: s.callMemoryStats},
		},
		schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"store", "recall", "update", "delete", "list", "stats"},
					"description": "Operation to perform.\n" +
						"- store: save a memory. Requires: content. Optional: title, type, tags, context, importance\n" +
						"- recall: search memories. Requires: query. Optional: type, context, tags, limit\n" +
						"- update: modify a memory. Requires: id. Optional: content, title, tags, importance\n" +
						"- delete: remove a memory. Requires: id\n" +
						"- list: browse memories. Optional: type, context, limit\n" +
						"- stats: get memory statistics. No additional params",
				},
				"content": map[string]any{"type": "string", "description": "Memory content (store, update)"},
				"title":   map[string]any{"type": "string", "description": "Short title (store, update)"},
				"type": map[string]any{
					"type":        "string",
					"enum":        []string{"episodic", "semantic", "procedural", "working", "all"},
					"description": "Memory type filter (store uses episodic/semantic/procedural/working; recall/list also accept 'all')",
					"default":     "semantic",
				},
				"tags": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Tags for categorization (store, recall, list)",
				},
				"context":    map[string]any{"type": "string", "description": "Task/session/project context (store, recall, list)"},
				"importance": map[string]any{"type": "number", "minimum": 0, "maximum": 1, "description": "Importance 0.0-1.0 (store, update)", "default": 0.5},
				"query":      map[string]any{"type": "string", "description": "Search query (recall)"},
				"id":         map[string]any{"type": "string", "description": "Memory ID (update, delete)"},
				"limit": map[string]any{
					"type":    "integer",
					"minimum": 1,
					"maximum": 100,
					"default": 10,
					"description": "Max results (recall, list)",
				},
			},
			"required": []string{"action"},
		},
	}
}

// ─── memory_admin ───────────────────────────────────────────────────────────

func (s *MCPServer) memoryAdminGroupDef() toolGroupDef {
	return toolGroupDef{
		name:          "memory_admin",
		description:   "Memory administration: merge duplicates, mark outdated, promote to canonical, view conflicts, list/recall canonical knowledge",
		requireMemory: true,
		actions: map[string]groupedAction{
			"merge":            {legacyName: "merge_duplicates", handler: s.callMergeDuplicates},
			"mark_outdated":    {legacyName: "mark_outdated", handler: s.callMarkOutdated},
			"promote":          {legacyName: "promote_to_canonical", handler: s.callPromoteToCanonical},
			"conflicts":        {legacyName: "conflicts_report", handler: s.callConflictsReport},
			"list_canonical":   {legacyName: "list_canonical_knowledge", handler: s.callListCanonicalKnowledge},
			"recall_canonical": {legacyName: "recall_canonical_knowledge", handler: s.callRecallCanonicalKnowledge},
		},
		schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"merge", "mark_outdated", "promote", "conflicts", "list_canonical", "recall_canonical"},
					"description": "Operation to perform.\n" +
						"- merge: merge duplicates. Requires: primary_id, duplicate_ids\n" +
						"- mark_outdated: mark a memory outdated. Requires: id. Optional: reason, superseded_by\n" +
						"- promote: promote to canonical. Requires: id. Optional: owner\n" +
						"- conflicts: report conflicts. Optional: context, service, type, tags, limit\n" +
						"- list_canonical: list canonical entries. Optional: type, context, service, tags, limit\n" +
						"- recall_canonical: search canonical only. Requires: query. Optional: type, context, service, tags, limit",
				},
				"id":            map[string]any{"type": "string", "description": "Memory ID (mark_outdated, promote)"},
				"primary_id":    map[string]any{"type": "string", "description": "Primary memory ID to keep (merge)"},
				"duplicate_ids": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "IDs to merge into primary (merge)"},
				"reason":        map[string]any{"type": "string", "description": "Why outdated (mark_outdated)"},
				"superseded_by": map[string]any{"type": "string", "description": "Newer memory ID (mark_outdated)"},
				"owner":         map[string]any{"type": "string", "description": "Owner or team (promote)"},
				"query":         map[string]any{"type": "string", "description": "Search query (recall_canonical)"},
				"context":       map[string]any{"type": "string", "description": "Project or task context"},
				"service":       map[string]any{"type": "string", "description": "Service or component name"},
				"type": map[string]any{
					"type":    "string",
					"enum":    []string{"episodic", "semantic", "procedural", "working", "all"},
					"default": "all",
					"description": "Memory type filter",
				},
				"tags":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Tag filter"},
				"limit": map[string]any{"type": "integer", "minimum": 1, "maximum": 50, "default": 10, "description": "Max results"},
			},
			"required": []string{"action"},
		},
	}
}

// ─── engineering ─────────────────────────────────────────────────────────────

func (s *MCPServer) engineeringGroupDef() toolGroupDef {
	return toolGroupDef{
		name:          "engineering",
		description:   "Store engineering artifacts: decisions, incidents, runbooks, or postmortems",
		requireMemory: true,
		actions: map[string]groupedAction{
			"decision":   {legacyName: "store_decision", handler: s.callStoreDecision},
			"incident":   {legacyName: "store_incident", handler: s.callStoreIncident},
			"runbook":    {legacyName: "store_runbook", handler: s.callStoreRunbook},
			"postmortem": {legacyName: "store_postmortem", handler: s.callStorePostmortem},
		},
		schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"decision", "incident", "runbook", "postmortem"},
					"description": "Type of engineering artifact to store.\n" +
						"- decision: Requires: decision. Optional: title, rationale, consequences, context, service, owner, status, tags, importance\n" +
						"- incident: Requires: summary. Optional: title, impact, root_cause, resolution, context, service, severity, tags, importance\n" +
						"- runbook: Requires: procedure. Optional: title, trigger, verification, rollback, context, service, tags, importance\n" +
						"- postmortem: Requires: summary. Optional: title, impact, root_cause, action_items, follow_up, context, service, severity, tags, importance",
				},
				"title":        map[string]any{"type": "string", "description": "Short title"},
				"decision":     map[string]any{"type": "string", "description": "What was decided (decision)"},
				"rationale":    map[string]any{"type": "string", "description": "Why decided (decision)"},
				"consequences": map[string]any{"type": "string", "description": "Expected impact (decision)"},
				"summary":      map[string]any{"type": "string", "description": "Summary (incident, postmortem)"},
				"impact":       map[string]any{"type": "string", "description": "What was affected (incident, postmortem)"},
				"root_cause":   map[string]any{"type": "string", "description": "Root cause (incident, postmortem)"},
				"resolution":   map[string]any{"type": "string", "description": "How resolved (incident)"},
				"procedure":    map[string]any{"type": "string", "description": "Main procedure (runbook)"},
				"trigger":      map[string]any{"type": "string", "description": "When to use (runbook)"},
				"verification": map[string]any{"type": "string", "description": "How to verify (runbook)"},
				"rollback":     map[string]any{"type": "string", "description": "Rollback steps (runbook)"},
				"action_items": map[string]any{"type": "string", "description": "Follow-up actions (postmortem)"},
				"follow_up":    map[string]any{"type": "string", "description": "Next steps (postmortem)"},
				"context":      map[string]any{"type": "string", "description": "Project, task, or service context"},
				"service":      map[string]any{"type": "string", "description": "Service or component name"},
				"owner":        map[string]any{"type": "string", "description": "Decision owner (decision)"},
				"status":       map[string]any{"type": "string", "description": "Status e.g. proposed/accepted (decision)"},
				"severity":     map[string]any{"type": "string", "description": "Severity e.g. sev1/sev2 (incident, postmortem)"},
				"tags":         map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Additional tags"},
				"importance":   map[string]any{"type": "number", "minimum": 0, "maximum": 1, "default": 0.85, "description": "Importance 0.0-1.0"},
			},
			"required": []string{"action"},
		},
	}
}

// ─── search ─────────────────────────────────────────────────────────────────

func (s *MCPServer) searchGroupDef() toolGroupDef {
	return toolGroupDef{
		name:          "search",
		description:   "Search across documents, runbooks, incidents, and project context",
		requireHybrid: true,
		actions: map[string]groupedAction{
			"semantic":        {legacyName: "semantic_search", handler: s.callSemanticSearch},
			"runbooks":        {legacyName: "search_runbooks", handler: s.callSearchRunbooks},
			"incidents":       {legacyName: "recall_similar_incidents", handler: s.callRecallSimilarIncidents},
			"project_context": {legacyName: "summarize_project_context", handler: s.callSummarizeProjectContext},
		},
		schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"semantic", "runbooks", "incidents", "project_context"},
					"description": "Search scope.\n" +
						"- semantic: hybrid search across indexed docs. Requires: query. Optional: source_type, debug, limit\n" +
						"- runbooks: search runbook memories and docs. Requires: query. Optional: context, service, tags, limit, debug\n" +
						"- incidents: recall similar incidents/postmortems. Requires: query. Optional: context, service, tags, limit, debug\n" +
						"- project_context: summarize recent decisions, runbooks, incidents. Optional: context, focus, service, limit",
				},
				"query":       map[string]any{"type": "string", "description": "Search query (semantic, runbooks, incidents)"},
				"focus":       map[string]any{"type": "string", "description": "Focus query for narrowing (project_context)"},
				"source_type": map[string]any{"type": "string", "description": "Source type filter: docs, adr, rfc, changelog, runbook, postmortem, ci_config, helm, terraform, k8s (semantic)"},
				"context":     map[string]any{"type": "string", "description": "Project or task context"},
				"service":     map[string]any{"type": "string", "description": "Service or component name"},
				"tags":        map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Tag filter"},
				"debug":       map[string]any{"type": "boolean", "default": false, "description": "Include score breakdown"},
				"limit":       map[string]any{"type": "integer", "minimum": 1, "maximum": 50, "default": 10, "description": "Max results"},
			},
			"required": []string{"action"},
		},
	}
}

// ─── session ────────────────────────────────────────────────────────────────

func (s *MCPServer) sessionGroupDef() toolGroupDef {
	return toolGroupDef{
		name:          "session",
		description:   "Session lifecycle: close/analyze sessions, review changes, accept changes, resolve review items",
		requireMemory: true,
		actions: map[string]groupedAction{
			"close":         {legacyName: "close_session", handler: s.callCloseSession},
			"review":        {legacyName: "review_session_changes", handler: s.callReviewSessionChanges},
			"accept":        {legacyName: "accept_session_changes", handler: s.callAcceptSessionChanges},
			"resolve_item":  {legacyName: "resolve_review_item", handler: s.callResolveReviewItem},
			"resolve_queue": {legacyName: "resolve_review_queue", handler: s.callResolveReviewQueue},
		},
		schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action": map[string]any{
					"type": "string",
					"enum": []string{"close", "review", "accept", "resolve_item", "resolve_queue"},
					"description": "Session operation.\n" +
						"- close: analyze finished session. Requires: summary. Optional: mode, context, service, started_at, ended_at, tags, metadata, dry_run, save_raw, auto_apply_low_risk, format\n" +
						"- review: review session changes. Requires: summary. Optional: mode, context, service, started_at, ended_at, tags, metadata, format\n" +
						"- accept: persist and auto-apply low-risk changes. Requires: summary. Optional: mode, context, service, started_at, ended_at, tags, metadata, format\n" +
						"- resolve_item: resolve one review item. Requires: id. Optional: resolution, note, owner, format\n" +
						"- resolve_queue: bulk-resolve review items. Optional: ids, resolution, note, owner, context, service, tags, limit, dry_run, format",
				},
				"summary":    map[string]any{"type": "string", "description": "Raw session summary text (close, review, accept)"},
				"mode":       map[string]any{"type": "string", "enum": []string{"coding", "incident", "migration", "research", "cleanup"}, "default": "coding", "description": "Session mode (close, review, accept)"},
				"context":    map[string]any{"type": "string", "description": "Project, task, or workflow context"},
				"service":    map[string]any{"type": "string", "description": "Service or component name"},
				"started_at": map[string]any{"type": "string", "description": "RFC3339 session start (close, review, accept)"},
				"ended_at":   map[string]any{"type": "string", "description": "RFC3339 session end (close, review, accept)"},
				"tags":       map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Tags"},
				"metadata": map[string]any{
					"type":                 "object",
					"additionalProperties": map[string]any{"type": "string"},
					"description":          "String metadata (close, review, accept)",
				},
				"dry_run":             map[string]any{"type": "boolean", "default": true, "description": "Plan without saving (close), preview without changing (resolve_queue)"},
				"save_raw":            map[string]any{"type": "boolean", "default": false, "description": "Persist raw summary (close)"},
				"auto_apply_low_risk": map[string]any{"type": "boolean", "default": false, "description": "Auto-apply low-risk actions (close)"},
				"format":              map[string]any{"type": "string", "enum": []string{"text", "json"}, "default": "text", "description": "Output format"},
				"id":                  map[string]any{"type": "string", "description": "Review item ID (resolve_item)"},
				"ids":                 map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Review item IDs (resolve_queue)"},
				"resolution":          map[string]any{"type": "string", "enum": []string{"resolved", "dismissed", "deferred"}, "default": "resolved", "description": "Resolution type (resolve_item, resolve_queue)"},
				"note":                map[string]any{"type": "string", "description": "Resolution note (resolve_item, resolve_queue)"},
				"owner":               map[string]any{"type": "string", "description": "Reviewer (resolve_item, resolve_queue)"},
				"limit":               map[string]any{"type": "integer", "minimum": 1, "maximum": 100, "default": 20, "description": "Max items (resolve_queue)"},
			},
			"required": []string{"action"},
		},
	}
}

// ─── grouped tools list ─────────────────────────────────────────────────────

// groupedToolsList returns the tool list for grouped mode, filtering by backend availability.
func (s *MCPServer) groupedToolsList() []tool {
	groups := s.buildToolGroupDefs()
	tools := make([]tool, 0, len(groups)+2) // +2 for project_bank_view and index_documents

	for _, g := range groups {
		if g.requireRAG && s.ragEngine == nil {
			continue
		}
		if g.requireMemory && s.memoryStore == nil {
			continue
		}
		if g.requireHybrid && s.memoryStore == nil && s.ragEngine == nil {
			continue
		}
		tools = append(tools, tool{
			Name:        g.name,
			Description: g.description,
			InputSchema: g.schema,
		})
	}

	// Standalone tools that don't fit into groups
	if s.ragEngine != nil {
		tools = append(tools, tool{
			Name:        "index_documents",
			Description: "Re-index documents for RAG search",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		})
	}

	if s.memoryStore != nil {
		tools = append(tools, tool{
			Name:        "project_bank_view",
			Description: "Show a structured project bank view for canonical knowledge, decisions, runbooks, incidents, caveats, migrations, or the review queue",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"view": map[string]any{
						"type":        "string",
						"enum":        []string{"canonical_overview", "overview", "decisions", "runbooks", "incidents", "caveats", "migrations", "review_queue"},
						"default":     "canonical_overview",
						"description": "Which project bank view to render",
					},
					"context": map[string]any{"type": "string", "description": "Optional project or task context"},
					"service": map[string]any{"type": "string", "description": "Optional service or component filter"},
					"status":  map[string]any{"type": "string", "description": "Optional lifecycle or status filter"},
					"owner":   map[string]any{"type": "string", "description": "Optional owner filter"},
					"tags":    map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional tag filter"},
					"limit":   map[string]any{"type": "integer", "minimum": 1, "maximum": 50, "default": 10, "description": "Max items per section"},
					"format":  map[string]any{"type": "string", "enum": []string{"text", "json"}, "default": "text", "description": "Output format"},
				},
			},
		})
	}

	return tools
}
