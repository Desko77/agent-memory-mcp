package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/ipiton/agent-memory-mcp/internal/config"
	"github.com/ipiton/agent-memory-mcp/internal/paths"
	"github.com/ipiton/agent-memory-mcp/internal/server"
)

func main() {
	// Backward compat: no subcommand or flags starting with "-" → serve
	if len(os.Args) < 2 || strings.HasPrefix(os.Args[1], "-") {
		runServe(os.Args[1:])
		return
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "serve":
		runServe(args)
	case "store":
		runStore(args)
	case "recall":
		runRecall(args)
	case "list":
		runList(args)
	case "delete":
		runDelete(args)
	case "search":
		runSearch(args)
	case "index":
		runIndex(args)
	case "stats":
		runStats(args)
	case "export":
		runExport(args)
	case "import":
		runImport(args)
	default:
		printUsage()
		os.Exit(1)
	}
}

func runServe(args []string) {
	// Restore os.Args so flag.Parse() in config.Load() works correctly
	os.Args = append([]string{os.Args[0]}, args...)

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	guard, err := paths.NewGuard(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to build path guard: %v\n", err)
		os.Exit(1)
	}

	srv := server.New(cfg, guard)
	defer srv.Shutdown()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if cfg.HTTPMode == "http" {
		fmt.Fprintf(os.Stderr, "Starting HTTP server on :%d\n", cfg.HTTPPort)
		if err := server.RunHTTP(ctx, srv); err != nil {
			fmt.Fprintf(os.Stderr, "http server error: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Server stopped gracefully\n")
	} else {
		if err := server.RunStdio(srv); err != nil {
			fmt.Fprintf(os.Stderr, "mcp server stopped: %v\n", err)
			os.Exit(1)
		}
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `agent-memory-mcp — MCP server and CLI for agent memory & RAG

Usage:
  agent-memory-mcp [command] [flags]

Commands:
  serve     Start MCP server (stdio/http) — default when no command given
  store     Store a memory
  recall    Semantic search in memories
  list      List memories with filters
  delete    Delete a memory by ID
  search    RAG semantic search across documents
  index     Re-index documents for RAG
  stats     Show memory statistics
  export    Export all memories to JSON (stdout)
  import    Import memories from JSON (file or stdin)

Run "agent-memory-mcp <command> -help" for details on a command.
`)
}
