BINARY_NAME=agent-memory-mcp

.PHONY: build run test vet

build:
	go build -o bin/$(BINARY_NAME) ./cmd/agent-memory-mcp

run:
	go run ./cmd/agent-memory-mcp

test:
	go test ./...

vet:
	go vet ./...
