// Package embedder provides text embedding with Jina AI, OpenAI, and Ollama providers.
package embedder

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

// DefaultDimension is the default vector dimension for embedding providers.
// Jina v3 and bge-m3 produce 1024 natively; OpenAI supports Matryoshka truncation to any size.
// Changing requires re-indexing.
const DefaultDimension = 1024

// Config holds provider credentials and tuning parameters for the Embedder.
type Config struct {
	JinaToken     string
	OpenAIToken   string
	OpenAIBaseURL string        // OpenAI-compatible base URL (default: https://api.openai.com/v1)
	OpenAIModel   string        // Embedding model (default: text-embedding-3-small)
	OllamaBaseURL string
	Dimension     int           // Required embedding dimension (default: 1024)
	MaxRetries    int
	Timeout       time.Duration
}

// Embedder generates vector embeddings using Jina AI as primary with OpenAI and Ollama fallback.
type Embedder struct {
	config            Config
	logger            *zap.Logger
	client            *http.Client
	Dimension         int        // Embedding dimension
	jinaDisabled      bool       // Flag to disable Jina after auth errors
	jinaDisabledUntil time.Time  // Time when Jina can be retried again (for auth errors)
	jinaErrorCount    int        // Count of consecutive Jina errors
	jinaDisabledMu    sync.Mutex // Mutex for jinaDisabled flag
}

// New creates a new Embedder with the given configuration and logger.
func New(config Config, logger *zap.Logger) (*Embedder, error) {
	if logger == nil {
		logger = zap.NewNop()
	}
	if config.OllamaBaseURL == "" {
		config.OllamaBaseURL = "http://localhost:11434"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 2
	}
	if config.Dimension == 0 {
		config.Dimension = DefaultDimension
	}

	return &Embedder{
		config: config,
		logger: logger,
		client: &http.Client{
			Timeout: config.Timeout,
		},
		Dimension:         config.Dimension,
		jinaDisabled:      false,
		jinaDisabledUntil: time.Time{},
		jinaErrorCount:    0,
		jinaDisabledMu:    sync.Mutex{},
	}, nil
}

// Embed generates a vector embedding for the given text, optimized for document passages.
func (e *Embedder) Embed(text string) ([]float32, error) {
	return e.EmbedWithTask(text, "retrieval.passage")
}

// EmbedQuery generates a vector embedding for a search query, optimized for retrieval.
func (e *Embedder) EmbedQuery(text string) ([]float32, error) {
	return e.EmbedWithTask(text, "retrieval.query")
}

// EmbedWithTask generates a vector embedding with the specified Jina task type.
func (e *Embedder) EmbedWithTask(text string, task string) ([]float32, error) {
	// Safety check: ensure logger is not nil
	if e == nil {
		return nil, fmt.Errorf("embedder is nil")
	}
	if e.logger == nil {
		e.logger = zap.NewNop()
	}

	// Check if Jina is disabled (after auth errors)
	e.jinaDisabledMu.Lock()
	jinaDisabled := e.jinaDisabled
	// Check if disabled period has expired (retry after 1 hour for auth errors)
	if jinaDisabled && !e.jinaDisabledUntil.IsZero() && time.Now().After(e.jinaDisabledUntil) {
		e.jinaDisabled = false
		e.jinaDisabledUntil = time.Time{}
		e.jinaErrorCount = 0
		jinaDisabled = false
		e.jinaDisabledMu.Unlock()
		e.logger.Info("Retrying Jina AI after timeout period",
			zap.String("task", task),
		)
	} else {
		e.jinaDisabledMu.Unlock()
	}

	// tryProvider attempts an embedding call and validates dimensions.
	// Returns the embedding if successful and dimensions match, nil otherwise.
	tryProvider := func(name string, embed func() ([]float32, error)) []float32 {
		embedding, err := embed()
		if err != nil {
			e.logger.Warn("Embedding provider failed", zap.String("provider", name), zap.Error(err))
			return nil
		}
		if len(embedding) != e.Dimension {
			e.logger.Error("Embedding dimension mismatch — check model configuration",
				zap.String("provider", name),
				zap.Int("got", len(embedding)),
				zap.Int("expected", e.Dimension),
				zap.String("hint", fmt.Sprintf("The model returned %d dimensions but %d are required. Set MCP_EMBEDDING_DIMENSION=%d or use a model that supports %d-dimensional output.", len(embedding), e.Dimension, len(embedding), e.Dimension)),
			)
			return nil
		}
		return embedding
	}

	// 1. Try Jina AI first (preferred — high quality, multilingual) if not disabled
	if !jinaDisabled && e.config.JinaToken != "" {
		embedding := tryProvider("jina", func() ([]float32, error) {
			return e.embedJinaWithTask(text, task)
		})
		if embedding != nil {
			e.jinaDisabledMu.Lock()
			e.jinaErrorCount = 0
			e.jinaDisabledMu.Unlock()
			return embedding, nil
		}

		// Handle Jina-specific auth errors
		e.jinaDisabledMu.Lock()
		e.jinaErrorCount++
		errorCount := e.jinaErrorCount
		// Auto-disable Jina after repeated failures
		if errorCount >= 3 && !e.jinaDisabled {
			e.jinaDisabled = true
			e.jinaDisabledUntil = time.Now().Add(1 * time.Hour)
			e.jinaDisabledMu.Unlock()
			e.logger.Error("Jina AI disabled after repeated failures, using fallback providers",
				zap.String("hint", "Check JINA_API_KEY or remove it to skip Jina"),
			)
		} else {
			e.jinaDisabledMu.Unlock()
		}
	}

	// 2. Try OpenAI-compatible API (OpenAI, Together, Mistral, etc.)
	if e.config.OpenAIToken != "" {
		embedding := tryProvider("openai", func() ([]float32, error) {
			return e.embedOpenAI(text)
		})
		if embedding != nil {
			return embedding, nil
		}
	}

	// 3. Fallback to Ollama (local, free)
	if e.config.OllamaBaseURL != "" {
		// Try bge-m3 first
		embedding := tryProvider("ollama/bge-m3", func() ([]float32, error) {
			return e.embedOllamaModel(text, "bge-m3:latest")
		})
		if embedding != nil {
			return embedding, nil
		}

		// Try mxbai-embed-large as secondary
		embedding = tryProvider("ollama/mxbai-embed-large", func() ([]float32, error) {
			return e.embedOllamaModel(text, "mxbai-embed-large:latest")
		})
		if embedding != nil {
			return embedding, nil
		}
	}

	return nil, fmt.Errorf("all embedding providers failed: configure at least one of JINA_API_KEY, OPENAI_API_KEY, or OLLAMA_BASE_URL")
}

// BatchEmbed generates vector embeddings for multiple texts sequentially.
func (e *Embedder) BatchEmbed(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))

	for i, text := range texts {
		embedding, err := e.Embed(text)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text at index %d: %w", i, err)
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

// embedOllamaModel generates embeddings using specified Ollama model.
func (e *Embedder) embedOllamaModel(text, model string) ([]float32, error) {
	url := e.config.OllamaBaseURL + "/api/embeddings"

	payload := map[string]interface{}{
		"model":  model,
		"prompt": text,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	resp, err := e.client.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("Ollama %s request failed: %w", model, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama %s returned status %d: %s", model, resp.StatusCode, sanitizeErrorBody(body))
	}

	var ollamaResp struct {
		Embedding []float64 `json:"embedding"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode Ollama %s response: %w", model, err)
	}

	// Convert to float32
	embedding := make([]float32, len(ollamaResp.Embedding))
	for i, v := range ollamaResp.Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}

// embedOpenAI generates embeddings using OpenAI-compatible API.
// Works with OpenAI, Together AI, Mistral, Azure OpenAI, and any /v1/embeddings endpoint.
func (e *Embedder) embedOpenAI(text string) ([]float32, error) {
	model := e.config.OpenAIModel
	if model == "" {
		model = "text-embedding-3-small"
	}
	baseURL := e.config.OpenAIBaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	url := strings.TrimRight(baseURL, "/") + "/embeddings"

	payload := map[string]interface{}{
		"input":      text,
		"model":      model,
		"dimensions": e.Dimension,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+e.config.OpenAIToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API returned status %d: %s", resp.StatusCode, sanitizeErrorBody(body))
	}

	var openaiResp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		return nil, fmt.Errorf("failed to decode OpenAI response: %w", err)
	}

	if len(openaiResp.Data) == 0 {
		return nil, fmt.Errorf("OpenAI returned no embeddings")
	}

	embedding := make([]float32, len(openaiResp.Data[0].Embedding))
	for i, v := range openaiResp.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}

// embedJinaWithTask generates embeddings using Jina AI API with task type.
// Task types: "retrieval.passage" for documents, "retrieval.query" for queries.
func (e *Embedder) embedJinaWithTask(text string, task string) ([]float32, error) {
	url := "https://api.jina.ai/v1/embeddings"

	payload := map[string]interface{}{
		"input":           []string{text},
		"model":           "jina-embeddings-v3",
		"encoding_format": "float",
		"dimensions":      e.Dimension,
		"task":            task, // "retrieval.passage" or "retrieval.query"
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+e.config.JinaToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Jina AI API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Jina AI API returned status %d: %s", resp.StatusCode, sanitizeErrorBody(body))
	}

	var jinaResp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&jinaResp); err != nil {
		return nil, fmt.Errorf("failed to decode Jina AI response: %w", err)
	}

	if len(jinaResp.Data) == 0 {
		return nil, fmt.Errorf("Jina AI returned no embeddings")
	}

	// Convert to float32
	embedding := make([]float32, len(jinaResp.Data[0].Embedding))
	for i, v := range jinaResp.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}

// sanitizeErrorBody truncates API error response bodies to prevent
// accidental exposure of API keys or tokens in error messages.
func sanitizeErrorBody(body []byte) string {
	const maxLen = 200
	s := strings.TrimSpace(string(body))
	if len(s) > maxLen {
		return s[:maxLen] + "... (truncated)"
	}
	return s
}
