// Package rag provides RAG (Retrieval-Augmented Generation) with document indexing and semantic search.
package rag

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ipiton/agent-memory-mcp/internal/config"
	"github.com/ipiton/agent-memory-mcp/internal/embedder"
	"github.com/ipiton/agent-memory-mcp/internal/logger"
	"github.com/ipiton/agent-memory-mcp/internal/vectorstore"
	"go.uber.org/zap"
)

// SearchResult represents a single document match from a RAG search query.
type SearchResult struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	Path         string    `json:"path"`
	Type         string    `json:"type"`
	Category     string    `json:"category"`
	TaskSlug     string    `json:"task_slug,omitempty"`
	TaskPhase    string    `json:"task_phase,omitempty"`
	Score        float64   `json:"score"`
	Snippet      string    `json:"snippet"`
	LastModified time.Time `json:"last_modified"`
}

// SearchResponse holds the results and metadata for a search query.
type SearchResponse struct {
	Query      string         `json:"query"`
	Results    []SearchResult `json:"results"`
	TotalFound int            `json:"total_found"`
	SearchTime int64          `json:"search_time_ms"`
}

// Engine provides document indexing and semantic vector search over a repository.
type Engine struct {
	config         config.Config
	repoRoot       string
	logger         *zap.Logger
	docService     *documentService
	vecService     *vectorService
	mu             sync.Mutex
	indexing       bool
	lastIndexCheck time.Time
	stopWatcher    chan struct{}
	stopOnce       sync.Once
}

type docServiceConfig struct {
	IndexDirs    []string
	RepoRoot     string
	ChunkSize    int
	ChunkOverlap int
}

type vecServiceConfig struct {
	IndexPath  string
	Embedder   *embedder.Embedder
	MaxResults int
}

type document struct {
	ID           string
	Content      string
	Title        string
	Path         string
	Type         string
	Category     string
	TaskSlug     string
	TaskPhase    string
	LastModified time.Time
	FileHash     string
}

type searchQuery struct {
	Query   string
	Limit   int
	Filters map[string]interface{}
}

type indexResult struct {
	SuccessIDs []string
	FailedIDs  []string
	Errors     []error
}

// NewEngine creates a new Engine with the given configuration and optional file logger.
func NewEngine(cfg config.Config, fileLogger *logger.FileLogger) *Engine {
	var zapLogger *zap.Logger
	if fileLogger != nil {
		zapLogger = fileLogger.Logger
	} else {
		zapConfig := zap.NewProductionConfig()
		zapConfig.Level = zap.NewAtomicLevelAt(zap.FatalLevel)
		zapConfig.OutputPaths = []string{"/dev/null"}
		zapLogger, _ = zapConfig.Build()
	}

	repoRoot := cfg.RootPath

	indexDirs := cfg.IndexDirs
	if len(indexDirs) == 0 {
		indexDirs = []string{"docs"}
	}
	if cfg.ChangelogPath != "" {
		indexDirs = append(indexDirs, cfg.ChangelogPath)
	} else {
		indexDirs = append(indexDirs, "CHANGELOG.md")
	}

	dsCfg := docServiceConfig{
		IndexDirs:    indexDirs,
		RepoRoot:     repoRoot,
		ChunkSize:    cfg.ChunkSize,
		ChunkOverlap: cfg.ChunkOverlap,
	}
	if dsCfg.ChunkSize == 0 {
		dsCfg.ChunkSize = 2000
	}
	if dsCfg.ChunkOverlap == 0 {
		dsCfg.ChunkOverlap = 200
	}

	docSvc := newDocumentService(dsCfg, zapLogger)

	emb, err := embedder.New(embedder.Config{
		JinaToken:     cfg.JinaAPIKey,
		OpenAIToken:   cfg.OpenAIAPIKey,
		OpenAIBaseURL: cfg.OpenAIBaseURL,
		OpenAIModel:   cfg.OpenAIModel,
		OllamaBaseURL: cfg.OllamaBaseURL,
		Dimension:     cfg.EmbeddingDimension,
		MaxRetries:    2,
		Timeout:       10 * time.Second,
	}, zapLogger)
	if err != nil {
		if fileLogger != nil {
			fileLogger.Error("Failed to initialize embedder",
				zap.Error(err),
				zap.String("jina_api_key_set", config.BoolToString(cfg.JinaAPIKey != "")),
				zap.String("ollama_url", cfg.OllamaBaseURL),
			)
		}
		return nil
	}

	vecSvc, err := newVectorService(vecServiceConfig{
		IndexPath:  cfg.RAGIndexPath,
		Embedder:   emb,
		MaxResults: cfg.RAGMaxResults,
	}, zapLogger)
	if err != nil {
		if fileLogger != nil {
			fileLogger.Error("Failed to initialize vector service",
				zap.Error(err),
				zap.String("rag_index_path", cfg.RAGIndexPath),
			)
		}
		return nil
	}

	engine := &Engine{
		config:      cfg,
		repoRoot:    repoRoot,
		logger:      zapLogger,
		docService:  docSvc,
		vecService:  vecSvc,
		indexing:    false,
		stopWatcher: make(chan struct{}),
	}

	autoIndex := config.EnvOrDefault("MCP_RAG_AUTO_INDEX", "true")
	if autoIndex == "true" {
		if fileLogger != nil {
			fileLogger.Info("Starting auto-indexing check")
		}
		go engine.autoIndexIfNeeded()

		watcherEnabled := config.EnvOrDefault("MCP_RAG_FILE_WATCHER", "true")
		if watcherEnabled == "true" {
			if fileLogger != nil {
				fileLogger.Info("Starting file watcher for auto-reindexing")
			}
			go engine.startFileWatcher()
		}
	}

	if fileLogger != nil {
		fileLogger.Info("RAG engine created successfully",
			zap.String("repo_root", repoRoot),
			zap.Strings("index_dirs", indexDirs),
			zap.String("rag_index_path", cfg.RAGIndexPath),
		)
	}

	return engine
}

// Search performs a semantic search query with optional document type and category filters.
func (re *Engine) Search(query string, limit int, docType, category string) (*SearchResponse, error) {
	if re == nil || re.vecService == nil {
		return nil, fmt.Errorf("RAG engine not available")
	}

	if limit <= 0 {
		limit = re.config.RAGMaxResults
	}
	if limit > re.config.RAGMaxResults {
		limit = re.config.RAGMaxResults
	}

	filters := make(map[string]interface{})
	if docType != "" && docType != "all" {
		filters["type"] = docType
	}
	if category != "" {
		filters["category"] = category
	}

	result, err := re.vecService.search(searchQuery{
		Query:   query,
		Limit:   limit,
		Filters: filters,
	})
	if err != nil {
		re.logger.Error("Vector search failed",
			zap.Error(err),
			zap.String("query", query),
			zap.Int("limit", limit),
		)
		return nil, fmt.Errorf("search failed: %w", err)
	}

	return &SearchResponse{
		Query:      result.Query,
		Results:    result.Results,
		TotalFound: result.TotalFound,
		SearchTime: result.SearchTime,
	}, nil
}

// IndexDocuments performs incremental indexing of documents in configured directories.
func (re *Engine) IndexDocuments() error {
	if re == nil || re.docService == nil || re.vecService == nil {
		return fmt.Errorf("RAG engine not available")
	}

	startTime := time.Now()

	const embeddingModel = "jina-v3-bge-m3-1024d"
	const chunkerVersion = "char-v1"

	store := re.vecService.store.(*vectorstore.SQLiteStore)
	oldModel, _ := store.GetMetadata("embedding_model")
	oldChunker, _ := store.GetMetadata("chunker_version")

	needsRebuild := false
	if oldModel != "" && oldModel != embeddingModel {
		re.logger.Warn("Embedding model changed - full rebuild required")
		needsRebuild = true
	}
	if oldChunker != "" && oldChunker != chunkerVersion {
		re.logger.Warn("Chunker version changed - full rebuild required")
		needsRebuild = true
	}

	store.SetMetadata("embedding_model", embeddingModel)
	store.SetMetadata("chunker_version", chunkerVersion)

	allDocs, err := re.docService.collectDocuments()
	if err != nil {
		return fmt.Errorf("failed to collect documents: %w", err)
	}

	if len(allDocs) == 0 {
		return fmt.Errorf("no documents found to index")
	}

	indexedFiles, err := store.GetAllIndexedFiles()
	if err != nil {
		indexedFiles = make(map[string]*vectorstore.IndexedFileInfo)
	}

	toAdd, toRemove := re.calculateIndexChanges(allDocs, indexedFiles, needsRebuild)

	re.logger.Info("Indexing analysis",
		zap.Int("total_documents", len(allDocs)),
		zap.Int("indexed_files", len(indexedFiles)),
		zap.Int("documents_to_add", len(toAdd)),
		zap.Int("files_to_remove", len(toRemove)),
		zap.Bool("force_rebuild", needsRebuild),
	)

	if len(toRemove) > 0 {
		re.logger.Info("Removing deleted documents", zap.Int("count", len(toRemove)))
		for _, filePath := range toRemove {
			if err := re.vecService.removeDocument(filePath); err != nil {
				re.logger.Warn("Failed to remove document", zap.String("path", filePath), zap.Error(err))
			}
			store.DeleteIndexedFile(filePath)
		}
	}

	if len(toAdd) > 0 {
		re.logger.Info("Starting to index documents", zap.Int("count", len(toAdd)))

		chunkCountsByFile := make(map[string]int)
		for _, doc := range toAdd {
			chunkCountsByFile[doc.Path]++
		}

		batchSize := 10
		totalBatches := (len(toAdd) + batchSize - 1) / batchSize

		for batchNum := 0; batchNum < totalBatches; batchNum++ {
			start := batchNum * batchSize
			end := start + batchSize
			if end > len(toAdd) {
				end = len(toAdd)
			}
			batch := toAdd[start:end]

			re.logger.Info("Processing batch",
				zap.Int("batch", batchNum+1),
				zap.Int("total", totalBatches))

			result, err := re.vecService.indexDocuments(batch)
			if err != nil {
				return fmt.Errorf("failed to index: %w", err)
			}

			updatedFiles := make(map[string]bool)

			for _, successID := range result.SuccessIDs {
				for _, doc := range batch {
					if doc.ID == successID && !updatedFiles[doc.Path] {
						fileHash := doc.FileHash
						if fileHash == "" {
							fileHash = calculateFileHash(doc.Content)
						}

						chunkCount := chunkCountsByFile[doc.Path]
						if chunkCount == 0 {
							chunkCount = 1
						}

						store.SetIndexedFile(&vectorstore.IndexedFileInfo{
							FilePath:   doc.Path,
							Hash:       fileHash,
							ModTime:    doc.LastModified,
							Size:       int64(len(doc.Content)),
							ChunkCount: chunkCount,
						})
						updatedFiles[doc.Path] = true
						break
					}
				}
			}

			re.logger.Info("Batch indexed",
				zap.Int("success", len(result.SuccessIDs)),
				zap.Int("failed", len(result.FailedIDs)))

			if batchNum < totalBatches-1 {
				time.Sleep(500 * time.Millisecond)
			}
		}
	}

	store.SetMetadata("last_indexed", time.Now().Format(time.RFC3339))

	duration := time.Since(startTime)
	re.logger.Info("Indexing completed", zap.Duration("duration", duration))

	return nil
}

func (re *Engine) calculateIndexChanges(currentDocs []document, indexedFiles map[string]*vectorstore.IndexedFileInfo, forceRebuild bool) (toAdd []document, toRemove []string) {
	currentByPath := make(map[string]document)
	for _, doc := range currentDocs {
		if _, exists := currentByPath[doc.Path]; !exists {
			currentByPath[doc.Path] = doc
		}
	}

	for filePath := range indexedFiles {
		if _, exists := currentByPath[filePath]; !exists {
			toRemove = append(toRemove, filePath)
		}
	}

	addedFiles := make(map[string]bool)

	for _, doc := range currentDocs {
		if addedFiles[doc.Path] {
			continue
		}

		indexed, exists := indexedFiles[doc.Path]
		if forceRebuild || !exists {
			for _, d := range currentDocs {
				if d.Path == doc.Path {
					toAdd = append(toAdd, d)
				}
			}
			addedFiles[doc.Path] = true
			continue
		}

		fileHash := doc.FileHash
		if fileHash == "" {
			fileHash = calculateFileHash(doc.Content)
		}

		if indexed.Hash != fileHash || indexed.ModTime.Before(doc.LastModified) {
			for _, d := range currentDocs {
				if d.Path == doc.Path {
					toAdd = append(toAdd, d)
				}
			}
			addedFiles[doc.Path] = true
		}
	}

	return toAdd, toRemove
}

func (re *Engine) autoIndexIfNeeded() {
	time.Sleep(2 * time.Second)

	if re.needsIndexing() {
		re.logger.Info("Index needs updating, starting auto-indexing")
		re.indexWithLock("initial_startup")
	} else {
		re.logger.Info("Index is up to date, skipping auto-indexing")
	}
}

func (re *Engine) startFileWatcher() {
	intervalStr := config.EnvOrDefault("MCP_RAG_WATCH_INTERVAL", "5m")
	interval, err := time.ParseDuration(intervalStr)
	if err != nil {
		interval = 5 * time.Minute
	}

	debounceStr := config.EnvOrDefault("MCP_RAG_DEBOUNCE", "30s")
	debounceDuration, err := time.ParseDuration(debounceStr)
	if err != nil {
		debounceDuration = 30 * time.Second
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	var pendingMu sync.Mutex
	var pendingReindex bool
	var debounceTimer *time.Timer

	for {
		select {
		case <-re.stopWatcher:
			if debounceTimer != nil {
				debounceTimer.Stop()
			}
			return

		case <-ticker.C:
			if re.needsIndexing() {
				pendingMu.Lock()
				pendingReindex = true
				pendingMu.Unlock()

				if debounceTimer != nil {
					debounceTimer.Stop()
				}

				debounceTimer = time.AfterFunc(debounceDuration, func() {
					pendingMu.Lock()
					shouldIndex := pendingReindex
					pendingReindex = false
					pendingMu.Unlock()
					if shouldIndex {
						re.indexWithLock("file_watcher")
					}
				})
			}
		}
	}
}

func (re *Engine) indexWithLock(trigger string) {
	re.mu.Lock()
	if re.indexing {
		re.mu.Unlock()
		re.logger.Debug("Indexing already in progress, skipping", zap.String("trigger", trigger))
		return
	}
	re.indexing = true
	re.mu.Unlock()

	defer func() {
		re.mu.Lock()
		re.indexing = false
		re.mu.Unlock()
	}()

	re.logger.Info("Starting indexing", zap.String("trigger", trigger))
	err := re.IndexDocuments()
	if err != nil {
		re.logger.Error("Indexing failed", zap.Error(err), zap.String("trigger", trigger))
	} else {
		re.logger.Info("Indexing completed successfully", zap.String("trigger", trigger))
		re.mu.Lock()
		re.lastIndexCheck = time.Now()
		re.mu.Unlock()
	}
}

// Stop gracefully stops the Engine, terminating the file watcher.
func (re *Engine) Stop() {
	if re != nil {
		re.stopOnce.Do(func() {
			close(re.stopWatcher)
		})
	}
}

func (re *Engine) needsIndexing() bool {
	store := re.vecService.store.(*vectorstore.SQLiteStore)

	indexedFiles, err := store.GetAllIndexedFiles()
	if err != nil {
		re.logger.Warn("Failed to get indexed files from store, indexing required",
			zap.Error(err),
			zap.String("store_type", "SQLiteStore"),
		)
		return true
	}
	if len(indexedFiles) == 0 {
		re.logger.Info("No indexed files found, indexing required",
			zap.String("repo_root", re.repoRoot),
		)
		return true
	}

	currentDocs, err := re.docService.collectDocuments()
	if err != nil {
		re.logger.Warn("Failed to check filesystem, indexing required",
			zap.Error(err),
			zap.String("repo_root", re.repoRoot),
		)
		return true
	}

	toAdd, toRemove := re.calculateIndexChanges(currentDocs, indexedFiles, false)

	needsIndex := len(toAdd) > 0 || len(toRemove) > 0
	if needsIndex {
		re.logger.Info("Index needs updating",
			zap.Int("indexed", len(indexedFiles)),
			zap.Int("current", len(currentDocs)),
			zap.Int("to_add", len(toAdd)),
			zap.Int("to_remove", len(toRemove)),
			zap.String("repo_root", re.repoRoot),
		)
	} else {
		re.logger.Debug("Index is up to date",
			zap.Int("indexed", len(indexedFiles)),
			zap.Int("current", len(currentDocs)),
		)
	}

	return needsIndex
}

func calculateFileHash(content string) string {
	hasher := md5.New()
	hasher.Write([]byte(content))
	return hex.EncodeToString(hasher.Sum(nil))
}

// === Document Service ===

type documentService struct {
	config docServiceConfig
	logger *zap.Logger
}

func newDocumentService(cfg docServiceConfig, logger *zap.Logger) *documentService {
	return &documentService{config: cfg, logger: logger}
}

func (ds *documentService) collectDocuments() ([]document, error) {
	var allDocs []document

	for _, dir := range ds.config.IndexDirs {
		fullPath := filepath.Join(ds.config.RepoRoot, dir)

		info, err := os.Stat(fullPath)
		if err != nil {
			ds.logger.Warn("Path not found", zap.String("path", fullPath))
			continue
		}

		if !info.IsDir() {
			if strings.HasSuffix(fullPath, ".md") {
				docs, err := ds.processFile(fullPath)
				if err != nil {
					ds.logger.Warn("Failed to process file", zap.String("path", fullPath), zap.Error(err))
				} else {
					allDocs = append(allDocs, docs...)
				}
			}
			continue
		}

		err = filepath.WalkDir(fullPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && strings.HasSuffix(path, ".md") {
				docs, err := ds.processFile(path)
				if err != nil {
					ds.logger.Warn("Failed to process file", zap.String("path", path), zap.Error(err))
					return nil
				}
				allDocs = append(allDocs, docs...)
			}
			return nil
		})
		if err != nil {
			ds.logger.Error("Failed to walk directory", zap.String("path", fullPath), zap.Error(err))
		}
	}

	return allDocs, nil
}

func (ds *documentService) getDocType(path string) string {
	relPath := strings.TrimPrefix(path, ds.config.RepoRoot)
	relPath = strings.TrimPrefix(relPath, "/")

	switch {
	case strings.HasPrefix(relPath, "tasks/"):
		return "tasks"
	case strings.HasPrefix(relPath, "memory-bank/"):
		return "memory"
	case strings.HasPrefix(relPath, "docs/"):
		return "docs"
	case strings.Contains(relPath, "CHANGELOG"):
		return "changelog"
	default:
		return "docs"
	}
}

func (ds *documentService) processFile(path string) ([]document, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	docType := ds.getDocType(path)
	relPath := strings.TrimPrefix(path, ds.config.RepoRoot)
	relPath = strings.TrimPrefix(relPath, "/")

	title := ds.extractTitle(string(content), filepath.Base(path))
	cleanContent := ds.removeFrontmatter(string(content))

	fileHash := calculateFileHash(cleanContent)
	modTime := ds.getFileModTime(path)

	var taskSlug, taskPhase string
	if docType == "tasks" {
		taskSlug, taskPhase = ds.extractTaskInfo(relPath)
	}

	chunks := ds.splitIntoChunks(cleanContent)

	var docs []document
	for i, chunk := range chunks {
		docs = append(docs, document{
			ID:           fmt.Sprintf("%s-%d", relPath, i),
			Content:      chunk,
			Title:        title,
			Path:         relPath,
			Type:         docType,
			Category:     ds.extractCategory(relPath),
			TaskSlug:     taskSlug,
			TaskPhase:    taskPhase,
			LastModified: modTime,
			FileHash:     fileHash,
		})
	}

	return docs, nil
}

func (ds *documentService) extractTitle(content, filename string) string {
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimPrefix(line, "# ")
		}
	}
	return strings.TrimSuffix(filename, filepath.Ext(filename))
}

func (ds *documentService) removeFrontmatter(content string) string {
	lines := strings.Split(content, "\n")
	if len(lines) > 1 && strings.TrimSpace(lines[0]) == "---" {
		for i, line := range lines[1:] {
			if strings.TrimSpace(line) == "---" {
				return strings.Join(lines[i+2:], "\n")
			}
		}
	}
	return content
}

func (ds *documentService) splitIntoChunks(content string) []string {
	chunkSize := ds.config.ChunkSize
	overlap := ds.config.ChunkOverlap

	if len(content) <= chunkSize {
		return []string{content}
	}

	var chunks []string
	contentLen := len(content)
	step := chunkSize - overlap

	for start := 0; start < contentLen; start += step {
		end := start + chunkSize
		if end > contentLen {
			end = contentLen
		}

		if end < contentLen {
			breakPoint := end
			for i := end; i > end-100 && i > start; i-- {
				if content[i] == ' ' || content[i] == '\n' {
					breakPoint = i
					break
				}
			}
			end = breakPoint
		}

		chunk := strings.TrimSpace(content[start:end])
		if len(chunk) > 0 {
			chunks = append(chunks, chunk)
		}

		if end >= contentLen {
			break
		}
	}

	return chunks
}

func (ds *documentService) extractCategory(path string) string {
	parts := strings.Split(path, string(filepath.Separator))
	if len(parts) > 0 {
		category := strings.TrimSuffix(parts[0], filepath.Ext(parts[0]))
		return strings.ToLower(category)
	}
	return "general"
}

func (ds *documentService) extractTaskInfo(path string) (string, string) {
	parts := strings.Split(path, string(filepath.Separator))

	var taskSlug string
	for _, part := range parts {
		if strings.Contains(part, "_") && !strings.Contains(part, ".") {
			taskSlug = strings.Split(part, "_")[0]
			break
		}
	}

	filename := filepath.Base(path)
	var phase string
	switch {
	case strings.Contains(filename, "requirements"):
		phase = "requirements"
	case strings.Contains(filename, "design"):
		phase = "design"
	case strings.Contains(filename, "tasks"):
		phase = "tasks"
	}

	return taskSlug, phase
}

func (ds *documentService) getFileModTime(path string) time.Time {
	info, err := os.Stat(path)
	if err != nil {
		return time.Now()
	}
	return info.ModTime()
}

// === Vector Service ===

type vectorService struct {
	config vecServiceConfig
	logger *zap.Logger
	store  vectorstore.Store
}

func newVectorService(cfg vecServiceConfig, logger *zap.Logger) (*vectorService, error) {
	if err := os.MkdirAll(cfg.IndexPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create index directory: %w", err)
	}

	dbPath := filepath.Join(cfg.IndexPath, "vectors.db")
	logger.Info("Using SQLite vector store", zap.String("db_path", dbPath))

	store, err := vectorstore.NewSQLiteStore(dbPath, cfg.Embedder.Dimension, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}

	return &vectorService{
		config: cfg,
		logger: logger,
		store:  store,
	}, nil
}

func (vs *vectorService) search(query searchQuery) (*SearchResponse, error) {
	startTime := time.Now()

	queryEmbedding, err := vs.config.Embedder.EmbedQuery(query.Query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	var filters map[string]string
	if query.Filters != nil {
		filters = make(map[string]string)
		for k, v := range query.Filters {
			if str, ok := v.(string); ok && str != "" {
				filters[k] = str
			}
		}
	}

	results, err := vs.store.Search(queryEmbedding, filters, query.Limit)
	if err != nil {
		return nil, fmt.Errorf("failed to search: %w", err)
	}

	var searchResults []SearchResult
	for _, result := range results {
		snippet := result.Content
		if len(snippet) > 200 {
			snippet = snippet[:200] + "..."
		}

		searchResults = append(searchResults, SearchResult{
			ID:           result.ID,
			Title:        result.Title,
			Path:         result.Path,
			Type:         result.Type,
			Category:     result.Category,
			TaskSlug:     result.TaskSlug,
			TaskPhase:    result.TaskPhase,
			Score:        result.Score,
			Snippet:      snippet,
			LastModified: result.LastModified,
		})
	}

	return &SearchResponse{
		Query:      query.Query,
		Results:    searchResults,
		TotalFound: len(searchResults),
		SearchTime: time.Since(startTime).Milliseconds(),
	}, nil
}

func (vs *vectorService) indexDocuments(docs []document) (*indexResult, error) {
	result := &indexResult{
		SuccessIDs: make([]string, 0, len(docs)),
		FailedIDs:  make([]string, 0),
		Errors:     make([]error, 0),
	}

	var chunks []vectorstore.Chunk
	for i, doc := range docs {
		if i > 0 {
			time.Sleep(200 * time.Millisecond)
		}

		embedding, err := vs.config.Embedder.Embed(doc.Content)
		if err != nil {
			vs.logger.Warn("Failed to embed document", zap.String("id", doc.ID), zap.Error(err))
			result.FailedIDs = append(result.FailedIDs, doc.ID)
			result.Errors = append(result.Errors, err)
			continue
		}

		chunks = append(chunks, vectorstore.Chunk{
			ID:           doc.ID,
			DocPath:      doc.Path,
			Content:      doc.Content,
			Title:        doc.Title,
			Path:         doc.Path,
			Type:         doc.Type,
			Category:     doc.Category,
			TaskSlug:     doc.TaskSlug,
			TaskPhase:    doc.TaskPhase,
			LastModified: doc.LastModified,
			Embedding:    embedding,
		})

		result.SuccessIDs = append(result.SuccessIDs, doc.ID)
	}

	if len(chunks) > 0 {
		if err := vs.store.Upsert(chunks); err != nil {
			vs.logger.Error("Failed to upsert chunks", zap.Error(err))
			return result, err
		}
	}

	return result, nil
}

func (vs *vectorService) removeDocument(path string) error {
	return vs.store.DeleteByDocPath(path)
}

func (vs *vectorService) count() int {
	return vs.store.Count()
}
