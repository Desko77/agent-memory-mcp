// Package logger provides file-based structured logging for MCP diagnostics.
package logger

import (
	"os"
	"path/filepath"
	"sync"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// FileLogger provides thread-safe file-based logging for MCP diagnostics.
type FileLogger struct {
	Logger *zap.Logger
	mu     sync.Mutex
}

// New creates a new FileLogger that writes JSON-formatted logs to the given path.
func New(logPath string) (*FileLogger, error) {
	if err := os.MkdirAll(filepath.Dir(logPath), 0755); err != nil {
		return nil, err
	}

	file, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}

	encoderConfig := zap.NewProductionEncoderConfig()
	encoderConfig.TimeKey = "timestamp"
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	encoderConfig.LevelKey = "level"
	encoderConfig.MessageKey = "message"
	encoderConfig.CallerKey = "caller"
	encoderConfig.StacktraceKey = "stacktrace"

	core := zapcore.NewCore(
		zapcore.NewJSONEncoder(encoderConfig),
		zapcore.AddSync(file),
		zapcore.InfoLevel,
	)

	zapLogger := zap.New(core, zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel))

	return &FileLogger{
		Logger: zapLogger,
	}, nil
}

// Info logs a message at info level.
func (fl *FileLogger) Info(msg string, fields ...zap.Field) {
	if fl == nil || fl.Logger == nil {
		return
	}
	fl.mu.Lock()
	defer fl.mu.Unlock()
	fl.Logger.Info(msg, fields...)
}

// Warn logs a message at warn level.
func (fl *FileLogger) Warn(msg string, fields ...zap.Field) {
	if fl == nil || fl.Logger == nil {
		return
	}
	fl.mu.Lock()
	defer fl.mu.Unlock()
	fl.Logger.Warn(msg, fields...)
}

// Error logs a message at error level.
func (fl *FileLogger) Error(msg string, fields ...zap.Field) {
	if fl == nil || fl.Logger == nil {
		return
	}
	fl.mu.Lock()
	defer fl.mu.Unlock()
	fl.Logger.Error(msg, fields...)
}

// Debug logs a message at debug level.
func (fl *FileLogger) Debug(msg string, fields ...zap.Field) {
	if fl == nil || fl.Logger == nil {
		return
	}
	fl.mu.Lock()
	defer fl.mu.Unlock()
	fl.Logger.Debug(msg, fields...)
}

// Sync flushes any buffered log entries to disk.
func (fl *FileLogger) Sync() error {
	if fl == nil || fl.Logger == nil {
		return nil
	}
	fl.mu.Lock()
	defer fl.mu.Unlock()
	return fl.Logger.Sync()
}
