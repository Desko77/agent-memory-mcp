package server

import (
	"encoding/json"
	"fmt"
)

type toolContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type toolResult struct {
	Content []toolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

func toolResultText(text string) toolResult {
	return toolResult{
		Content: []toolContent{{Type: "text", Text: text}},
	}
}

func toolResultJSON(value any) toolResult {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return toolResultText(fmt.Sprintf("failed to encode result: %v", err))
	}
	return toolResultText(string(data))
}
