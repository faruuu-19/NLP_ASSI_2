#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-qwen2.5:1.5b}"
echo "Pulling ${MODEL_NAME} into the local Ollama registry..."
ollama pull "${MODEL_NAME}"
