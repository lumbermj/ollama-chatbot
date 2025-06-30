#!/usr/bin/env bash
set -e

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &

# Wait until Ollama is ready
until ollama list >/dev/null 2>&1; do
  sleep 1
done

# Ensure the model is available
echo "Pulling llama3 model..."
ollama pull llama3

echo "Pulling embeddings model (hf.co/CompendiumLabs/bge-base-en-v1.5-gguf)..."
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf

# Launch the FastAPI application (pointing at src/main.py)
echo "Starting FastAPI app..."
exec ./venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080