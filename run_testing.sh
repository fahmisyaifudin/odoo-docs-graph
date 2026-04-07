#!/bin/bash

# Run testing.py with different LLM models using no-context method

echo "=========================================="
echo "Running: meta-llama/llama-3.1-8b-instruct"
echo "=========================================="
python testing.py "meta-llama/llama-3.1-8b-instruct" "no-context"

echo ""
echo "=========================================="
echo "Running: meta-llama/llama-3.1-70b-instruct"
echo "=========================================="
python testing.py "meta-llama/llama-3.1-70b-instruct" "no-context"

echo ""
echo "=========================================="
echo "Running: qwen/qwen3.5-9b"
echo "=========================================="
python testing.py "qwen/qwen3.5-9b" "no-context"

echo ""
echo "=========================================="
echo "Running: qwen/qwen3.5-35b-a3b"
echo "=========================================="
python testing.py "qwen/qwen3.5-35b-a3b" "no-context"

echo ""
echo "=========================================="
echo "Running: google/gemma-3-12b-it"
echo "=========================================="
python testing.py "google/gemma-3-12b-it" "no-context"

echo ""
echo "=========================================="
echo "Running: google/gemma-3-27b-it"
echo "=========================================="
python testing.py "google/gemma-3-27b-it" "no-context"

echo ""
echo "All tests completed!"
