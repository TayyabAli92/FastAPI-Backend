#!/bin/bash

# Deployment script for Book RAG Agent

echo "Starting deployment of Book RAG Agent..."

# Check if environment variables are set
if [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ] || [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: Required environment variables are not set"
    echo "Please set QDRANT_URL, QDRANT_API_KEY, and GEMINI_API_KEY"
    exit 1
fi

echo "Environment variables are set, proceeding with deployment..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run tests if available
if [ -f "test.py" ]; then
    echo "Running tests..."
    python test.py
fi

# Start the application
echo "Starting Book RAG Agent application..."
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}

echo "Deployment completed!"