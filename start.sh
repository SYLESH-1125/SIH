#!/bin/bash
# Railway startup script for Agriculture AI

echo "🌾 Starting Agriculture AI on Railway..."
echo "📦 Installing dependencies..."

# Install Python dependencies
pip install -r requirements.txt

echo "🚀 Starting server..."
uvicorn whisper_main:app --host 0.0.0.0 --port $PORT
