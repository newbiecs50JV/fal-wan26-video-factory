#!/bin/bash

# Wan 2.6 Video Generation Pipeline
# Run this script to start the application

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set API key (you can also set this in a .env file)
export FAL_KEY="${FAL_KEY:-48255837-e6df-4f52-9f8b-d1e7ce581dd1:07f44bcbf3fd8fabba4024bb1a61e973}"
export VOICE_ID="${VOICE_ID:-Voice4c5cab3d1765912370}"

echo "Starting Wan 2.6 Video Generator..."
echo "   FAL_KEY: ${FAL_KEY:0:10}..."
echo "   VOICE_ID: $VOICE_ID"
echo ""

# Run the Flask app
python app.py






