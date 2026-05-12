#!/bin/bash
# PIAFE-AI — one-command setup and launch
# Usage: bash run.sh

echo ""
echo "======================================================"
echo "  PIAFE-AI  ·  Property Image Analysis Using AI"
echo "  ICT802 Capstone  ·  Tech Adaptive  ·  VIT"
echo "======================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies (first run may take 2-3 minutes)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Starting PIAFE-AI..."
echo "Open your browser at: http://localhost:8501"
echo ""

streamlit run app.py
