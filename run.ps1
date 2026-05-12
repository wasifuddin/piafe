# PIAFE-AI — Windows PowerShell launch script
# Usage: .\run.ps1

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  PIAFE-AI  ·  Property Image Analysis Using AI"        -ForegroundColor Cyan
Write-Host "  ICT802 Capstone  ·  Tech Adaptive  ·  VIT"           -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate venv
Write-Host "Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies (first run may take 2-3 minutes)..."
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

Write-Host ""
Write-Host "Starting PIAFE-AI..." -ForegroundColor Green
Write-Host "Open your browser at: http://localhost:8501" -ForegroundColor Green
Write-Host ""

streamlit run app.py
