@echo off
REM PIAFE-AI — Windows Batch launch script
REM Usage: run.bat

echo.
echo ======================================================
echo   PIAFE-AI  .  Property Image Analysis Using AI
echo   ICT802 Capstone  .  Tech Adaptive  .  VIT
echo ======================================================
echo.

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed. Please install Python 3.10+
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies (first run may take 2-3 minutes)...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo.
echo Starting PIAFE-AI...
echo Open your browser at: http://localhost:8501
echo.

streamlit run app.py
pause
