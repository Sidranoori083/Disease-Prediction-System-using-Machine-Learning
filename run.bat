@echo off
title Disease Predictor Setup
echo ==========================================
echo   Disease Predictor - Starting up...
echo ==========================================
echo.

echo [1/2] Installing required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install packages.
    echo Make sure Python and pip are installed.
    echo Download Python from https://www.python.org
    pause
    exit /b 1
)

echo.
echo [2/2] Starting the app...
echo Open your browser and go to: http://127.0.0.1:5000
echo Press Ctrl+C to stop the app.
echo.
python app.py
pause
