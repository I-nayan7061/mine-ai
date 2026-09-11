@echo off
title NexGen Mine Subsidence AI Server
echo ================================================================
echo Starting NexGen Mine Subsidence AI Localhost Server...
echo ================================================================
if exist "%~dp0mine-ai\api\main.py" (
    cd /d "%~dp0mine-ai"
) else (
    cd /d "%~dp0"
)
echo.
echo Localhost Dashboard: http://localhost:8000/
echo API Documentation:   http://localhost:8000/docs
echo Health Check:        http://localhost:8000/api/health
echo.
echo Press Ctrl+C in this console window to stop the server.
echo ================================================================
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
