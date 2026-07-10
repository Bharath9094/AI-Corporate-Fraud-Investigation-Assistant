@echo off
echo ===================================================================
echo   AI Corporate Fraud Investigation Assistant - Launch Script
echo ===================================================================
echo.
echo [1/2] Starting Python FastAPI Backend Server on http://localhost:8000...
start "Sentinel Fraud - FastAPI Backend" cmd /k "cd backend && python main.py"

echo.
echo [2/2] Starting React Vite Frontend Server on http://localhost:5173...
start "Sentinel Fraud - Vite Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Waiting 3 seconds for servers to initialize...
timeout /t 3 /nobreak > nul

echo.
echo Launching default web browser to portal...
start http://localhost:5173

echo.
echo ===================================================================
echo   System running! Close the spawned terminals to stop the servers.
echo ===================================================================
echo.
pause
