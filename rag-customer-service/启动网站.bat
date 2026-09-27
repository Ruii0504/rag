@echo off
setlocal
title RAG Customer Service

cd /d "%~dp0"
set "APP_URL=http://127.0.0.1:8000"
set "STATUS_URL=http://127.0.0.1:8000/api/status"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" goto missing_python

powershell.exe -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%STATUS_URL%' -TimeoutSec 2; if ($response.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
    echo Service is already running. Opening browser...
    start "" "%APP_URL%"
    exit /b 0
)

echo Starting RAG Customer Service...
echo URL: %APP_URL%
echo Close this window or press Ctrl+C to stop the service.
echo.

start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$url = '%APP_URL%'; $statusUrl = '%STATUS_URL%'; for ($attempt = 0; $attempt -lt 60; $attempt++) { try { $response = Invoke-WebRequest -UseBasicParsing -Uri $statusUrl -TimeoutSec 1; if ($response.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; exit 1"

"%PYTHON_EXE%" -m uvicorn rag_customer_service.server:create_production_app --factory --host 127.0.0.1 --port 8000
set "SERVER_EXIT_CODE=%ERRORLEVEL%"

if not "%SERVER_EXIT_CODE%"=="0" (
    echo.
    echo Server failed to start. Review the error above.
    pause
)

exit /b %SERVER_EXIT_CODE%

:missing_python
echo Python virtual environment was not found:
echo %PYTHON_EXE%
echo.
echo Complete the local installation steps in README first.
pause
exit /b 1
