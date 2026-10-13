@echo off
title Blueprint-X
cd /d "%~dp0"

echo =======================================================
echo              Starting Blueprint-X
echo =======================================================

if exist "venv313\Scripts\python.exe" (
    echo Using environment: venv313
    venv313\Scripts\python.exe main.py
) else if exist "venv\Scripts\python.exe" (
    echo Using environment: venv
    venv\Scripts\python.exe main.py
) else if exist ".venv\Scripts\python.exe" (
    echo Using environment: .venv
    .venv\Scripts\python.exe main.py
) else (
    echo Using system Python from PATH...
    python main.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo ERROR: App exited with error code %ERRORLEVEL%
    echo If dependencies are missing, please run:
    echo   pip install -r requirements.txt
    echo =======================================================
    pause
)

