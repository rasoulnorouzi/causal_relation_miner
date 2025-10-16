@echo off
REM Causal Relationship Extractor - Batch Launcher
echo ================================================
echo   Causal Relationship Extractor
echo   Starting Streamlit App...
echo ================================================
echo.

REM Change to the folder of this script
cd /d "%~dp0"

set "VENV_PY=%~dp0myenv\Scripts\python.exe"
if not exist "%VENV_PY%" (
	echo [WARN] venv Python not found at %VENV_PY%
	echo        Falling back to system python on PATH.
	set "VENV_PY=python"
)

"%VENV_PY%" -m streamlit run app.py
pause
