@echo off
cd /d "%~dp0"
echo ======================================================================
echo Running Digital Twin Semantic Communication Main Pipeline...
echo ======================================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    python main.py
)
pause
