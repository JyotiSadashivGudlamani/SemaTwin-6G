@echo off
cd /d "%~dp0"
echo ======================================================================
echo Launching 6G Semantic Communication ^& Digital Twin Control Center...
echo ======================================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run dashboard/app.py
) else (
    python -m streamlit run dashboard/app.py
)
pause
