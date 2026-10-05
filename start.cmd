@echo off
rem setlocal keeps TCM_RELOAD and the current directory from leaking into an already-open cmd session.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The Python environment .venv was not found. Double-click install.cmd first ^(see README.md^).
    pause
    exit /b 1
)
rem One-click start does not use NiceGUI's auto-reload (a development feature); run "python TCM_Meridian_main.py" directly to get it.
set TCM_RELOAD=0
".venv\Scripts\python.exe" TCM_Meridian_main.py
set RESULT=%errorlevel%
if not "%RESULT%"=="0" pause
exit /b %RESULT%
