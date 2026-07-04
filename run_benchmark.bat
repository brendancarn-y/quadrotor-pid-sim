@echo off
REM Double-click this file to run the PID vs PPO benchmark and open the result.
cd /d "%~dp0"
".venv\Scripts\python.exe" evaluate.py
echo.
echo Opening the result figure...
start "" "results\comparison.png"
echo.
pause
