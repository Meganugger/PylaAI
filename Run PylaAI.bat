@echo off
setlocal
cd /d %~dp0

echo.
echo PylaAI launcher:
echo   - Use this file for normal launches
echo   - Use scripts\setup.bat only for first-time install or manual repair
echo.

if not exist ".venv\Scripts\python.exe" goto setup_needed

.\.venv\Scripts\python.exe tools\runtime_preflight.py
if errorlevel 2 goto repair
if errorlevel 1 goto setup_needed
goto start

:setup_needed
echo.
echo PylaAI is not ready yet. Running setup...
call .\scripts\setup.bat
if errorlevel 1 goto repair_failed
if not exist ".venv\Scripts\python.exe" goto repair_failed
.\.venv\Scripts\python.exe tools\runtime_preflight.py
if errorlevel 1 goto repair_failed
goto start

:repair
echo.
echo Repairing PylaAI dependencies. This can take a few minutes...
call .\scripts\setup.bat
if errorlevel 1 goto repair_failed
if not exist ".venv\Scripts\python.exe" goto repair_failed
.\.venv\Scripts\python.exe tools\runtime_preflight.py
if errorlevel 1 goto repair_failed
goto start

:repair_failed
echo.
echo PylaAI could not be repaired automatically.
echo If ONNX Runtime still fails to import, install Microsoft Visual C++ Redistributable 2015-2022 x64:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Then restart Windows and run this launcher again.
pause
exit /b 1

:start
echo Starting PylaAI...
.\.venv\Scripts\python.exe main.py
pause
