@echo off
cd /d "%~dp0.."
powershell.exe -ExecutionPolicy Bypass -File ".\scripts\install_windows.ps1" -Editable
pause
