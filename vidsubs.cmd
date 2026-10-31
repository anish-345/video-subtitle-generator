@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "run.ps1" %*
