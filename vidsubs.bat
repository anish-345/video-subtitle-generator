@echo off
title Whisper Subtitle Studio
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\run.ps1" %*
