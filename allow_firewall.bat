@echo off
echo ===================================================
echo   Opening Windows Firewall Port 8000 for Wi-Fi Access
echo ===================================================
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo [ERROR] This script requires Administrator privileges.
    echo Please right-click 'allow_firewall.bat' and choose 'Run as administrator'.
    echo.
    pause
    exit /b 1
)

netsh advfirewall firewall add rule name="Whisper Studio Rust Gateway Port 8000" dir=in action=allow protocol=TCP localport=8000
echo.
echo [SUCCESS] Port 8000 is now open on your local Wi-Fi!
echo Other devices on your local network can now open:
echo   http://^<YOUR_LOCAL_IP^>:8000
echo.
echo Check the exact URL printed in 'run.ps1' or 'start.bat' when launching the application.
echo.
pause
