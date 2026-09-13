
@echo off
chcp 65001 >nul
title 3080 Dual-Card qwen38-27b Launch
setlocal

rem [1/4] Ensure WSL keepalive session (prevents distro idle userspace reset)
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'wsl.exe' -and $_.CommandLine -match 'tail -f /dev/null' })) { Start-Process -FilePath wsl -ArgumentList '-d','Ubuntu','--exec','tail','-f','/dev/null' -WindowStyle Hidden }"

rem --- Check whether the service is already running ---
for /f %%a in ('curl -s -o NUL -w ""%%{http_code}"" --max-time 3 http://127.0.0.1:8788/health 2^>nul') do set HC=%%a
if "%HC%"=="200" (
  echo [Info] Service is already running: http://127.0.0.1:8788/v1
  echo Model: Qwen3.8-27B-FP4-Dflash-test; key can be any placeholder value, e.g. sk-local
  echo To stop the service, run: wsl -d Ubuntu -u root -- systemctl stop sglang-qwen38
  goto end
)

echo [2/4] Starting SGLang service via systemd (NVFP4 + DFlash + HiCache, GPU1+GPU2, port 8788)...
wsl -d Ubuntu -u root -- systemctl start sglang-qwen38
if errorlevel 1 (
  echo [Error] systemctl start failed. Check: wsl -d Ubuntu -u root -- journalctl -u sglang-qwen38 -n 50
  goto end
)

echo [3/4] Waiting for service to be ready (usually 2-4 minutes, this window can be minimized)...
set /a n=0
:loop
set /a n+=1
if %n% gtr 45 goto timeout
ping -n 11 127.0.0.1 >nul
for /f %%a in ('curl -s -o NUL -w ""%%{http_code}"" --max-time 3 http://127.0.0.1:8788/health 2^>nul') do set HC=%%a
if "%HC%"=="200" goto ready
goto loop

:timeout
echo [Timeout] Still not ready after 7.5 minutes. Check the log:
echo   wsl -d Ubuntu -u root -- journalctl -u sglang-qwen38 -n 50
goto end

:ready
echo [4/4] [Done] Service ready: http://127.0.0.1:8788/v1
echo Model: Qwen3.8-27B-FP4-Dflash-test; key can be any placeholder value, e.g. sk-local
echo Stable context: 220K; KV pool: 226816 tokens; HiCache: 12GB/rank

:end
endlocal
ping -n 6 127.0.0.1 >nul
