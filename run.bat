@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY=python"
if "%WHH_VENV%"=="1" goto :build     
python -c "import numpy, scipy, soundfile, librosa, matplotlib, tensorflow" >nul 2>nul
if not errorlevel 1 goto :ready


:build
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment in .venv ...
    where python >nul 2>nul
    if errorlevel 1 (
        echo error: Python was not found on PATH. Install Python 3.10+ first.
        exit /b 1
    )
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
set "PY=%CD%\.venv\Scripts\python.exe"

if not exist ".venv\.installed" (
    echo [setup] Installing dependencies - first run only, a few minutes ...
    "%PY%" -m pip install --upgrade pip >nul
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
    echo ok> ".venv\.installed"
)
:ready

if /i not "%~1"=="record" goto :analyse
set "ARGS="
:rec_loop
shift
if "%~1"=="" goto :rec_run
set ARGS=!ARGS! "%~1"
goto :rec_loop
:rec_run
"%PY%" record.py !ARGS!
exit /b %errorlevel%

:analyse
set "AUDIO=%~1"
if "%AUDIO%"=="" set "AUDIO=audio\street_3min.wav"
if not exist "%AUDIO%" (
    echo error: no such file: %AUDIO%
    exit /b 1
)
set "OUT=%~2"
if "%OUT%"=="" set "OUT=output"

"%PY%" what_happened_here.py "%AUDIO%" --outdir "%OUT%"
exit /b %errorlevel%
