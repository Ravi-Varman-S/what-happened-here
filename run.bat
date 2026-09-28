@echo off
rem ===========================================================================
rem  What Happened Here? - one command to set up and run.
rem
rem    run.bat                 analyse the bundled demo clip (default)
rem    run.bat my.wav          analyse your own recording
rem    run.bat my.wav my_out   ... writing results into my_out\
rem    run.bat record          record 3 min from the microphone, then analyse
rem    run.bat record --seconds 150 --device 1
rem
rem  First run installs the dependencies (a few minutes, one time only);
rem  afterwards the same command just runs the analysis.
rem ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem --- prefer an existing Python that already has the libraries --------------
set "PY=python"
python -c "import numpy, scipy, soundfile, librosa, matplotlib, tensorflow" >nul 2>nul
if not errorlevel 1 goto :ready

rem --- otherwise build a local virtual environment ---------------------------
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

rem --- "run.bat record [flags] ..." ------------------------------------------
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

rem --- "run.bat [audio] [outdir]" --------------------------------------------
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
