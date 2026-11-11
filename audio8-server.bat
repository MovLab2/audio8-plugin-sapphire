@echo off
rem Audio8 as a GPU server on its own, no Sapphire on this computer.
rem Another computer's Sapphire uses it with Settings > Audio8 > Client only.
rem Every start prints the same pairing string: paste it there once.
rem Everything it keeps lands in .\user\ beside this file (its certificate,
rem the keys of the computers let in, the voices they sent); the model (~2.5 GB)
rem goes to the Hugging Face cache. Inside a Sapphire install
rem (user\plugins\audio8) it uses that Sapphire's user\ instead: never files in
rem the plugin's own folder, which its signature would refuse.
rem Run it from an Anaconda Prompt (Miniconda), in a short folder path.
rem From PowerShell, pairing strings lose their quotes: use the Anaconda Prompt.
rem
rem   audio8-server.bat               run it (builds its conda env the first time, ~4 GB)
rem   audio8-server.bat add NAME      let another computer in: prints its pairing string
rem   audio8-server.bat remove NAME   shut one out
rem   audio8-server.bat list          who is let in
rem
rem Port 5015 (set PORT to change it) is the door other computers use: allow it in the firewall.
setlocal
pushd "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
set "ENV=audio8-server"
if not "%AUDIO8_ENV%"=="" set "ENV=%AUDIO8_ENV%"
if "%PORT%"=="" set "PORT=5015"
set "ROOT=%CD%"
if exist "..\..\..\sapphire.py" for %%I in ("..\..\..") do set "ROOT=%%~fI"
set "DOOR=%ROOT%\user\worker_door\audio8"

where conda >nul 2>nul || (echo conda not found: run this from an Anaconda Prompt & exit /b 1)
rem Made only when conda lists no such env: a failed activate can also mean conda
rem isn't set up in this window, and conda create -y over an env wipes it.
conda env list | findstr /b /c:"%ENV% " >nul
if errorlevel 1 (
    echo Making the %ENV% environment, once...
    call conda create -y -n %ENV% python=3.12 --override-channels -c conda-forge || exit /b 1
)
call conda activate %ENV% || (echo conda could not activate %ENV%: run this from an Anaconda Prompt & exit /b 1)

rem The packages are the ones plugin.json lists, read from it: one list, never two
rem (plus cryptography for the door). What was installed is noted inside the
rem env, so a remade env installs again.
if not exist "%ROOT%\user" mkdir "%ROOT%\user"
set "STAMP=%CONDA_PREFIX%\audio8-requirements.txt"
set "NEW=%ROOT%\user\audio8-requirements.new"
python -c "import json; print('\n'.join(json.load(open('plugin.json', encoding='utf-8'))['environment']['pip'] + ['cryptography']))" > "%NEW%" || exit /b 1
fc /b "%NEW%" "%STAMP%" >nul 2>nul
if errorlevel 1 (
    echo Installing packages, the first time downloads ~4 GB...
    python -m pip install -r "%NEW%" || exit /b 1
    move /y "%NEW%" "%STAMP%" >nul
) else (
    del "%NEW%"
)

if /i "%~1"=="add" (python worker_door.py add "%DOOR%" %2 %PORT% & exit /b)
if /i "%~1"=="remove" (python worker_door.py remove "%DOOR%" %2 & exit /b)
if /i "%~1"=="list" (python worker_door.py list "%DOOR%" & exit /b)
if not "%~1"=="" (echo usage: audio8-server.bat [add NAME ^| remove NAME ^| list] & exit /b 2)

set "SAPPHIRE_ROOT=%ROOT%"
set "SAPPHIRE_SERVICE_PORT=5014"
set "SAPPHIRE_PLUGIN_SETTINGS={"share": true, "share_port": %PORT%}"
python server.py
