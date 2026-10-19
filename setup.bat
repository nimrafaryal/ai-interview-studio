@echo off
REM ==== One-time setup: double-click this file ====
cd /d "%~dp0"
echo.
echo === AI Interview Studio - setup ===
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
%PY% --version || (echo Python not found. Install Python 3.11 from python.org and tick "Add to PATH". & pause & exit /b)
if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing CPU-only PyTorch (smaller, no GPU needed)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
echo.
echo === Setup finished. Now double-click run.bat ===
pause
