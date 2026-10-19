@echo off
REM ==== Start the app: double-click this file ====
cd /d "%~dp0"
if not exist .venv (echo Run setup.bat first. & pause & exit /b)
call .venv\Scripts\activate.bat
streamlit run app.py
pause
