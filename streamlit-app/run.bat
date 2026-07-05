@echo off
rem Launch the Safer Roads Streamlit dashboard locally.
cd /d "%~dp0"
python -m streamlit run app.py
pause
