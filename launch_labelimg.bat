@echo off
echo Starting LabelImg for Tower Detection Dataset...
cd /d "%~dp0"
call "venv\Scripts\python.exe" launch_labelimg.py
pause
