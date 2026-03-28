@echo off
setlocal

set "ROOT=E:\Python Workspace\MyFramework\oracle"
set "PYTHON_EXE=D:\conda_envs\envs\pytorch\python.exe"

cd /d "%ROOT%"
"%PYTHON_EXE%" train_oracle_gate.py --save_log True >> "%ROOT%\autoresearch-runtime-cmd.log" 2>&1
