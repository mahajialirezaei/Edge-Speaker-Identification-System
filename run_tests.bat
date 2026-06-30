@echo off
echo 🚀 Starting tests inside the Docker container...
echo ===================================================

python -m pytest tests/ -v

echo ===================================================
echo ✅ Test execution finished.
pause