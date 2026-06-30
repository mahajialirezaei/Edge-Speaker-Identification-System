@echo off
echo 🚀 Starting tests inside the Docker container...
echo ===================================================

docker-compose run --rm speaker-id pytest tests/ -v

echo ===================================================
echo ✅ Test execution finished.
pause