@echo off
echo ============================================
echo   Perovskite Defect Detection Web App
echo ============================================
echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.
echo Starting server at http://localhost:8000
echo Press Ctrl+C to stop.
echo.
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
