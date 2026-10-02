@echo off
echo Starting SIH 26084 Nowcasting Prototype...

echo 1. Starting FastAPI Backend...
start "FastAPI Backend" cmd /c "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo 2. Frontend Instructions:
echo To run the frontend, you need Node.js installed.
echo If you have Node.js, run the following in another terminal:
echo cd frontend
echo npm install
echo npm run dev

echo The backend is running at http://localhost:8000
echo You can view the API docs at http://localhost:8000/docs
pause
