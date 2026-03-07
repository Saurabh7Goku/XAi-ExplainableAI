@echo off
REM Mango Leaf Disease Detection System Setup Script for Windows

echo Setting up Mango Leaf Disease Detection System...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is required but not installed. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is required but not installed. Please install Node.js 18+
    pause
    exit /b 1
)

echo Prerequisites check completed

REM Setup Backend
echo.
echo Setting up Backend...
cd backend

REM Create virtual environment
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Setup environment file
if not exist ".env" (
    echo Creating environment file...
    copy .env.example .env
    echo ⚠️  Please edit backend\.env with your configuration
)

REM Setup database
echo Setting up database...
python setup_database.py

cd ..

REM Setup Frontend
echo.
echo Setting up Frontend...
cd frontend

REM Install Node.js dependencies
echo Installing Node.js dependencies...
npm install

REM Setup environment file
if not exist ".env.local" (
    echo Creating frontend environment file...
    echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local
)

cd ..

echo.
echo Setup completed successfully!
echo.
echo Next Steps:
echo 1. Edit backend\.env with your database and API keys
echo 2. Organize your training data in backend\data\ folder
echo 3. Start the services:
echo.
echo    Terminal 1 - Backend:
echo    cd backend && venv\Scripts\activate && uvicorn app.main:app --reload
echo.
echo    Terminal 2 - Frontend:
echo    cd frontend && npm run dev
echo.
echo 4. Open http://localhost:3000 to access the application
echo 5. Visit http://localhost:8000/docs for API documentation
echo.
echo For more information, see backend\README.md
pause
