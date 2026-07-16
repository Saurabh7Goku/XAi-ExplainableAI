# 🥭 Mango Leaf Disease Detection System - Getting Started Guide

This guide will help you set up the complete Mango Leaf Disease Detection system with database integration and frontend-backend connectivity.

## 📋 Prerequisites

### Required Software
- **Python 3.9+** - Backend development
- **Node.js 18+** - Frontend development  
- **PostgreSQL 12+** - Database (or use Docker)
- **Git** - Version control

### Optional but Recommended
- **NVIDIA GPU** - For faster model training
- **Docker & Docker Compose** - For easy deployment
- **Gemini API Key** - For LLM report generation

## 🚀 Quick Setup (Automated)

### Option 1: Docker (Recommended)
```bash
# From project root - Single command!
docker-compose up -d --build
```

### Option 2: Manual Setup

#### Windows Users
```bash
# Run setup script
setup.bat
```

## 🛠️ Manual Setup

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment file
.env
# Edit .env with your settings
```

#### Environment Configuration
Edit `backend/.env`:
```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/mango_leaf_db

# LLM (optional but recommended)
GEMINI_API_KEY=your_gemini_api_key_here

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

#### Database Setup
```bash
# Option 1: Using PostgreSQL directly
createdb mango_leaf_db

# Option 2: Using Docker
docker-compose up -d postgres

# Initialize database tables
python setup_database.py
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Setup environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

## 🏃‍♂️ Running the Application

### Method 1: Development Mode

#### Terminal 1 - Backend Server
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2 - Frontend Server
```bash
cd frontend
npm run dev
```

### Method 2: Docker (Recommended for Production)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🌐 Access Points

Once running, you can access:

- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health/

## 📊 Training Your Model

### Prepare Dataset
Organize your images in this structure:
```
backend/data/
├── Healthy/
├── Anthracnose/
├── Bacterial Canker/
├── Cutting Weevil Damage/
└── Die Back/
```

### Start Training

#### Option A: Command Line
```bash
cd backend
python training/train.py --data-dir data --epochs 50 --batch-size 32
```

#### Option B: API Endpoint
```bash
curl -X POST "http://localhost:8000/training/start" \
     -H "Content-Type: application/json" \
     -d '{
       "model_name": "vit_mango_v1",
       "epochs": 50,
       "batch_size": 32,
       "data_dir": "data"
     }'
```

## 🔧 Testing the Integration

### 1. Test API Connection
```bash
curl http://localhost:8000/health/
```

### 2. Test Prediction
```bash
curl -X POST "http://localhost:8000/predict/" \
     -F "file=@your_test_image.jpg"
```

### 3. Test Frontend Integration
1. Open http://localhost:3000
2. Upload a mango leaf image
3. View prediction results and LIME explanation

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
# Check PostgreSQL status
pg_isready -h localhost -p 5432

# Check database exists
psql -h localhost -p 5432 -U postgres -l

# Reset database
dropdb mango_leaf_db && createdb mango_leaf_db
```

#### Frontend Cannot Connect to Backend
1. Check if backend is running on port 8000
2. Verify `NEXT_PUBLIC_API_URL` in `frontend/.env.local`
3. Check CORS settings in backend configuration

#### Model Loading Errors
1. Ensure you have trained a model or have a pre-trained one
2. Check model path in `backend/.env`
3. Verify PyTorch installation with CUDA support if using GPU

#### LLM Reports Not Working
1. Add your Gemini API key to `backend/.env`
2. Check internet connection
3. Verify API key is valid and has credits

### Getting Help

1. **Check logs**: `backend/logs/app.log`
2. **API Documentation**: http://localhost:8000/docs
3. **Health Check**: http://localhost:8000/health/
4. **System Info**: http://localhost:8000/health/system

## 📚 Next Steps

1. **Prepare Training Data**: Collect and organize mango leaf images
2. **Train Model**: Run training with your dataset
3. **Test System**: Verify predictions work correctly
4. **Deploy**: Use Docker for production deployment
5. **Monitor**: Set up logging and monitoring

## 🔐 Security Considerations

- **Never commit** `.env` files with API keys
- **Use HTTPS** in production
- **Validate inputs** on both frontend and backend
- **Implement rate limiting** for API endpoints
- **Regular updates** of dependencies

## 📈 Performance Optimization

- **GPU Training**: Use CUDA-enabled PyTorch
- **Batch Processing**: Process multiple images simultaneously
- **Caching**: Implement Redis for frequent queries
- **Load Balancing**: Multiple API instances for scale

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

For detailed documentation, see:
- `backend/README.md` - Backend specific documentation
- `frontend/README.md` - Frontend specific documentation
- API docs at http://localhost:8000/docs
