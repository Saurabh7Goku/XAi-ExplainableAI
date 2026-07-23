# 🥭 Mango Leaf Disease Detection System - Getting Started Guide

> **Project Focus**: This system demonstrates **Explainable AI (XAI)** capabilities in agricultural disease detection. While the Vision Transformer model provides disease classifications, the primary objective is to showcase **how AI models reason** and **which visual features influence predictions** — making black-box AI transparent and interpretable.

This guide will help you set up the complete Mango Leaf Disease Detection system with database integration and frontend-backend connectivity.

## 📋 Prerequisites

### Required Software
- **Python 3.9+** - Backend development
- **Node.js 18+** - Frontend development  
- **PostgreSQL 12+** - Database (or use Docker)
- **Git** - Version control

### Optional but Recommended
- **GPU** - For faster model training
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

## 🚀 Deployment Guide

### Backend: Northflank Deployment

This project is configured for deployment on [Northflank](https://northflank.com/).

#### Prerequisites
1. A [Northflank](https://app.northflank.com/) account
2. Your code pushed to a Git repository (GitHub, GitLab, etc.)
3. A [Gemini API Key](https://aistudio.google.com/apikey) (optional, for LLM reports)

#### Step 1: Create a Northflank Service
1. Log in to [Northflank Dashboard](https://app.northflank.com/)
2. Create a new **Web Service**
3. Connect your Git repository containing this project
4. Set the **Docker context** to `backend/`
5. Set the **Dockerfile path** to `backend/Dockerfile`

#### Step 2: Configure Environment Variables
In the Northflank service settings, add these environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `PORT` | `8000` | Internal port (Northflank injects this) |
| `DEBUG` | `false` | Disable debug mode |
| `DISABLE_DB_OPERATIONS` | `true` | Use in-memory database |
| `DATABASE_URL` | `sqlite:///:memory:` | In-memory DB (no persistence needed) |
| `MODEL_PATH` | `models/vit_mango_quantized.pth` | Path to model file |
| `HF_REPO_ID` | `Saurabh7Goku/vit-mango-leaf-disease` | HuggingFace repo for model download |
| `HF_FILENAME` | `vit_mango_quantized.pth` | Model filename on HuggingFace |
| `GEMINI_API_KEY` | *(your key)* | Optional: For LLM-generated reports |
| `CORS_ORIGINS` | `https://x-ai-explainable-ai.vercel.app` | Your Vercel frontend URL |
| `LOG_LEVEL` | `INFO` | Logging level |

#### Step 3: Configure Resources
- **CPU**: 1 vCPU (minimum)
- **Memory**: 2 GB (minimum, 4 GB recommended for faster model loading)
- **Disk**: 5 GB (for model storage and temp files)

#### Step 4: Deploy
1. Click **Create Service** and wait for the build to complete
2. Once deployed, Northflank will provide a public URL (e.g., `https://mango-leaf-api--xxxxxx.uc.r.appspot.com`)
3. Verify deployment by visiting `https://your-northflank-url.uc.r.appspot.com/`

### Frontend: Vercel Configuration

The frontend is already deployed on Vercel. To connect it to your Northflank backend:

1. Go to your Vercel project dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add or update:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: `https://your-northflank-url.uc.r.appspot.com` (replace with your actual Northflank URL)
4. Redeploy the frontend for changes to take effect

### Updating CORS for Production

After deployment, update the CORS origins in `backend/app/config.py` or via the `CORS_ORIGINS` environment variable to include your Northflank backend URL:

```env
CORS_ORIGINS=https://x-ai-explainable-ai.vercel.app,https://your-northflank-url.uc.r.appspot.com
```

## 🔐 Security Considerations

- **Never commit** `.env` files with API keys
- **Use HTTPS** in production (Northflank provides this automatically)
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
