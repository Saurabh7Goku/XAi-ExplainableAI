import psutil
import torch
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.repositories import PredictionRepository, SystemMetricsRepository
from app.database.models import ModelVersion
from app.core.inference import get_inference_pipeline, is_pipeline_initialized
from app.services.file_service import file_service
from app.config import settings
from app.utils.logger import logger

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Comprehensive health check of the system
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "checks": {}
        }
        
        # Database health check
        try:
            db.execute("SELECT 1")
            health_status["checks"]["database"] = {
                "status": "healthy",
                "message": "Database connection successful"
            }
        except Exception as e:
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "message": f"Database connection failed: {str(e)}"
            }
            health_status["status"] = "unhealthy"
        
        # Model health check (Lazy)
        if is_pipeline_initialized():
            try:
                model = get_inference_pipeline().model
                health_status["checks"]["model"] = {
                    "status": "healthy",
                    "message": "Model loaded successfully",
                    "device": str(next(model.parameters()).device)
                }
            except Exception as e:
                health_status["checks"]["model"] = {
                    "status": "unhealthy",
                    "message": f"Model loading failed: {str(e)}"
                }
                health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["model"] = {
                "status": "pending",
                "message": "Model not loaded yet (lazy initialization)"
            }
        
        # File system health check
        try:
            file_stats = file_service.get_file_stats()
            health_status["checks"]["filesystem"] = {
                "status": "healthy",
                "message": "File system accessible",
                "stats": file_stats
            }
        except Exception as e:
            health_status["checks"]["filesystem"] = {
                "status": "unhealthy",
                "message": f"File system error: {str(e)}"
            }
            health_status["status"] = "unhealthy"
        
        # Memory check
        memory = psutil.virtual_memory()
        health_status["checks"]["memory"] = {
            "status": "healthy" if memory.percent < 90 else "warning",
            "message": f"Memory usage: {memory.percent:.1f}%",
            "details": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            }
        }
        
        # GPU check (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_percent = (gpu_allocated / gpu_memory) * 100
                
                health_status["checks"]["gpu"] = {
                    "status": "healthy" if gpu_percent < 90 else "warning",
                    "message": f"GPU memory usage: {gpu_percent:.1f}%",
                    "details": {
                        "total_gb": round(gpu_memory / (1024**3), 2),
                        "allocated_gb": round(gpu_allocated / (1024**3), 2),
                        "percent": gpu_percent
                    }
                }
            except Exception as e:
                health_status["checks"]["gpu"] = {
                    "status": "warning",
                    "message": f"GPU check failed: {str(e)}"
                }
        else:
            health_status["checks"]["gpu"] = {
                "status": "info",
                "message": "GPU not available, using CPU"
            }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/system")
async def system_info() -> dict:
    """
    Get detailed system information
    """
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # GPU info
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_cached": torch.cuda.memory_reserved(0)
            }
        else:
            gpu_info = {"available": False}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": round((disk.used / disk.total) * 100, 2)
                },
                "gpu": gpu_info
            },
            "application": {
                "name": settings.app_name,
                "version": settings.app_version,
                "debug": settings.debug,
                "model_config": {
                    "img_size": settings.img_size,
                    "patch_size": settings.patch_size,
                    "num_classes": settings.num_classes,
                    "embed_dim": settings.embed_dim
                }
            }
        }
        
    except Exception as e:
        logger.error(f"System info failed: {str(e)}")
        return {"error": str(e)}


@router.get("/metrics")
async def get_system_metrics(
    metric_type: str = None,
    hours: int = 24,
    db: Session = Depends(get_db)
) -> dict:
    """
    Get system performance metrics
    """
    try:
        metrics_repo = SystemMetricsRepository(db)
        
        if metric_type:
            metrics = metrics_repo.get_recent_metrics(metric_type, hours)
            return {
                "metric_type": metric_type,
                "hours": hours,
                "metrics": [
                    {
                        "value": metric.metric_value,
                        "unit": metric.metric_unit,
                        "timestamp": metric.created_at.isoformat(),
                        "additional_info": metric.additional_info
                    }
                    for metric in metrics
                ]
            }
        else:
            # Get all recent metrics
            all_metrics = {}
            for mtype in ["prediction_time", "memory_usage", "cpu_usage"]:
                metrics = metrics_repo.get_recent_metrics(mtype, hours)
                if metrics:
                    all_metrics[mtype] = [
                        {
                            "value": metric.metric_value,
                            "unit": metric.metric_unit,
                            "timestamp": metric.created_at.isoformat()
                        }
                        for metric in metrics
                    ]
            
            return {
                "hours": hours,
                "metrics": all_metrics
            }
            
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        return {"error": str(e)}


@router.get("/model")
async def model_info(db: Session = Depends(get_db)) -> dict:
    """
    Get model information and version details
    """
    try:
        # Get active model from database
        active_model = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        
        # Load model info only if initialized
        if is_pipeline_initialized():
            model = get_inference_pipeline().model
            model_info = {
                "status": "loaded",
                "device": str(next(model.parameters()).device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
        else:
            model_info = {
                "status": "pending",
                "message": "Model not loaded yet (lazy load on first request)",
                "parameters": 0,
                "trainable_parameters": 0
            }
        
        model_info["config"] = {
            "img_size": settings.img_size,
            "patch_size": settings.patch_size,
            "num_classes": settings.num_classes,
            "embed_dim": settings.embed_dim,
            "depth": settings.depth,
            "num_heads": settings.num_heads
        }
        
        if active_model:
            model_info.update({
                "version": active_model.version,
                "architecture": active_model.architecture,
                "training_accuracy": active_model.training_accuracy,
                "validation_accuracy": active_model.validation_accuracy,
                "created_at": active_model.created_at.isoformat(),
                "deployed_at": active_model.deployed_at.isoformat() if active_model.deployed_at else None
            })
        else:
            model_info["version"] = "unknown"
            model_info["message"] = "No active model found in database"
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return {"error": str(e), "status": "failed"}
