import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database.database import get_db
from app.database.repositories import ModelVersionRepository, TrainingRunRepository
from app.core.training import TrainingPipeline
from app.config import settings
from app.utils.exceptions import DatabaseError
from app.utils.logger import logger

router = APIRouter(prefix="/training", tags=["training"])


class TrainingRequest(BaseModel):
    """Request model for training"""
    model_name: str = "vit_mango_v1"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    data_dir: str = "data"
    validation_split: float = 0.2
    save_best_only: bool = True
    early_stopping_patience: int = 10


class TrainingResponse(BaseModel):
    """Response model for training"""
    success: bool
    message: str
    training_run_id: Optional[int] = None
    error: Optional[str] = None


@router.post("/start", response_model=TrainingResponse)
async def start_training(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> TrainingResponse:
    """
    Start model training in background
    """
    try:
        # Validate data directory exists
        if not os.path.exists(training_request.data_dir):
            raise HTTPException(
                status_code=400, 
                detail=f"Data directory not found: {training_request.data_dir}"
            )
        
        # Create training run record
        training_run_id = None
        try:
            training_repo = TrainingRunRepository(db)
            training_run_data = {
                'run_name': training_request.model_name,
                'total_epochs': training_request.epochs,
                'epochs_completed': 0,
                'status': 'running',
                'dataset_info': {
                    'data_dir': training_request.data_dir,
                    'validation_split': training_request.validation_split,
                    'batch_size': training_request.batch_size
                },
                'hyperparameters': {
                    'learning_rate': training_request.learning_rate,
                    'batch_size': training_request.batch_size,
                    'epochs': training_request.epochs
                }
            }
            
            training_run = training_repo.create(training_run_data)
            training_run_id = training_run.id
        except Exception as e:
            logger.warning(f"Failed to create training run record: {str(e)}")
        
        # Start training in background
        background_tasks.add_task(
            run_training_task,
            training_request.dict(),
            training_run_id
        )
        
        logger.info(f"Training started: {training_request.model_name} (run_id: {training_run.id})")
        
        return TrainingResponse(
            success=True,
            message=f"Training started successfully. Run ID: {training_run_id}",
            training_run_id=training_run_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        return TrainingResponse(
            success=False,
            message="Failed to start training",
            error=str(e)
        )


@router.get("/status/{run_id}")
async def get_training_status(
    run_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get training status for a specific run
    """
    try:
        training_repo = TrainingRunRepository(db)
        training_run = training_repo.get_by_id(run_id)
        
        if not training_run:
            raise HTTPException(status_code=404, detail="Training run not found")
        
        return {
            "success": True,
            "training_run": {
                "id": training_run.id,
                "run_name": training_run.run_name,
                "status": training_run.status,
                "epochs_completed": training_run.epochs_completed,
                "total_epochs": training_run.total_epochs,
                "final_training_loss": training_run.final_training_loss,
                "final_validation_loss": training_run.final_validation_loss,
                "best_validation_accuracy": training_run.best_validation_accuracy,
                "training_time_minutes": training_run.training_time_minutes,
                "error_message": training_run.error_message,
                "started_at": training_run.started_at.isoformat(),
                "completed_at": training_run.completed_at.isoformat() if training_run.completed_at else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get training status")


@router.get("/runs")
async def list_training_runs(
    limit: int = 50,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    List all training runs
    """
    try:
        training_repo = TrainingRunRepository(db)
        runs = training_repo.get_recent(limit)
        
        return {
            "success": True,
            "training_runs": [
                {
                    "id": run.id,
                    "run_name": run.run_name,
                    "status": run.status,
                    "epochs_completed": run.epochs_completed,
                    "total_epochs": run.total_epochs,
                    "best_validation_accuracy": run.best_validation_accuracy,
                    "started_at": run.started_at.isoformat(),
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None
                }
                for run in runs
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list training runs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list training runs")


@router.post("/stop/{run_id}")
async def stop_training(
    run_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Stop a running training job
    """
    try:
        training_repo = TrainingRunRepository(db)
        training_run = training_repo.get_by_id(run_id)
        
        if not training_run:
            raise HTTPException(status_code=404, detail="Training run not found")
        
        if training_run.status != "running":
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot stop training with status: {training_run.status}"
            )
        
        # Update status to stopped
        training_repo.update_status(run_id, "stopped")
        
        logger.info(f"Training stopped: run_id {run_id}")
        
        return {
            "success": True,
            "message": f"Training run {run_id} stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop training")


@router.get("/models")
async def list_model_versions(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    List all model versions
    """
    try:
        model_repo = ModelVersionRepository(db)
        models = model_repo.get_all()
        
        return {
            "success": True,
            "models": [
                {
                    "id": model.id,
                    "version": model.version,
                    "architecture": model.architecture,
                    "num_parameters": model.num_parameters,
                    "training_accuracy": model.training_accuracy,
                    "validation_accuracy": model.validation_accuracy,
                    "test_accuracy": model.test_accuracy,
                    "is_active": model.is_active,
                    "created_at": model.created_at.isoformat(),
                    "deployed_at": model.deployed_at.isoformat() if model.deployed_at else None
                }
                for model in models
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list model versions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list model versions")


@router.post("/models/{version}/activate")
async def activate_model(
    version: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Activate a specific model version
    """
    try:
        model_repo = ModelVersionRepository(db)
        success = model_repo.set_active(version)
        
        if success:
            logger.info(f"Model activated: {version}")
            return {
                "success": True,
                "message": f"Model version {version} activated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Model version not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to activate model")


async def run_training_task(training_params: Dict[str, Any], training_run_id: int) -> None:
    """
    Background task to run training
    """
    try:
        # Initialize training pipeline
        training_pipeline = TrainingPipeline()
        
        # Run training
        result = training_pipeline.train(
            data_dir=training_params['data_dir'],
            epochs=training_params['epochs'],
            batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate'],
            validation_split=training_params['validation_split'],
            save_best_only=training_params['save_best_only'],
            early_stopping_patience=training_params['early_stopping_patience'],
            run_id=training_run_id
        )
        
        logger.info(f"Training completed: run_id {training_run_id}")
        
    except Exception as e:
        logger.error(f"Training failed: run_id {training_run_id}, error: {str(e)}")
        
        # Update database with error
        if training_run_id:
            try:
                from app.database.database import get_db
                db = get_db()
                training_repo = TrainingRunRepository(db)
                training_repo.update_status(training_run_id, "failed", str(e))
            except Exception as db_error:
                logger.error(f"Failed to update training status: {str(db_error)}")
