import time
import io
import base64
from typing import Dict, Any
import json
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from PIL import Image

from app.database.database import get_db
from app.database.repositories import PredictionRepository, LLMReportRepository
from app.core.inference import get_inference_pipeline
from app.core.xai import get_lime_explainer
from app.services.knowledge_base import knowledge_base_service
from app.services.llm_service import llm_service
from app.services.file_service import file_service
from app.utils.exceptions import (
    PredictionError, InvalidImageError, FileUploadError, LLMServiceError
)
from app.utils.logger import logger

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/")
async def predict_disease(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Predict mango leaf disease from uploaded image
    
    Args:
        file: Image file to analyze
        db: Database session
        
    Returns:
        Prediction results with LIME explanation and LLM report
    """
    # Pre-prediction cleanup - clear all existing data
    try:
        import gc
        import torch
        
        # Clear any existing cached data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear old temp files
        file_service.cleanup_temp_files(max_age_hours=0)  # Clean all temp files
        
        logger.info("Pre-prediction cleanup completed")
    except Exception as e:
        logger.warning(f"Pre-cleanup failed: {str(e)}")
    
    start_time = time.time()
    
    try:
        # Save uploaded file
        file_path, filename = file_service.save_uploaded_file(file)
        logger.info(f"File uploaded: {filename}")
        
        # Make prediction
        predicted_class, confidence, class_probabilities = get_inference_pipeline().predict(file_path)
        
        # Generate LIME explanation
        explanation_class, lime_explanation = get_lime_explainer().explain_image(file_path)
        
        # Save explanation image
        explanation_path = file_service.save_explanation_image(
            lime_explanation, f"lime_{filename}"
        )
        
        # Get disease information
        disease_info = knowledge_base_service.get_disease_info(predicted_class)
        
        # Generate LLM report
        report_start = time.time()
        report = llm_service.generate_farmer_report(
            disease_name=predicted_class,
            symptoms=disease_info.get('symptoms', []),
            treatments=disease_info.get('treatments', []),
            confidence_score=confidence
        )
        report_time = (time.time() - report_start) * 1000
        
        # Convert LIME explanation to base64 for frontend
        lime_pil = Image.fromarray((lime_explanation * 255).astype('uint8'))
        buffer = io.BytesIO()
        lime_pil.save(buffer, format='PNG')
        lime_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Save prediction to database (skip if disabled)
        prediction_id = None
        if not settings.disable_db_operations:
            try:
                prediction_repo = PredictionRepository(db)
                prediction_data = {
                    'image_filename': filename,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_probabilities': class_probabilities,
                    'lime_explanation_path': explanation_path,
                    'processing_time_ms': processing_time
                }
                prediction = prediction_repo.create(prediction_data)
                prediction_id = prediction.id
                
                # Save LLM report to database (skip if disabled)
                llm_report_data = {
                    'prediction_id': prediction_id,
                    'disease_name': predicted_class,
                    'confidence_score': confidence,
                    'report_content': report,
                    'llm_provider': 'gemini',
                    'generation_time_ms': report_time
                }
                
                llm_repo = LLMReportRepository(db)
                llm_repo.create(llm_report_data)
            except Exception as e:
                logger.warning(f"Failed to save prediction/report to database: {str(e)}")
        
        logger.info(f"Prediction completed: {predicted_class} ({confidence:.3f}) in {processing_time:.0f}ms")
        
        # Clean up memory
        import gc
        import torch
        del lime_pil, buffer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "lime_explanation": lime_base64,
            "disease_info": disease_info,
            "report": report,
            "processing_time_ms": processing_time,
            "prediction_id": prediction_id
        }
        
    except InvalidImageError as e:
        logger.error(f"Invalid image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except FileUploadError as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except PredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except LLMServiceError as e:
        logger.error(f"LLM service error: {str(e)}")
        # Don't fail the whole request if LLM fails
        return {
            "success": True,
            "prediction": predicted_class if 'predicted_class' in locals() else "Unknown",
            "confidence": confidence if 'confidence' in locals() else 0.0,
            "lime_explanation": lime_base64 if 'lime_base64' in locals() else "",
            "disease_info": disease_info if 'disease_info' in locals() else {},
            "report": "Report generation failed. Please try again later.",
            "processing_time_ms": processing_time if 'processing_time' in locals() else 0,
            "warning": "LLM report generation failed"
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stream")
async def predict_disease_stream(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Predict mango leaf disease with streaming results.
    Sends prediction instantly, then LLM report, then LIME explanation.
    """
    async def event_generator():
        # Pre-prediction cleanup - clear all existing data
        try:
            import gc
            import torch
            
            # Clear any existing cached data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear old temp files
            file_service.cleanup_temp_files(max_age_hours=0)  # Clean all temp files
            
            logger.info("Pre-prediction cleanup completed")
        except Exception as e:
            logger.warning(f"Pre-cleanup failed: {str(e)}")
        
        start_time = time.time()
        filename = None
        predicted_class = None
        confidence = 0.0
        prediction_id = None

        try:
            # 1. Prediction Phase (FAST)
            file_path, filename = file_service.save_uploaded_file(file)
            predicted_class, confidence, class_probabilities = get_inference_pipeline().predict(file_path)
            disease_info = knowledge_base_service.get_disease_info(predicted_class)
            
            # Save basic prediction to DB (skip if disabled)
            prediction_id = None
            if not settings.disable_db_operations:
                try:
                    prediction_repo = PredictionRepository(db)
                    prediction = prediction_repo.create({
                        'image_filename': filename,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'class_probabilities': class_probabilities,
                        'processing_time_ms': (time.time() - start_time) * 1000
                    })
                    prediction_id = prediction.id
                except Exception as e:
                    logger.warning(f"Failed to save prediction to DB: {str(e)}")

            yield json.dumps({
                "type": "prediction",
                "prediction": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "disease_info": disease_info,
                "prediction_id": prediction_id
            }) + "\n"

            # 2. LLM Report Phase (MEDIUM)
            try:
                report_start = time.time()
                # Run sync LLM call in a thread pool to avoid blocking async loop
                report = await run_in_threadpool(
                    llm_service.generate_farmer_report,
                    disease_name=predicted_class,
                    symptoms=disease_info.get('symptoms', []),
                    treatments=disease_info.get('treatments', []),
                    confidence_score=confidence
                )
                report_time = (time.time() - report_start) * 1000

                # Save report to DB (skip if disabled)
                if prediction_id and not settings.disable_db_operations:
                    try:
                        llm_repo = LLMReportRepository(db)
                        llm_repo.create({
                            'prediction_id': prediction_id,
                            'disease_name': predicted_class,
                            'confidence_score': confidence,
                            'report_content': report,
                            'llm_provider': 'gemini',
                            'generation_time_ms': report_time
                        })
                    except Exception as e:
                        logger.warning(f"Failed to save streaming report to DB: {str(e)}")

                yield json.dumps({
                    "type": "report",
                    "report": report
                }) + "\n"
            except Exception as e:
                logger.error(f"LLM report phase failed: {str(e)}")
                yield json.dumps({"type": "report", "report": "Expert analysis currently unavailable."}) + "\n"

            # 3. XAI/LIME Phase (SLOW)
            try:
                # Run sync LIME call in a thread pool
                explainer = get_lime_explainer()
                explanation_class, lime_explanation = await run_in_threadpool(
                    explainer.explain_image,
                    image_path=file_path
                )
                
                explanation_path = file_service.save_explanation_image(
                    lime_explanation, f"lime_{filename}"
                )

                # Update prediction with LIME path
                try:
                    if prediction_id:
                        prediction.lime_explanation_path = explanation_path
                        db.commit()
                except Exception as e:
                    logger.warning(f"Failed to update LIME path in DB: {str(e)}")

                # Convert to base64
                lime_pil = Image.fromarray((lime_explanation * 255).astype('uint8'))
                buffer = io.BytesIO()
                lime_pil.save(buffer, format='PNG')
                lime_base64 = base64.b64encode(buffer.getvalue()).decode()

                yield json.dumps({
                    "type": "explanation",
                    "lime_explanation": lime_base64,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }) + "\n"
                
                # Clean up memory immediately after LIME
                import gc
                import torch
                del lime_explanation, lime_pil, buffer, explainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"LIME explanation phase failed: {str(e)}")
                yield json.dumps({"type": "explanation", "error": "Explanation generation failed."}) + "\n"

        except Exception as e:
            logger.error(f"Streaming prediction failed: {str(e)}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        
        finally:
            # Clean up uploaded file and resources
            try:
                if 'file_path' in locals() and file_path:
                    file_service.delete_file(file_path)
                # Final memory cleanup
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Memory cleanup completed after prediction")
            except Exception as e:
                logger.warning(f"Cleanup failed: {str(e)}")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.get("/history")
async def get_prediction_history(
    limit: int = 50,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get recent prediction history"""
    try:
        prediction_repo = PredictionRepository(db)
        predictions = prediction_repo.get_recent(limit)
        
        return {
            "success": True,
            "predictions": [
                {
                    "id": pred.id,
                    "image_filename": pred.image_filename,
                    "predicted_class": pred.predicted_class,
                    "confidence": pred.confidence,
                    "created_at": pred.created_at.isoformat(),
                    "processing_time_ms": pred.processing_time_ms
                }
                for pred in predictions
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get prediction history")


@router.get("/statistics")
async def get_prediction_statistics(
    days: int = 30,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get prediction statistics"""
    try:
        prediction_repo = PredictionRepository(db)
        stats = prediction_repo.get_statistics(days)
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get prediction statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get prediction statistics")


@router.get("/batch")
async def batch_predict(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Batch prediction for multiple images"""
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    start_time = time.time()
    
    for file in files:
        try:
            # Save uploaded file
            file_path, filename = file_service.save_uploaded_file(file)
            
            # Make prediction
            predicted_class, confidence, class_probabilities = get_inference_pipeline().predict(file_path)
            
            # Get disease information
            disease_info = knowledge_base_service.get_disease_info(predicted_class)
            
            results.append({
                "filename": filename,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "disease_info": disease_info,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "success": True,
        "results": results,
        "total_files": len(files),
        "successful_predictions": sum(1 for r in results if r.get("success")),
        "processing_time_ms": processing_time
    }
