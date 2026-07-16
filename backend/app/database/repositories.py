from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, timedelta
from app.database.models import Prediction, ModelVersion, TrainingRun, DiseaseInfo, SystemMetrics, LLMReport
from app.utils.exceptions import DatabaseError
from app.utils.logger import logger


class LLMReportRepository:
    """Repository for LLM report operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, report_data: Dict[str, Any]) -> LLMReport:
        """Create a new LLM report"""
        try:
            report = LLMReport(**report_data)
            self.db.add(report)
            self.db.commit()
            self.db.refresh(report)
            return report
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create LLM report: {str(e)}")
            raise DatabaseError(f"Failed to create LLM report: {str(e)}")


class TrainingRunRepository:
    """Repository for training run operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, training_data: Dict[str, Any]) -> TrainingRun:
        """Create a new training run"""
        try:
            training_run = TrainingRun(**training_data)
            self.db.add(training_run)
            self.db.commit()
            self.db.refresh(training_run)
            return training_run
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create training run: {str(e)}")
            raise DatabaseError(f"Failed to create training run: {str(e)}")
    
    def get_by_id(self, run_id: int) -> Optional[TrainingRun]:
        """Get training run by ID"""
        try:
            return self.db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        except Exception as e:
            logger.error(f"Failed to get training run {run_id}: {str(e)}")
            raise DatabaseError(f"Failed to get training run: {str(e)}")
    
    def get_recent(self, limit: int = 50) -> List[TrainingRun]:
        """Get recent training runs"""
        try:
            return self.db.query(TrainingRun).order_by(desc(TrainingRun.created_at)).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to get recent training runs: {str(e)}")
            raise DatabaseError(f"Failed to get recent training runs: {str(e)}")
    
    def update_status(self, run_id: int, status: str, error_message: str = None) -> bool:
        """Update training run status"""
        try:
            training_run = self.db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if training_run:
                training_run.status = status
                if error_message:
                    training_run.error_message = error_message
                if status in ["completed", "failed", "stopped"]:
                    training_run.completed_at = datetime.utcnow()
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update training run status: {str(e)}")
            raise DatabaseError(f"Failed to update training run status: {str(e)}")


class SystemMetricsRepository:
    """Repository for system metrics operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, metric_data: Dict[str, Any]) -> SystemMetrics:
        """Create a system metric record"""
        try:
            metric = SystemMetrics(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create system metric: {str(e)}")
            raise DatabaseError(f"Failed to create system metric: {str(e)}")
    
    def get_recent_metrics(self, metric_type: str, hours: int = 24) -> List[SystemMetrics]:
        """Get recent metrics of a specific type"""
        try:
            since_time = datetime.utcnow() - timedelta(hours=hours)
            return self.db.query(SystemMetrics).filter(
                SystemMetrics.metric_type == metric_type,
                SystemMetrics.created_at >= since_time
            ).order_by(desc(SystemMetrics.created_at)).all()
        except Exception as e:
            logger.error(f"Failed to get recent metrics {metric_type}: {str(e)}")
            raise DatabaseError(f"Failed to get recent metrics: {str(e)}")


class PredictionRepository:
    """Repository for prediction operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Create a new prediction record"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"Created prediction record: {prediction.id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create prediction: {str(e)}")
            raise DatabaseError(f"Failed to create prediction: {str(e)}")
    
    def get_by_id(self, prediction_id: int) -> Optional[Prediction]:
        """Get prediction by ID"""
        try:
            return self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
        except Exception as e:
            logger.error(f"Failed to get prediction {prediction_id}: {str(e)}")
            raise DatabaseError(f"Failed to get prediction: {str(e)}")
    
    def get_recent(self, limit: int = 100) -> List[Prediction]:
        """Get recent predictions"""
        try:
            return self.db.query(Prediction).order_by(desc(Prediction.created_at)).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {str(e)}")
            raise DatabaseError(f"Failed to get recent predictions: {str(e)}")
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get prediction statistics for the last N days"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            total_predictions = self.db.query(Prediction).filter(
                Prediction.created_at >= since_date
            ).count()
            
            class_distribution = self.db.query(
                Prediction.predicted_class,
                func.count(Prediction.id).label('count')
            ).filter(
                Prediction.created_at >= since_date
            ).group_by(Prediction.predicted_class).all()
            
            avg_confidence = self.db.query(
                func.avg(Prediction.confidence)
            ).filter(
                Prediction.created_at >= since_date
            ).scalar()
            
            return {
                'total_predictions': total_predictions,
                'class_distribution': dict(class_distribution),
                'average_confidence': float(avg_confidence) if avg_confidence else 0.0,
                'period_days': days
            }
        except Exception as e:
            logger.error(f"Failed to get prediction statistics: {str(e)}")
            raise DatabaseError(f"Failed to get prediction statistics: {str(e)}")


class ModelVersionRepository:
    """Repository for model version operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, model_data: Dict[str, Any]) -> ModelVersion:
        """Create a new model version"""
        try:
            model_version = ModelVersion(**model_data)
            self.db.add(model_version)
            self.db.commit()
            self.db.refresh(model_version)
            logger.info(f"Created model version: {model_version.version}")
            return model_version
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create model version: {str(e)}")
            raise DatabaseError(f"Failed to create model version: {str(e)}")
    
    def get_active(self) -> Optional[ModelVersion]:
        """Get the active model version"""
        try:
            return self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        except Exception as e:
            logger.error(f"Failed to get active model: {str(e)}")
            raise DatabaseError(f"Failed to get active model: {str(e)}")
    
    def set_active(self, version: str) -> bool:
        """Set a model version as active"""
        try:
            # Deactivate all models
            self.db.query(ModelVersion).update({ModelVersion.is_active: False})
            
            # Activate the specified model
            model = self.db.query(ModelVersion).filter(ModelVersion.version == version).first()
            if model:
                model.is_active = True
                model.deployed_at = datetime.utcnow()
                self.db.commit()
                logger.info(f"Activated model version: {version}")
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to set active model {version}: {str(e)}")
            raise DatabaseError(f"Failed to set active model: {str(e)}")
    
    def get_all(self) -> List[ModelVersion]:
        """Get all model versions"""
        try:
            return self.db.query(ModelVersion).order_by(desc(ModelVersion.created_at)).all()
        except Exception as e:
            logger.error(f"Failed to get model versions: {str(e)}")
            raise DatabaseError(f"Failed to get model versions: {str(e)}")


class DiseaseInfoRepository:
    """Repository for disease information operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, disease_data: Dict[str, Any]) -> DiseaseInfo:
        """Create disease information"""
        try:
            disease = DiseaseInfo(**disease_data)
            self.db.add(disease)
            self.db.commit()
            self.db.refresh(disease)
            logger.info(f"Created disease info: {disease.disease_name}")
            return disease
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create disease info: {str(e)}")
            raise DatabaseError(f"Failed to create disease info: {str(e)}")
    
    def get_by_name(self, disease_name: str) -> Optional[DiseaseInfo]:
        """Get disease information by name"""
        try:
            return self.db.query(DiseaseInfo).filter(DiseaseInfo.disease_name == disease_name).first()
        except Exception as e:
            logger.error(f"Failed to get disease info {disease_name}: {str(e)}")
            raise DatabaseError(f"Failed to get disease info: {str(e)}")
    
    def get_all(self) -> List[DiseaseInfo]:
        """Get all disease information"""
        try:
            return self.db.query(DiseaseInfo).all()
        except Exception as e:
            logger.error(f"Failed to get all disease info: {str(e)}")
            raise DatabaseError(f"Failed to get all disease info: {str(e)}")
    
    def initialize_default_diseases(self) -> None:
        """Initialize default disease information"""
        default_diseases = [
            {
                "disease_name": "Healthy",
                "symptoms": ["No visible spots or discoloration", "Uniform green color"],
                "treatments": ["Maintain good hygiene", "Regular pruning"],
                "description": "Healthy mango leaf with no signs of disease",
                "severity": "low"
            },
            {
                "disease_name": "Anthracnose",
                "symptoms": ["Dark sunken lesions on leaves", "Black spores under humid conditions"],
                "treatments": ["Apply copper-based fungicides", "Remove infected parts"],
                "description": "Fungal disease causing dark lesions and leaf spots",
                "severity": "high"
            },
            {
                "disease_name": "Bacterial Canker",
                "symptoms": ["Water-soaked spots", "Yellow halos around lesions"],
                "treatments": ["Use bactericides like streptomycin", "Avoid overhead watering"],
                "description": "Bacterial infection causing water-soaked lesions",
                "severity": "medium"
            },
            {
                "disease_name": "Cutting Weevil Damage",
                "symptoms": ["Irregular cuts along leaf margins", "Wilting appearance"],
                "treatments": ["Manual removal of weevils", "Neem oil application"],
                "description": "Insect damage from weevil feeding activity",
                "severity": "low"
            },
            {
                "disease_name": "Die Back",
                "symptoms": ["Branch tips dying back", "Gradual decline in canopy health"],
                "treatments": ["Prune dead branches", "Improve soil nutrition"],
                "description": "Progressive death of branch tips and twigs",
                "severity": "medium"
            }
        ]
        
        for disease_data in default_diseases:
            existing = self.get_by_name(disease_data["disease_name"])
            if not existing:
                self.create(disease_data)


class SystemMetricsRepository:
    """Repository for system metrics operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, metric_data: Dict[str, Any]) -> SystemMetrics:
        """Create a system metric record"""
        try:
            metric = SystemMetrics(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create system metric: {str(e)}")
            raise DatabaseError(f"Failed to create system metric: {str(e)}")
    
    def get_recent_metrics(self, metric_type: str, hours: int = 24) -> List[SystemMetrics]:
        """Get recent metrics of a specific type"""
        try:
            since_time = datetime.utcnow() - timedelta(hours=hours)
            return self.db.query(SystemMetrics).filter(
                SystemMetrics.metric_type == metric_type,
                SystemMetrics.created_at >= since_time
            ).order_by(desc(SystemMetrics.created_at)).all()
        except Exception as e:
            logger.error(f"Failed to get recent metrics {metric_type}: {str(e)}")
            raise DatabaseError(f"Failed to get recent metrics: {str(e)}")
