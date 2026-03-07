from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.sql import func
from app.database.database import Base
from app.config import ModelConfig


class Prediction(Base):
    """Model for storing prediction results"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String(255), nullable=False)
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(JSON, nullable=True)  # Store all class probabilities
    lime_explanation_path = Column(String(500), nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, class={self.predicted_class}, confidence={self.confidence:.3f})>"


class ModelVersion(Base):
    """Model for storing model version information"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False)
    model_path = Column(String(500), nullable=False)
    architecture = Column(String(100), nullable=False)  # e.g., "ViT-B/16"
    num_parameters = Column(Integer, nullable=False)
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    training_loss = Column(Float, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<ModelVersion(id={self.id}, version={self.version}, active={self.is_active})>"


class TrainingRun(Base):
    """Model for storing training run information"""
    __tablename__ = "training_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, nullable=False)
    run_name = Column(String(200), nullable=False)
    epochs_completed = Column(Integer, nullable=False)
    total_epochs = Column(Integer, nullable=False)
    final_training_loss = Column(Float, nullable=True)
    final_validation_loss = Column(Float, nullable=True)
    best_validation_accuracy = Column(Float, nullable=True)
    training_time_minutes = Column(Float, nullable=True)
    dataset_info = Column(JSON, nullable=True)  # dataset size, splits, etc.
    metrics_history = Column(JSON, nullable=True)  # training/validation metrics over time
    status = Column(String(50), default="running")  # running, completed, failed
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<TrainingRun(id={self.id}, status={self.status}, epochs={self.epochs_completed}/{self.total_epochs})>"


class DiseaseInfo(Base):
    """Model for storing disease information"""
    __tablename__ = "disease_info"
    
    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String(100), unique=True, nullable=False)
    symptoms = Column(JSON, nullable=False)  # List of symptoms
    treatments = Column(JSON, nullable=False)  # List of treatments
    description = Column(Text, nullable=True)
    severity = Column(String(20), nullable=True)  # low, medium, high
    prevention_methods = Column(JSON, nullable=True)  # List of prevention methods
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<DiseaseInfo(id={self.id}, name={self.disease_name})>"


class SystemMetrics(Base):
    """Model for storing system performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String(50), nullable=False)  # prediction_time, memory_usage, etc.
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # ms, MB, etc.
    additional_info = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, type={self.metric_type}, value={self.metric_value})>"


class LLMReport(Base):
    """Model for storing LLM generated reports"""
    __tablename__ = "llm_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=False)
    disease_name = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    report_content = Column(Text, nullable=False)
    llm_provider = Column(String(50), nullable=False)  # gemini, claude, etc.
    prompt_used = Column(Text, nullable=True)
    generation_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<LLMReport(id={self.id}, disease={self.disease_name}, provider={self.llm_provider})>"
