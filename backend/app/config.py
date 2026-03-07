import os
from typing import Optional, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),
        extra="ignore"
    )
    
    # Application
    app_name: str = "Mango Leaf Disease Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    # Database
    database_url: Optional[str] = "sqlite:///./mango_leaf_db.sqlite"
    
    # Model Configuration
    model_path: str = "models/vit_mango_quantized.pth"
    hf_repo_id: Optional[str] = None  # e.g., "username/mango-leaf-vit"
    hf_filename: str = "vit_mango_quantized.pth"
    img_size: int = 224
    patch_size: int = 16
    num_classes: int = 8
    
    # ViT-Tiny Architecture (Optimized)
    embed_dim: int = 192   # Old: 768
    depth: int = 12
    num_heads: int = 3     # Old: 12
    
    # LLM Configuration
    gemini_api_key: Optional[str] = None
    
    # File Upload
    max_file_size: int = 5 * 1024 * 1024  # 5MB
    allowed_extensions: Any = [".jpg", ".jpeg", ".png", ".webp"]
    
    # CORS
    cors_origins: Any = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Logging
    log_level: str = "INFO"
    
    @field_validator("allowed_extensions", "cors_origins", mode="before")
    @classmethod
    def validate_lists(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v):
        if not v:
            print("Warning: Database URL not provided. Running without persistent storage.")
        return v
    
    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def validate_gemini_key(cls, v):
        if not v:
            print("Warning: Gemini API key not provided. LLM reports will be disabled.")
        return v


# Global settings instance
settings = Settings()


# Model configuration class
class ModelConfig:
    """Model-specific configuration"""
    
    # Model architecture constants (ViT-Tiny)
    IMG_SIZE = 224
    PATCH_SIZE = 16
    EMBED_DIM = 192  # Old: 768
    DEPTH = 12
    NUM_HEADS = 3    # Old: 12
    
    # Class names for predictions
    CLASS_NAMES = [
        "Anthracnose",
        "Bacterial Canker",
        "Cutting Weevil",
        "Die Back",
        "Gall Midge",
        "Healthy",
        "Powdery Mildew",
        "Sooty Mould"
    ]
    
    # Training hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    
    # Data augmentation
    AUGMENTATION_PROB = 0.5
    
    # LIME configuration
    LIME_NUM_SAMPLES = 250
    LIME_NUM_FEATURES = 10
