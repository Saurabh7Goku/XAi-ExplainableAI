import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict, Any
from app.config import settings, ModelConfig
from app.core.model import ViT
from app.core.model_selector import ModelSelector
from app.utils.exceptions import PredictionError, InvalidImageError
from app.utils.logger import logger


class InferencePipeline:
    """Pipeline for model inference"""
    
    def __init__(self, model_path: str = settings.model_path, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if local path exists or HF repo is specified
        if (model_path and os.path.exists(model_path)) or settings.hf_repo_id:
            try:
                model, self.class_to_idx, self.idx_to_class = ModelSelector.load_model(model_path, self.device)
                self.model = model
                logger.info(f"Model loaded successfully (from local or HF)")
            except Exception as e:
                logger.warning(f"Failed to load model: {str(e)}. Falling back to untrained model.")
                self._load_default_model()
        else:
            self._load_default_model()
        self.transform = self._get_transform()
        
        # Ensure labels match the model's number of classes
        self.labels = [self.idx_to_class.get(i, f"Class_{i}") for i in range(len(self.idx_to_class))]
        
    def _load_default_model(self):
        """Load default empty model structure"""
        self.model = ViT(
            num_classes=settings.num_classes,
            img_size=ModelConfig.IMG_SIZE,
            patch_size=ModelConfig.PATCH_SIZE,
            embed_dim=ModelConfig.EMBED_DIM,
            depth=ModelConfig.DEPTH,
            num_heads=ModelConfig.NUM_HEADS
        ).to(self.device)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(ModelConfig.CLASS_NAMES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        logger.info("Default/Untrained ViT model initialized as fallback")

    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((settings.img_size, settings.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return tensor.to(self.device)
        except Exception as e:
            raise InvalidImageError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Make prediction on image
        
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities)
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image_path)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
                
                # Convert to numpy
                probabilities_np = probabilities.cpu().numpy()[0]
                confidence_val = confidence.item()
                predicted_idx_val = predicted_idx.item()
                
                # Get class name
                predicted_class = self.idx_to_class.get(predicted_idx_val, "Unknown")
                
                # Create class probabilities dictionary
                class_probs = {
                    self.idx_to_class.get(i, f"Class_{i}"): float(probabilities_np[i]) 
                    for i in range(len(self.labels))
                }
                
                logger.info(f"Prediction: {predicted_class} with confidence {confidence_val:.3f}")
                
                return predicted_class, confidence_val, class_probs
                
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, image_paths: list) -> list:
        """Make predictions on multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                predicted_class, confidence, class_probs = self.predict(image_path)
                results.append({
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_probabilities': class_probs,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Failed to predict {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        return results


# Global inference pipeline instance (Lazy loaded)
_inference_pipeline = None

def get_inference_pipeline() -> InferencePipeline:
    """Get or initialize the inference pipeline (Singleton)"""
    global _inference_pipeline
    if _inference_pipeline is None:
        # Optimization for Render/Small instances
        import torch
        torch.set_num_threads(1)
        
        logger.info("Initializing InferencePipeline (Lazy Loading)...")
        _inference_pipeline = InferencePipeline()
    return _inference_pipeline

def is_pipeline_initialized() -> bool:
    """Check if the pipeline is already loaded without triggering a load"""
    return _inference_pipeline is not None
