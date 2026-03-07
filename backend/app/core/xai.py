"""
Explainable AI (XAI) module using LIME for model explanations
"""

import torch
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries, slic
import lime.lime_image as lime_img
from typing import Tuple, Any
from app.config import settings, ModelConfig
from app.core.model import ViT
from app.core.inference import get_inference_pipeline
from app.utils.exceptions import PredictionError
from app.utils.logger import logger


class LimeExplainer:
    """LIME explainer for ViT model"""
    
    def __init__(self, model: ViT = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use provided model or the one from the shared inference pipeline
        self.inference = get_inference_pipeline()
        self.model = model or self.inference.model
        self.model.eval()
        
        # Initialize LIME explainer
        self.explainer = lime_img.LimeImageExplainer()
        
        logger.info("LIME explainer initialized with shared model instance")
    
    def _predict_proba_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME
        
        Args:
            images: Batch of images as numpy arrays
            
        Returns:
            Probabilities as numpy array
        """
        try:
            logger.debug(f"Processing LIME batch of {len(images)} samples...")
            # Convert numpy arrays to PIL Images and apply transforms
            images_tensor = []
            for img in images:
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                tensor = self.inference.transform(pil_img)
                images_tensor.append(tensor)
            
            # Stack and move to device
            batch_tensor = torch.stack(images_tensor).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)
                
            return probabilities.cpu().numpy()
            
        except Exception as e:
            logger.error(f"LIME prediction function failed: {str(e)}")
            raise PredictionError(f"LIME prediction failed: {str(e)}")
    
    def explain_image(self, image_path: str, target_class: int = None) -> Tuple[str, np.ndarray]:
        """
        Generate LIME explanation for an image
        
        Args:
            image_path: Path to the image
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Tuple of (predicted_class, explanation_image)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_resized = image.resize((settings.img_size, settings.img_size))
            image_np = np.array(image_resized) / 255.0  # Normalize to [0, 1]
            
            # Get prediction if target_class not specified
            if target_class is None:
                predicted_class, _, _ = self.inference.predict(image_path)
                target_class = ModelConfig.CLASS_NAMES.index(predicted_class)
            
            # Use SLIC for more "leaf-aware" segmentation
            # It handles uniform backgrounds (like white) much better by grouping them
            segmenter = lambda x: slic(x, n_segments=100, compactness=10, sigma=1)
            
            explanation = self.explainer.explain_instance(
                image_np,
                self._predict_proba_fn,
                top_labels=1,
                hide_color=0,
                num_samples=ModelConfig.LIME_NUM_SAMPLES,
                segmentation_fn=segmenter
            )
            
            # Get explanation image and mask
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=ModelConfig.LIME_NUM_FEATURES,
                hide_rest=False
            )
            
            # Create explanation visualization
            lime_explanation = mark_boundaries(temp / 2 + 0.5, mask)
            
            # Get predicted class name
            predicted_class_idx = explanation.top_labels[0]
            predicted_class_name = ModelConfig.CLASS_NAMES[predicted_class_idx]
            
            logger.info(f"LIME explanation generated for class: {predicted_class_name}")
            
            return predicted_class_name, lime_explanation
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            raise PredictionError(f"LIME explanation failed: {str(e)}")
    
    def explain_batch(self, image_paths: list) -> list:
        """Generate explanations for multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                predicted_class, explanation = self.explain_image(image_path)
                results.append({
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'explanation': explanation,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Failed to explain {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        return results


# Global LIME explainer instance (Lazy loaded)
_lime_explainer = None

def get_lime_explainer() -> LimeExplainer:
    """Get or initialize the LIME explainer (Singleton)"""
    global _lime_explainer
    if _lime_explainer is None:
        logger.info("Initializing LimeExplainer (Lazy Loading)...")
        _lime_explainer = LimeExplainer()
    return _lime_explainer
