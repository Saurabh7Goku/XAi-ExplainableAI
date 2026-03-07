import os
import torch
from typing import Optional

from app.core.model import ViT
from app.config import settings
from app.utils.logger import logger
from app.utils.exceptions import PredictionError
from huggingface_hub import hf_hub_download


class ModelSelector:
    """Selector for loading ViT models"""
    
    @staticmethod
    def load_model(model_path: str, device: str = None):
        """
        Load ViT model from path
        
        Args:
            model_path: Path to model file
            device: Device to load model on
            
        Returns:
            Tuple of (model, class_to_idx, idx_to_class)
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Download from Hugging Face if hf_repo_id is set and model doesn't exist locally
        if settings.hf_repo_id and not os.path.exists(model_path):
            try:
                logger.info(f"Downloading model {settings.hf_filename} from Hugging Face repo {settings.hf_repo_id}...")
                model_path = hf_hub_download(
                    repo_id=settings.hf_repo_id,
                    filename=settings.hf_filename,
                    local_dir=os.path.dirname(model_path) if os.path.dirname(model_path) else None,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                logger.error(f"Failed to download model from Hugging Face: {str(e)}")
                # Fallback to local path check later
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            logger.info(f"Loading ViT model from {model_path}")
            
            # Load ViT configuration from checkpoint or defaults
            num_classes = checkpoint.get('num_classes', 8)
            img_size = checkpoint.get('img_size', 224)
            patch_size = checkpoint.get('patch_size', 16)
            embed_dim = checkpoint.get('embed_dim', 768)
            depth = checkpoint.get('depth', 12)
            num_heads = checkpoint.get('num_heads', 12)
            
            model = ViT(
                num_classes=num_classes,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads
            ).to(device)
            
            # Apply quantization to model structure if checkpoint is quantized
            if checkpoint.get('quantized', False):
                logger.info("Applying dynamic quantization to model structure for loading...")
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            class_to_idx = checkpoint.get('class_to_idx')
            idx_to_class = checkpoint.get('idx_to_class')
            
            return model, class_to_idx, idx_to_class
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise PredictionError(f"Model loading failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise PredictionError(f"Model loading failed: {str(e)}")
    
    @staticmethod
    def get_model_info(model_path: str) -> dict:
        """Get information about saved model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            return {
                'architecture': checkpoint.get('architecture', 'unknown'),
                'num_classes': checkpoint.get('num_classes', 0),
                'img_size': checkpoint.get('img_size', 224),
                'accuracy': checkpoint.get('accuracy', 0.0),
                'epoch': checkpoint.get('epoch', 0),
                'device': checkpoint.get('device', 'cpu')
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {'error': str(e)}
