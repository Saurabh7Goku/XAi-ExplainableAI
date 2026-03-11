"""
Explainable AI (XAI) module using VIT Attention Rollout
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Any, Dict
from app.core.inference import get_inference_pipeline
from app.utils.exceptions import PredictionError
from app.utils.logger import logger


class VITAttentionRollout:
    """Attention Rollout explainer for Vision Transformer"""

    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.hooks = []

        # Register hooks on attention layers
        # Handle both nn.Sequential and direct block access
        if hasattr(model.blocks, '__iter__'):
            for block in model.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'attn_drop'):
                    hook = block.attn.attn_drop.register_forward_hook(self.get_attention)
                    self.hooks.append(hook)
        
        logger.info(f"Registered hooks on {len(self.hooks)} attention layers")

    def get_attention(self, module, input, output):
        """Hook to capture attention weights"""
        # Extract attention weights from the output
        # The output should be [batch_size, num_heads, seq_len, seq_len]
        if len(output) > 0 and hasattr(output[0], 'shape'):
            self.attentions.append(output[0].detach())

    def rollout(self, input_tensor) -> np.ndarray:
        """Compute attention rollout mask"""
        self.attentions = []

        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attentions:
            logger.warning("No attention weights captured")
            return np.zeros((224, 224))

        # Start with identity matrix
        seq_len = self.attentions[0].size(-1)
        result = torch.eye(seq_len)

        for attn in self.attentions:
            # Average attention across heads: [batch, heads, seq, seq] -> [batch, seq, seq]
            if len(attn.shape) == 4:
                attn_heads_fused = attn.mean(dim=1)
            else:
                attn_heads_fused = attn  # Already averaged

            # Use first batch item
            attn_heads_fused = attn_heads_fused[0]

            # Add identity and normalize
            attn_heads_fused += torch.eye(seq_len)
            attn_heads_fused /= attn_heads_fused.sum(dim=-1, keepdim=True)

            # Matrix multiplication for rollout
            result = torch.matmul(attn_heads_fused, result)

        # Extract CLS token attention (excluding CLS token itself)
        if result.size(0) > 1:
            mask = result[0, 1:]  # Exclude CLS token
        else:
            mask = result[0]

        # Reshape to square grid
        patch_size = int(np.sqrt(mask.shape[0]))
        if patch_size * patch_size != mask.shape[0]:
            logger.warning(f"Cannot reshape mask of shape {mask.shape} to square")
            return np.zeros((224, 224))
            
        mask = mask.reshape(patch_size, patch_size).cpu().numpy()

        # Resize to original image size
        mask = cv2.resize(mask, (224, 224))

        # Normalize to [0, 1]
        if mask.max() > mask.min():
            mask = (mask - mask.min()) / (mask.max() - mask.min())
        else:
            mask = np.zeros_like(mask)

        return mask

    def generate_heatmap(self, input_tensor: torch.Tensor, original_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate attention heatmap overlay"""
        mask = self.rollout(input_tensor)

        # Resize original image to match mask size
        original_resized = cv2.resize(original_image, (224, 224))

        # Create heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * mask),
            cv2.COLORMAP_JET
        )

        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        overlay = heatmap * 0.4 + original_resized * 0.6
        overlay = overlay.astype(np.uint8)

        return overlay, mask


class AttentionExplainer:
    """Unified explainer using VIT Attention Rollout"""

    def __init__(self):
        self.inference = get_inference_pipeline()
        self.explainer = None

    def _get_explainer(self):
        """Lazy initialization of explainer"""
        if self.explainer is None:
            self.explainer = VITAttentionRollout(self.inference.model)
            logger.info("Attention Rollout explainer initialized")
        return self.explainer

    def explain_image(self, image_path: str) -> Dict[str, Any]:
        """
        Generate attention-based explanation for an image

        Args:
            image_path: Path to the image file

        Returns:
            Dict containing explanation image and mask
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_array = np.array(image)

            # Get preprocessed tensor from inference pipeline
            input_tensor = self.inference.preprocess_image(image_path)

            # Generate explanation
            explainer = self._get_explainer()
            explanation_image, mask = explainer.generate_heatmap(input_tensor, original_array)

            logger.info("Attention rollout explanation generated successfully")

            return {
                'image': explanation_image,
                'mask': mask,
                'method': 'attention_rollout'
            }

        except Exception as e:
            logger.error(f"Attention rollout explanation failed: {str(e)}")
            raise PredictionError(f"Failed to generate explanation: {str(e)}")


# Global explainer instance
_explainer_instance = None

def get_explainer() -> AttentionExplainer:
    """Get global explainer instance"""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = AttentionExplainer()
    return _explainer_instance
