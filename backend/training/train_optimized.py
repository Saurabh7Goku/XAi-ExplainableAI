import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.model import ViT
from app.config import settings

class OptimizedConfig:
    # ViT-Tiny Configuration (~23MB)
    EMBED_DIM = 192
    DEPTH = 12
    NUM_HEADS = 3
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH = 'models/vit_mango_tiny.pth'

def main():
    print(f"Initializing Optimized Training (ViT-Tiny)...")
    print(f"Device: {OptimizedConfig.DEVICE}")
    
    # Initialize tiny model
    model = ViT(
        img_size=settings.img_size,
        patch_size=settings.patch_size,
        num_classes=settings.num_classes,
        embed_dim=OptimizedConfig.EMBED_DIM,
        depth=OptimizedConfig.DEPTH,
        num_heads=OptimizedConfig.NUM_HEADS
    ).to(OptimizedConfig.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (approx. {total_params*4 / (1024*1024):.2f} MB as float32)")
    
    # Note: Dataset loading code would go here (omitted for brevity, user should use their notebook data)
    print("\nTo use this optimized architecture, update your notebook model initialization to:")
    print(f"model = ViT(embed_dim={OptimizedConfig.EMBED_DIM}, depth={OptimizedConfig.DEPTH}, num_heads={OptimizedConfig.NUM_HEADS})")
    
    # Demonstrate Quantization
    print("\nApplying Post-Training Quantization Demo...")
    model.eval().cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(quantized_model.state_dict(), OptimizedConfig.SAVE_PATH + ".quantized")
    print(f"Quantized weights saved to {OptimizedConfig.SAVE_PATH}.quantized")
    print(f"Final estimated size: < 10 MB")

if __name__ == "__main__":
    main()
