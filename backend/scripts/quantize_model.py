import torch
import os
import sys

# Add backend to path to import model architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.model import ViT
from app.config import settings

def quantize_model(input_path, output_path):
    print(f"Loading model from {input_path}...")
    
    # Load original model
    checkpoint = torch.load(input_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Initialize model architecture
    model = ViT(
        img_size=settings.img_size,
        patch_size=settings.patch_size,
        num_classes=settings.num_classes,
        embed_dim=settings.embed_dim,
        depth=settings.depth,
        num_heads=settings.num_heads
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Applying dynamic quantization...")
    # Apply quantization to linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Save quantized model
    print(f"Saving quantized model to {output_path}...")
    
    # Prepare metadata
    quantized_checkpoint = checkpoint.copy()
    quantized_checkpoint['model_state_dict'] = quantized_model.state_dict()
    quantized_checkpoint['quantized'] = True
    
    torch.save(quantized_checkpoint, output_path)
    
    # Compare sizes
    old_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nOptimization Results:")
    print(f"Original Size: {old_size:.2f} MB")
    print(f"Quantized Size: {new_size:.2f} MB")
    print(f"Reduction: {((old_size - new_size) / old_size) * 100:.2f}%")

if __name__ == "__main__":
    input_model = "models/vit_mango.pth"
    output_model = "models/vit_mango_quantized.pth"
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if os.path.exists(input_model):
        quantize_model(input_model, output_model)
    else:
        print(f"Error: {input_model} not found.")
