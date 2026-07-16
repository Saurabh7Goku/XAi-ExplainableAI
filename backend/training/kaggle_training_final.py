import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm.auto import tqdm

# --- CONFIGURATION ---
class Config:
    DATA_DIR = '/kaggle/input/mango-leaf-disease-dataset/MangoLeaf'
    IMG_SIZE = 224
    PATCH_SIZE = 16
    
    # --- OPTIMIZED ARCHITECTURE (ViT-Tiny) ---
    EMBED_DIM = 192    # Original was 768
    DEPTH = 12
    NUM_HEADS = 3      # Original was 12
    NUM_CLASSES = 8
    
    # --- HYPERPARAMETERS ---
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    VALIDATION_SPLIT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH = 'vit_mango_tiny.pth'
    QUANTIZED_PATH = 'vit_mango_quantized.pth'

# --- MODEL ARCHITECTURE ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=3, qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=8, 
                 embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks: x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# --- DATA PREPARATION ---
class LeafDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "filepaths"]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.class_to_idx[self.df.loc[idx, "labels"]]

def prepare_data():
    filepaths, labels = [], []
    for fold in sorted(os.listdir(Config.DATA_DIR)):
        foldpath = os.path.join(Config.DATA_DIR, fold)
        if os.path.isdir(foldpath):
            for file in os.listdir(foldpath):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(foldpath, file))
                    labels.append(fold)
    df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
    classes = sorted(df["labels"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    train_df, val_df = train_test_split(df, test_size=Config.VALIDATION_SPLIT, random_state=123)
    tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return (DataLoader(LeafDataset(train_df, class_to_idx, tf), batch_size=Config.BATCH_SIZE, shuffle=True),
            DataLoader(LeafDataset(val_df, class_to_idx, tf), batch_size=Config.BATCH_SIZE),
            class_to_idx)

# --- TRAINING LOOP ---
def train():
    train_loader, val_loader, class_to_idx = prepare_data()
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    model = ViT(embed_dim=Config.EMBED_DIM, num_heads=Config.NUM_HEADS).to(Config.DEVICE)
    opt = optim.Adamax(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            opt.zero_grad(); crit(model(images), labels).backward(); opt.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                _, pred = model(images).max(1)
                total += labels.size(0); correct += pred.eq(labels).sum().item()
        acc = 100 * correct / total
        print(f"Val Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'num_classes': Config.NUM_CLASSES,
                'embed_dim': Config.EMBED_DIM,
                'num_heads': Config.NUM_HEADS,
                'img_size': Config.IMG_SIZE,
                'accuracy': acc
            }, Config.SAVE_PATH)

    # --- QUANTIZATION ---
    print("\nTraining Done! Applying Quantization...")
    checkpoint = torch.load(Config.SAVE_PATH, map_location='cpu')
    model_cpu = ViT(embed_dim=Config.EMBED_DIM, num_heads=Config.NUM_HEADS).cpu()
    model_cpu.load_state_dict(checkpoint['model_state_dict'])
    quantized_model = torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    
    checkpoint['model_state_dict'] = quantized_model.state_dict()
    checkpoint['quantized'] = True
    torch.save(checkpoint, Config.QUANTIZED_PATH)
    print(f"Optimized model saved as: {Config.QUANTIZED_PATH}")
    print(f"Original size: {os.path.getsize(Config.SAVE_PATH)/1024/1024:.2f}MB")
    print(f"Quantized size: {os.path.getsize(Config.QUANTIZED_PATH)/1024/1024:.2f}MB")

if __name__ == "__main__": train()
