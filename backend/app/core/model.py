import os
import torch
import torch.nn as nn
from app.config import settings, ModelConfig
from app.utils.exceptions import ModelLoadError


class PatchEmbedding(nn.Module):
    """Patch embedding layer for ViT"""
    
    def __init__(self, img_size=settings.img_size, patch_size=settings.patch_size, 
                 in_channels=3, embed_dim=settings.embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # [B, embed_dim, n_patches_sqrt, n_patches_sqrt]
        x = x.flatten(2)        # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)   # [B, n_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, dim, n_heads=settings.num_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block (Matches Kaggle Implementation)"""
    
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        
        # Using nn.Sequential to match the weight names (blocks.X.mlp.0, blocks.X.mlp.2)
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
    """Vision Transformer for Mango Leaf Disease Detection"""
    
    def __init__(self, img_size=settings.img_size, patch_size=settings.patch_size, 
                 in_chans=3, num_classes=settings.num_classes, embed_dim=settings.embed_dim, 
                 depth=settings.depth, num_heads=settings.num_heads, mlp_ratio=4., 
                 qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=embed_dim, n_heads=num_heads, mlp_ratio=mlp_ratio, 
                           qkv_bias=qkv_bias, p=p, attn_p=attn_p)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        self.apply(_init_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)


def get_model_summary(model: ViT) -> str:
    """Get model summary for logging"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
    Model Summary:
    - Total parameters: {total_params:,}
    - Trainable parameters: {trainable_params:,}
    - Image size: {settings.img_size}x{settings.img_size}
    - Patch size: {settings.patch_size}x{settings.patch_size}
    - Number of patches: {(settings.img_size // settings.patch_size) ** 2}
    - Embedding dimension: {settings.embed_dim}
    - Number of classes: {settings.num_classes}
    """
    
    return summary
