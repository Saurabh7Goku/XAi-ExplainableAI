import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.model import ViT
from app.config import settings, ModelConfig
from app.utils.logger import logger
from app.utils.exceptions import PredictionError


class LeafDataset(torch.utils.data.Dataset):
    """Custom dataset for mango leaf images"""
    
    def __init__(self, dataframe, class_to_idx, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filepaths"]
        label = self.df.loc[idx, "labels"]

        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx


class TrainingPipeline:
    """Enhanced pipeline for training the ViT model"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.class_to_idx = None
        self.idx_to_class = None
        
        logger.info(f"Training pipeline initialized on device: {self.device}")
    
    def setup_data_loaders(self, data_dir: str, batch_size: int = 32, 
                          validation_split: float = 0.2, test_split: float = 0.4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Setup training, validation, and test data loaders
        """
        try:
            # Load dataset
            filepaths = []
            labels = []
            
            for fold in os.listdir(data_dir):
                foldpath = os.path.join(data_dir, fold)
                if os.path.isdir(foldpath):
                    for file in os.listdir(foldpath):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            filepaths.append(os.path.join(foldpath, file))
                            labels.append(fold)
            
            # Create DataFrame
            df = pd.DataFrame({
                "filepaths": filepaths,
                "labels": labels
            })
            
            # Get classes and mappings
            classes = sorted(df["labels"].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            
            # Split data (train/val/test)
            train_df, temp_df = train_test_split(
                df, test_size=validation_split + test_split, shuffle=True, random_state=123
            )
            
            valid_df, test_df = train_test_split(
                temp_df, test_size=test_split/(validation_split + test_split), shuffle=True, random_state=123
            )
            
            logger.info(f"Data split - Train: {len(train_df)}, Val: {len(valid_df)}, Test: {len(test_df)}")
            
            # Data transformations
            img_size = ModelConfig.IMG_SIZE
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
            
            # Create datasets
            train_dataset = LeafDataset(train_df, self.class_to_idx, transform)
            valid_dataset = LeafDataset(valid_df, self.class_to_idx, transform)
            test_dataset = LeafDataset(test_df, self.class_to_idx, transform)
            
            # Create data loaders
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return self.train_loader, self.val_loader, self.test_loader
            
        except Exception as e:
            logger.error(f"Error setting up data loaders: {str(e)}")
            raise PredictionError(f"Data setup failed: {str(e)}")
    
    def setup_model(self, num_classes: int):
        """
        Setup ViT model with enhanced configuration
        """
        try:
            self.model = ViT(
                num_classes=num_classes,
                img_size=ModelConfig.IMG_SIZE,
                patch_size=ModelConfig.PATCH_SIZE,
                embed_dim=ModelConfig.EMBED_DIM,
                depth=ModelConfig.DEPTH,
                num_heads=ModelConfig.NUM_HEADS
            ).to(self.device)
            
            # Setup loss and optimizer (Adamax for better convergence)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adamax(self.model.parameters(), lr=0.001)
            
            logger.info(f"ViT model setup with {num_classes} classes")
            
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise PredictionError(f"Model setup failed: {str(e)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        correct, total = 0, 0
        total_loss = 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return train_acc, avg_loss
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        val_acc = 100 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return val_acc, avg_loss
    
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on given loader"""
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return 100 * correct / total
    
    def train(self, data_dir: str, epochs: int = 5, batch_size: int = 16, 
              learning_rate: float = 0.001, validation_split: float = 0.2,
              save_path: str = "models/vit_mango.pth", save_best_only: bool = True,
              early_stopping_patience: int = 10, run_id: int = None) -> Dict[str, Any]:
        """
        Train the ViT model
        """
        try:
            start_time = time.time()
            # Setup data and model
            self.setup_data_loaders(data_dir, batch_size, validation_split)
            num_classes = len(self.class_to_idx)
            self.setup_model(num_classes)
            
            # Update optimizer with learning rate if provided
            if learning_rate:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate
            
            # Training loop
            best_val_acc = 0
            training_history = {
                'train_acc': [], 'val_acc': [], 
                'train_loss': [], 'val_loss': []
            }
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            for epoch in range(epochs):
                # Train and validate
                train_acc, train_loss = self.train_epoch(epoch)
                val_acc, val_loss = self.validate_epoch()
                
                # Save history
                training_history['train_acc'].append(train_acc)
                training_history['val_acc'].append(val_acc)
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                
                # Log progress
                logger.info(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Acc: {train_acc:.2f}% "
                          f"Val Acc: {val_acc:.2f}% "
                          f"Train Loss: {train_loss:.4f} "
                          f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(save_path, epoch, val_acc)
                
                # Update training run in database if run_id provided
                if run_id:
                    try:
                        from app.database.database import get_db
                        from app.database.repositories import TrainingRunRepository
                        db = next(get_db())
                        repo = TrainingRunRepository(db)
                        repo.update_progress(
                            run_id, 
                            epoch + 1, 
                            train_acc, 
                            val_acc, 
                            train_loss, 
                            val_loss
                        )
                    except Exception as db_err:
                        logger.error(f"Failed to update training progress for run {run_id}: {str(db_err)}")
            
            # Final evaluation
            test_acc = self.evaluate(self.test_loader)
            logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
            
            # Update final status in database
            if run_id:
                try:
                    from app.database.database import get_db
                    from app.database.repositories import TrainingRunRepository
                    db = next(get_db())
                    repo = TrainingRunRepository(db)
                    repo.update_status(run_id, "completed")
                except Exception as db_err:
                    logger.error(f"Failed to complete training run {run_id}: {str(db_err)}")
            
            return {
                'success': True,
                'training_history': training_history,
                'test_accuracy': test_acc,
                'best_val_accuracy': best_val_acc,
                'best_validation_accuracy': best_val_acc,  # added for compatibility
                'training_time_minutes': (time.time() - start_time) / 60 if 'start_time' in locals() else 0,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise PredictionError(f"Training failed: {str(e)}")
    
    def save_model(self, path: str, epoch: int, accuracy: float):
        """Save model with metadata"""
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "class_to_idx": self.class_to_idx,
                "idx_to_class": self.idx_to_class,
                "img_size": ModelConfig.IMG_SIZE,
                "num_classes": len(self.class_to_idx),
                "architecture": "vision_transformer",
                "epoch": epoch,
                "accuracy": accuracy,
                "device": str(self.device)
            }, path)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise PredictionError(f"Model save failed: {str(e)}")
    
    def load_model(self, path: str):
        """Load saved model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Setup model architecture
            num_classes = checkpoint.get('num_classes', 5)
            self.setup_model(num_classes)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load mappings
            self.class_to_idx = checkpoint.get('class_to_idx')
            self.idx_to_class = checkpoint.get('idx_to_class')
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise PredictionError(f"Model load failed: {str(e)}")
