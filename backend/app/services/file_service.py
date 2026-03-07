import os
import uuid
import shutil
import tempfile
import io
from typing import Optional, Tuple
from pathlib import Path
from PIL import Image
from fastapi import UploadFile
from app.config import settings
from app.utils.exceptions import FileUploadError, InvalidImageError
from app.utils.validators import validate_image_upload, sanitize_filename
from app.utils.logger import logger


class FileService:
    """Service for handling file uploads and file management"""
    
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.temp_dir = Path("temp")
        self.explanation_dir = Path("explanations")
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        try:
            self.upload_dir.mkdir(exist_ok=True)
            self.temp_dir.mkdir(exist_ok=True)
            self.explanation_dir.mkdir(exist_ok=True)
            logger.info("File directories created/verified")
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            raise FileUploadError(f"Failed to create directories: {str(e)}")
    
    def save_uploaded_file(self, file: UploadFile) -> Tuple[str, str]:
        """
        Save uploaded file to disk
        
        Args:
            file: UploadFile object from FastAPI
            
        Returns:
            Tuple of (file_path, filename)
        """
        try:
            # Validate the uploaded file
            validate_image_upload(file)
            
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1].lower()
            safe_filename = sanitize_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{safe_filename}"
            
            # Create file path
            file_path = self.upload_dir / unique_filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved successfully: {file_path}")
            return str(file_path), unique_filename
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            raise FileUploadError(f"Failed to save file: {str(e)}")
    
    def save_temp_file(self, image_data: bytes, filename: str = None) -> str:
        """
        Save temporary file for processing
        
        Args:
            image_data: Raw image data
            filename: Optional filename
            
        Returns:
            Path to saved temporary file
        """
        try:
            if filename is None:
                filename = f"temp_{uuid.uuid4()}.jpg"
            
            temp_path = self.temp_dir / sanitize_filename(filename)
            
            with open(temp_path, "wb") as f:
                f.write(image_data)
            
            logger.debug(f"Temporary file saved: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to save temporary file: {str(e)}")
            raise FileUploadError(f"Failed to save temporary file: {str(e)}")
    
    def save_explanation_image(self, explanation_array, filename: str = None) -> str:
        """
        Save LIME explanation image
        
        Args:
            explanation_array: Numpy array of explanation
            filename: Optional filename
            
        Returns:
            Path to saved explanation image
        """
        try:
            if filename is None:
                filename = f"explanation_{uuid.uuid4()}.png"
            
            explanation_path = self.explanation_dir / sanitize_filename(filename)
            
            # Convert numpy array to PIL Image and save
            from skimage.segmentation import mark_boundaries
            import matplotlib.pyplot as plt
            
            # Create figure and save
            plt.figure(figsize=(8, 8))
            plt.imshow(explanation_array)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(explanation_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Explanation image saved: {explanation_path}")
            return str(explanation_path)
            
        except Exception as e:
            logger.error(f"Failed to save explanation image: {str(e)}")
            raise FileUploadError(f"Failed to save explanation image: {str(e)}")
    
    def validate_image_file(self, file_path: str) -> bool:
        """
        Validate that file is a valid image
        
        Args:
            file_path: Path to image file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"Image validation failed for {file_path}: {str(e)}")
            return False
    
    def get_image_info(self, file_path: str) -> dict:
        """
        Get image information
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(file_path) as img:
                return {
                    'filename': os.path.basename(file_path),
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'file_size': os.path.getsize(file_path)
                }
        except Exception as e:
            logger.error(f"Failed to get image info for {file_path}: {str(e)}")
            raise InvalidImageError(f"Failed to read image: {str(e)}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
            
        Returns:
            Number of files cleaned up
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0
            
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    file_age = current_time - temp_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        temp_file.unlink()
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {str(e)}")
            return 0
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file safely
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            file_path = Path(file_path)
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                logger.debug(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False
    
    def get_file_stats(self) -> dict:
        """Get statistics about stored files"""
        try:
            upload_count = len(list(self.upload_dir.glob("*")))
            temp_count = len(list(self.temp_dir.glob("*")))
            explanation_count = len(list(self.explanation_dir.glob("*")))
            
            upload_size = sum(f.stat().st_size for f in self.upload_dir.glob("*") if f.is_file())
            temp_size = sum(f.stat().st_size for f in self.temp_dir.glob("*") if f.is_file())
            explanation_size = sum(f.stat().st_size for f in self.explanation_dir.glob("*") if f.is_file())
            
            return {
                'upload_files': upload_count,
                'temp_files': temp_count,
                'explanation_files': explanation_count,
                'upload_size_mb': round(upload_size / (1024 * 1024), 2),
                'temp_size_mb': round(temp_size / (1024 * 1024), 2),
                'explanation_size_mb': round(explanation_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Failed to get file stats: {str(e)}")
            return {}


# Global file service instance
file_service = FileService()
