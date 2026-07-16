"""
Input validation utilities
"""

import os
import io
from typing import List
from fastapi import UploadFile, HTTPException
from PIL import Image
from app.config import settings
from app.utils.exceptions import InvalidImageError, FileUploadError


def validate_image_upload(file: UploadFile) -> None:
    """Validate uploaded image file"""
    
    # Check file size
    if file.size and file.size > settings.max_file_size:
        raise FileUploadError(
            f"File size exceeds maximum allowed size of {settings.max_file_size // (1024*1024)}MB"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.allowed_extensions:
        raise FileUploadError(
            f"File type {file_extension} not allowed. Allowed types: {', '.join(settings.allowed_extensions)}"
        )
    
    # Validate image content
    try:
        # Read file content to validate it's a valid image
        content = file.file.read()
        file.file.seek(0)  # Reset file pointer
        
        # Try to open with PIL
        image = Image.open(io.BytesIO(content))
        image.verify()  # Verify it's a valid image
        
        # Check if image can be reopened (verify() closes the file)
        image = Image.open(io.BytesIO(content))
        
        # Check image dimensions
        if image.size[0] < 32 or image.size[1] < 32:
            raise InvalidImageError("Image dimensions too small (minimum 32x32)")
            
    except Exception as e:
        if isinstance(e, InvalidImageError):
            raise
        raise InvalidImageError(f"Invalid image file: {str(e)}")


def validate_prediction_input(image_path: str) -> None:
    """Validate image path for prediction"""
    
    if not os.path.exists(image_path):
        raise InvalidImageError(f"Image file not found: {image_path}")
    
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        raise InvalidImageError(f"Invalid image file: {str(e)}")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename
