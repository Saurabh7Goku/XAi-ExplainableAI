"""
Custom exceptions for the application
"""


class MangoLeafException(Exception):
    """Base exception for the application"""
    pass


class ModelLoadError(MangoLeafException):
    """Raised when model fails to load"""
    pass


class PredictionError(MangoLeafException):
    """Raised when prediction fails"""
    pass


class InvalidImageError(MangoLeafException):
    """Raised when image is invalid"""
    pass


class DatabaseError(MangoLeafException):
    """Raised when database operation fails"""
    pass


class LLMServiceError(MangoLeafException):
    """Raised when LLM service fails"""
    pass


class FileUploadError(MangoLeafException):
    """Raised when file upload fails"""
    pass
