from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn

from app.config import settings
from app.database.database import init_db, db_manager
from app.services.knowledge_base import knowledge_base_service
from app.utils.logger import logger
from app.utils.exceptions import MangoLeafException
from app.api.routes import prediction, health, training


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered Mango Leaf Disease Detection using custom Vision Transformer",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(prediction.router)
    app.include_router(health.router)
    app.include_router(training.router)
    
    # Add startup and shutdown events
    setup_events(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Setup application middleware"""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup custom exception handlers"""
    
    @app.exception_handler(MangoLeafException)
    async def mango_leaf_exception_handler(request: Request, exc: MangoLeafException):
        logger.error(f"MangoLeafException: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(exc),
                "error_type": "application_error"
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.error(f"ValueError: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(exc),
                "error_type": "validation_error"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "error_type": "server_error"
            }
        )


def setup_events(app: FastAPI) -> None:
    """Setup application startup and shutdown events"""
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup"""
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        
        try:
            # Initialize database
            try:
                init_db()
                logger.info("Database initialized successfully")
                
                # Check database health
                if db_manager.health_check():
                    logger.info("Database health check passed")
                else:
                    logger.warning("Database health check failed")
            except Exception as db_err:
                logger.warning(f"Database initialization failed: {str(db_err)}. Using fallback for data.")
            
            # Initialize knowledge base
            knowledge_base_service.initialize_diseases()
            logger.info("Knowledge base initialized successfully")
            
            logger.info("Application startup completed")
            
        except Exception as e:
            logger.error(f"Startup failed: {str(e)}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown"""
        logger.info("Shutting down application...")
        
        try:
            # Close services
            knowledge_base_service.close()
            logger.info("Services closed successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")
        
        logger.info("Application shutdown completed")


# Create application instance
app = create_application()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Mango Leaf Disease Detection API",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "Documentation disabled in production"
    }


if __name__ == "__main__":
    import os
    # Priority for Render's injected PORT environment variable
    server_port = int(os.environ.get("PORT", settings.port))
    server_host = settings.host
    
    logger.info(f"Starting server on {server_host}:{server_port} (with reload={settings.debug})")
    
    uvicorn.run(
        "app.main:app",
        host=server_host,
        port=server_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
