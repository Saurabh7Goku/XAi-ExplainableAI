from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.utils.exceptions import DatabaseError
from app.utils.logger import logger

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for database models
Base = declarative_base()


def get_db() -> Session:
    """Get database session"""
    try:
        db = SessionLocal()
        return db
    except Exception as e:
        logger.error(f"Failed to create database session: {str(e)}")
        raise DatabaseError(f"Database connection failed: {str(e)}")


def init_db() -> None:
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise DatabaseError(f"Database initialization failed: {str(e)}")


def close_db(db: Session) -> None:
    """Close database session"""
    try:
        db.close()
    except Exception as e:
        logger.error(f"Failed to close database session: {str(e)}")


class DatabaseManager:
    """Database manager for handling connections and operations"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return get_db()
    
    def health_check(self) -> bool:
        """Check database health"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    def create_tables(self) -> None:
        """Create all database tables"""
        init_db()


# Global database manager instance
db_manager = DatabaseManager()
