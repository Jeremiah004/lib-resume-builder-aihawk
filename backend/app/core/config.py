import os
from dotenv import load_dotenv
import logging
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
logger.debug(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

class Settings(BaseSettings):
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str

    DATABASE_URL: str

    # GROQ API Configuration
    GROQ_API_KEY: str
        # Directory Configuration
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STYLE_BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOG_DIR: Path = BASE_DIR / "logs"
    STYLES_DIR: Path = STYLE_BASE_DIR / "resume_style"
    
    API_V1_STR: str
    PROJECT_NAME: str

    REDIS_HOST: Optional[str] = "localhost"
    REDIS_PORT: Optional[int] = 6379
    REDIS_DB: Optional[int] = 0
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Resume Builder API"
    
    # Security Configuration
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # File Configuration
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_FILE_TYPES: set = {'.pdf', '.docx'}
    
    # Frontend URL for redirects
    FRONTEND_URL: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Using SUPABASE_KEY from .env

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
logger.debug(f"SUPABASE_URL: {SUPABASE_URL}")
logger.debug(f"SUPABASE_ANON_KEY: {'Set' if SUPABASE_KEY else 'Not Set'}")
logger.debug(f"GROQ_API_KEY: {"able" if GROQ_API_KEY else "the nigga aint set"}")

# Frontend URL for redirects
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
logger.debug(f"FRONTEND_URL: {FRONTEND_URL}")

# Directory Configuration
STYLE_BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR: Path = BASE_DIR / "uploads"
OUTPUT_DIR: Path = BASE_DIR / "output"
LOG_DIR: Path = BASE_DIR / "logs"
STYLES_DIR: Path = STYLE_BASE_DIR / "resume_style"

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# API Configuration
API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
PROJECT_NAME = os.getenv("PROJECT_NAME", "Resume Builder API")

# Validate Supabase Configuration
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase credentials are missing in .env file")
    raise ValueError("Supabase credentials are missing in .env file") 