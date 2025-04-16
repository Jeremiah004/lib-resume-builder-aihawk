from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class GlobalConfig:
    def __init__(self):
        # Set module paths
        self.STRINGS_MODULE_RESUME_PATH = Path("lib_resume_builder_AIHawk/resume_prompt/strings_feder-cr.py")
        self.STRINGS_MODULE_RESUME_JOB_DESCRIPTION_PATH = Path("lib_resume_builder_AIHawk/resume_job_description_prompt/strings_feder-cr.py")
        self.STRINGS_MODULE_NAME = "strings_feder_cr"
        self.STYLES_DIRECTORY = Path("styles")
        self.LOG_OUTPUT_FILE_PATH = Path("logs")
        self.GROQ_API_KEY = None
        self.html_template = """
                            <!DOCTYPE html>
                            <html lang="en">
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <title>Resume</title>
                                <link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&display=swap" rel="stylesheet" />
                                <link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&display=swap" rel="stylesheet" /> 
                                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" /> 
                                <link rel="stylesheet" href="$style_path">
                            </head>
                            $markdown
                            </body>
                            </html>
                            """

class Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Load secrets from secrets.yaml
        self.secrets = self._load_secrets()
        
        # Initialize API keys
        self.groq_api_key = self._get_api_key('groq')
        self.openai_api_key = self._get_api_key('openai')
        self.google_api_key = self._get_api_key('google')
        self.huggingface_token = self._get_api_key('huggingface')
        
    def _load_secrets(self):
        """Load secrets from secrets.yaml file"""
        try:
            secrets_path = Path('secrets.yaml')
            if secrets_path.exists():
                with open(secrets_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load secrets.yaml: {str(e)}")
            return {}
    
    def _get_api_key(self, service):
        """Get API key from environment variables or secrets file"""
        # Check environment variables first
        env_mapping = {
            'groq': 'GROQ_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'huggingface': 'HUGGINGFACE_TOKEN'
        }
        
        secrets_mapping = {
            'groq': 'groq_api_key',
            'openai': 'openai_api_key',
            'google': 'google_api_key',
            'huggingface': 'huggingface_token'
        }
        
        # Try environment variable first
        if service in env_mapping:
            key = os.getenv(env_mapping[service])
            if key and key != f"your_{service}_api_key_here":
                return key
        
        # Try secrets file next
        if service in secrets_mapping and self.secrets:
            key = self.secrets.get(secrets_mapping[service])
            if key:
                return key
        
        return None
    
    def validate_config(self):
        """Validate that required API keys are present"""
        missing_keys = []
        
        # Check required keys
        if not self.groq_api_key:
            missing_keys.append("Groq API Key")
        
        # Optional keys with warnings
        if not self.huggingface_token:
            logger.warning("HuggingFace token not found. Some features may have reduced performance.")
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True
    
    def get_llm_config(self):
        """Get LLM configuration dictionary"""
        return {
            'groq_api_key': self.groq_api_key,
            'openai_api_key': self.openai_api_key,
            'huggingface_token': self.huggingface_token
        }
    
    def get_google_config(self):
        """Get Google API configuration"""
        return {
            'api_key': self.google_api_key
        }

# Create global instance
global_config = GlobalConfig()
