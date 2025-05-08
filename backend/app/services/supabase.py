from supabase import create_client, Client
from app.core.config import SUPABASE_URL, SUPABASE_KEY
import logging
import httpx
import socket
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def test_supabase_connection():
    try:
        # Test DNS resolution
        hostname = SUPABASE_URL.replace('https://', '')
        socket.gethostbyname(hostname)
        logger.info(f"DNS resolution successful for {hostname}")
        
        # Test HTTPS connection to a specific endpoint
        health_check_url = f"{SUPABASE_URL}/rest/v1/"
        logger.info(f"Testing connection to: {health_check_url}")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(health_check_url, headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            })
            response.raise_for_status()
            logger.info("Supabase connection successful")
    except socket.gaierror as e:
        logger.error(f"DNS resolution failed: {str(e)}")
        raise ValueError(f"Could not resolve Supabase hostname: {hostname}")
    except httpx.TimeoutException as e:
        logger.error(f"Connection timeout: {str(e)}")
        raise ValueError("Connection to Supabase timed out. Please check your network connection.")
    except httpx.HTTPError as e:
        logger.error(f"HTTPS connection failed: {str(e)}")
        if e.response.status_code == 404:
            raise ValueError(f"Supabase project not found. Please verify the project URL: {SUPABASE_URL}")
        raise
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        raise

try:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials are missing in .env file")
    
    logger.info(f"Initializing Supabase client with URL: {SUPABASE_URL}")
    
    # Test connection before creating client
    test_supabase_connection()
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise 