from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, validator
from app.services.supabase import supabase
import logging
import re

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Updated email validation pattern that better handles Gmail addresses
EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

class UserRegister(BaseModel):
    email: str
    password: str

    @validator('email')
    def validate_email(cls, v):
        # Convert to lowercase and remove any spaces
        v = v.lower().strip()
        if not re.match(EMAIL_PATTERN, v):
            raise ValueError('Email must be in a valid format (e.g., user@gmail.com)')
        # Additional checks for Supabase requirements
        if len(v) > 255:
            raise ValueError('Email must be less than 255 characters')
        if '..' in v:
            raise ValueError('Email cannot contain consecutive dots')
        if v.startswith('.') or v.endswith('.'):
            raise ValueError('Email cannot start or end with a dot')
        # Gmail-specific validation
        if '@gmail.com' in v:
            # Remove any dots in the username part for Gmail
            username, domain = v.split('@')
            username = username.replace('.', '')
            v = f"{username}@{domain}"
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

class UserLogin(BaseModel):
    email: str
    password: str

    @validator('email')
    def validate_email(cls, v):
        v = v.lower().strip()
        if not re.match(EMAIL_PATTERN, v):
            raise ValueError('Email must be in a valid format (e.g., user@gmail.com)')
        # Gmail-specific validation
        if '@gmail.com' in v:
            username, domain = v.split('@')
            username = username.replace('.', '')
            v = f"{username}@{domain}"
        return v

@router.post("/register")
async def register(user: UserRegister):
    try:
        logger.info(f"Attempting to register user: {user.email}")
        response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password
        })
        logger.info("Registration successful")
        return {
            "message": "User registered successfully",
            "user": response.user
        }
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        error_msg = str(e)
        if "email_address_invalid" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="Invalid email address. Please use a valid email format (e.g., user@gmail.com)"
            )
        elif "password" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="Invalid password. Password must be at least 6 characters long"
            )
        raise HTTPException(status_code=400, detail=error_msg)

@router.post("/login")
async def login(user: UserLogin):
    try:
        logger.info(f"Attempting to login user: {user.email}")
        response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        logger.info("Login successful")
        # Get the complete session data
        session = response.session
        # Log the complete token for debugging
        logger.info(f"Complete access token: {session.access_token}")
        return {
            "message": "Login successful",
            "access_token": session.access_token,  # This should be the complete token
            "token_type": "bearer",
            "user": response.user
        }
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        if "Invalid login credentials" in str(e):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        raise HTTPException(status_code=401, detail=str(e))

@router.post("/google/login")
async def google_login():
    """
    Initiates Google OAuth login flow through Supabase
    Returns the authorization URL for the frontend to redirect to
    """
    try:
        logger.info("Initiating Google OAuth login flow")
        # Get the OAuth URL from Supabase
        auth_url = supabase.auth.get_sign_in_with_oauth_credential(
            {
                "provider": "google",
                "options": {
                    "redirectTo": f"{FRONTEND_URL}/auth/callback"
                }
            }
        ).url
        
        return {
            "auth_url": auth_url
        }
    except Exception as e:
        logger.error(f"Failed to initiate Google OAuth: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/github/login")
async def github_login():
    """
    Initiates GitHub OAuth login flow through Supabase
    Returns the authorization URL for the frontend to redirect to
    """
    try:
        logger.info("Initiating GitHub OAuth login flow")
        # Get the OAuth URL from Supabase
        auth_url = supabase.auth.get_sign_in_with_oauth_credential(
            {
                "provider": "github",
                "options": {
                    "redirectTo": f"{FRONTEND_URL}/auth/callback"
                }
            }
        ).url
        
        return {
            "auth_url": auth_url
        }
    except Exception as e:
        logger.error(f"Failed to initiate GitHub OAuth: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/oauth/callback")
async def handle_oauth_callback(code: str, provider: str):
    """
    Handles OAuth callback from providers
    Exchanges code for user session and returns access token
    """
    try:
        logger.info(f"Processing OAuth callback for provider: {provider}")
        
        # Exchange the code for a user session
        response = supabase.auth.exchange_code_for_session(code)
        
        if not response or not hasattr(response, 'session'):
            raise HTTPException(
                status_code=400,
                detail="Failed to exchange code for session"
            )
        
        # Get the session data
        session = response.session
        
        return {
            "message": "OAuth authentication successful",
            "access_token": session.access_token,
            "token_type": "bearer",
            "user": response.user
        }
    except Exception as e:
        logger.error(f"OAuth callback failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/logout")
async def logout():
    """
    Handles user logout
    """
    try:
        logger.info("Processing user logout")
        # No need to actually do anything with Supabase since token invalidation
        # is handled client-side and tokens are short-lived
        return {
            "message": "Logout successful"
        }
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))