from fastapi import APIRouter, Depends, HTTPException, Request
from app.core.auth import get_current_user
from app.services.supabase import supabase
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/user", tags=["User"])

@router.get("/profile")
async def get_user_profile(request: Request, current_user = Depends(get_current_user)):
    try:
        logger.info(f"Fetching profile for user: {current_user.id}")
        
        # Just return the basic info we already have from the user object
        return {
            "message": "User profile retrieved successfully",
            "user": {
                "id": current_user.id
            }
        }
    except Exception as e:
        logger.error(f"Failed to fetch user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 