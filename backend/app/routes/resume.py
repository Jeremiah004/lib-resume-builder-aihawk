from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import JSONResponse
from app.core.auth import get_current_user
from app.utils.file_parser import extract_text_from_pdf, extract_text_from_docx
from lib_resume_builder_AIHawk import LLMResumer, Resume, StyleManager, ResumeGenerator
from lib_resume_builder_AIHawk.manager_facade import FacadeManager
from app.core.config import GROQ_API_KEY, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR, STYLES_DIR
from app.services.storage_service import StorageService
import logging
import os
import tempfile
from pathlib import Path
from datetime import datetime, UTC, timedelta  # Using UTC instead of utcnow()
import uuid
import base64
import time
import yaml
import hashlib
from sqlalchemy.exc import SQLAlchemyError
from functools import lru_cache
import threading
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "resume_api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "cache"), exist_ok=True)

router = APIRouter(prefix="/resume", tags=["Resume"])

# Initialize services
storage_service = StorageService()

# PDF cache for 24 hours
pdf_cache = {}
PDF_CACHE_EXPIRY = timedelta(hours=24)
DISK_CACHE_EXPIRY = timedelta(days=7)  # Keep files on disk for longer

# Function to clean up old cache files
def cleanup_cache():
    """Clean up expired items from the cache dictionary and cache directory"""
    try:
        current_time = datetime.now(UTC)
        
        # Clean memory cache
        expired_keys = []
        for key, (_, timestamp) in pdf_cache.items():
            if current_time - timestamp > PDF_CACHE_EXPIRY:
                expired_keys.append(key)
        
        for key in expired_keys:
            del pdf_cache[key]
            
        logger.info(f"Cleaned {len(expired_keys)} expired items from memory cache")
        
        # Clean disk cache
        cache_dir = os.path.join(OUTPUT_DIR, "cache")
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                if os.path.isfile(filepath):
                    file_modified = datetime.fromtimestamp(os.path.getmtime(filepath), tz=UTC)
                    if current_time - file_modified > DISK_CACHE_EXPIRY:
                        os.remove(filepath)
                        logger.info(f"Removed expired cache file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up cache: {str(e)}")

# Schedule periodic cache cleanup
def schedule_cache_cleanup(interval_hours=6):
    """Schedule cache cleanup to run periodically"""
    cleanup_cache()
    # Schedule the next run
    threading.Timer(interval_hours * 3600, schedule_cache_cleanup, [interval_hours]).start()
    
# Start the cache cleanup scheduler when the module loads
schedule_cache_cleanup()

# Function to generate cache key
def generate_cache_key(resume_id, style_name, job_description_text=None):
    """Generate a unique cache key based on inputs"""
    key_parts = [resume_id, style_name]
    if job_description_text:
        # Use a hash of the job description to avoid super long keys
        key_parts.append(hashlib.md5(job_description_text.encode()).hexdigest())
    return "_".join(key_parts)

# Helper function to extract JWT token from request
async def get_token_from_request(request: Request) -> str:
    """
    Extract the JWT token from the Authorization header
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.replace("Bearer ", "")
    return None

# Add a helper function to handle Resume object creation
def create_resume_object(resume_data):
    """
    Create a Resume object from either a dictionary or a YAML string.
    
    Args:
        resume_data: Either a dictionary or a YAML string containing resume data
        
    Returns:
        Resume: A Resume object populated with the data
    """
    logger.debug(f"Creating Resume object from data of type: {type(resume_data)}")
    
    # If it's already a dictionary, convert to YAML string first
    if isinstance(resume_data, dict):
        logger.debug("Converting dictionary to YAML string")
        yaml_str = yaml.dump(resume_data)
        return Resume(yaml_str)
    else:
        # Assume it's already a YAML string or file-like object
        return Resume(resume_data)

@router.post("/upload")
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    Upload and process a resume file, store it in Supabase, and return the resume object.
    """
    start_time = time.time()
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
            logger.info("Token extracted and set for authentication")
        
        # Validate file type
        file_type = file.filename.split('.')[-1].lower()
        if file_type not in ['pdf', 'docx']:
            # Return proper status code for invalid file type
            return JSONResponse(
                status_code=400,
                content={"detail": "Only PDF and DOCX files are supported"}
            )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Extract text from file
            if file_type == 'pdf':
                text = extract_text_from_pdf(temp_path)
            else:
                text = extract_text_from_docx(temp_path)

            if not text:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Failed to extract text from file"}
                )

            # Initialize LLMResumer and parse the text
            resumer = LLMResumer(GROQ_API_KEY, None)
            resume_data = resumer.extract_resume_info(text)
            
            # Log the resume data structure for debugging
            logger.debug(f"Extracted resume data type: {type(resume_data)}")
            
            # Create a test Resume object to verify it works (will be discarded)
            try:
                # This is just to verify the data format is valid before storing
                test_resume = create_resume_object(resume_data)
                logger.debug("Successfully created test Resume object from extracted data")
            except Exception as e:
                logger.error(f"Error creating test Resume object: {str(e)}")
                raise ValueError(f"400: Invalid resume data format: {str(e)}")

            # Store resume in Supabase
            resume_id = await storage_service.store_resume(
                user_id=current_user.id,
                resume_data=resume_data
            )

            # Get the stored resume
            stored_resume = await storage_service.get_resume(resume_id, current_user.id)

            return JSONResponse(content={
                "status": "success",
                "data": {
                    "resume": {
                        "id": resume_id,
                        "user_id": current_user.id,
                        "data": stored_resume["resume_data"],
                        "created_at": stored_resume["created_at"],
                        "updated_at": stored_resume["updated_at"]
                    }
                },
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "file_size": len(content),
                    "version": "1.0.0"
                }
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        # If the error message already contains a 400 status code context, return 400
        if "400:" in str(e):
            return JSONResponse(
                status_code=400,
                content={"detail": str(e).split("400:")[1].strip()}
            )
        # Otherwise return 500
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )

@router.post("/generate")
async def generate_resume(
    request: Request,
    resume_id: str = Form(...),
    style_name: str = Form(...),
    job_description_text: str = Form(None),
    job_description_url: str = Form(None),
    background_tasks: BackgroundTasks = None,
    current_user = Depends(get_current_user)
):
    """
    Generate a styled resume PDF based on the stored resume.
    """
    start_time = time.time()
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
            logger.info("Token extracted and set for authentication")
        
        # Generate cache key
        cache_key = generate_cache_key(resume_id, style_name, job_description_text)
        
        # Check if we have a cached version
        current_time = datetime.now(UTC)
        if cache_key in pdf_cache:
            cached_data, cache_time = pdf_cache[cache_key]
            # If cache is still valid (less than expiry period)
            if current_time - cache_time < PDF_CACHE_EXPIRY:
                logger.info(f"Using cached PDF for key: {cache_key}")
                
                # Get the resume for metadata
                try:
                    stored_resume = await storage_service.get_resume(resume_id, current_user.id)
                except Exception:
                    # If we can't get the resume, still return the cached PDF
                    stored_resume = {"created_at": None, "updated_at": None, "resume_data": None}
                
                # Return the cached data
                return JSONResponse(content={
                    "status": "success",
                    "data": {
                        "resume": {
                            "id": resume_id,
                            "user_id": current_user.id,
                            "data": stored_resume.get("resume_data"),
                            "created_at": stored_resume.get("created_at"),
                            "updated_at": stored_resume.get("updated_at")
                        },
                        "pdf": {
                            "content_base64": cached_data,
                            "style": style_name,
                            "filename": f"resume_{resume_id}_{style_name}.pdf",
                            "generated_at": current_time.isoformat(),
                            "from_cache": True
                        }
                    },
                    "metadata": {
                        "processing_time": time.time() - start_time,
                        "version": "1.0.0"
                    }
                })
        
        # Get the resume from Supabase
        try:
            stored_resume = await storage_service.get_resume(resume_id, current_user.id)
            if not stored_resume:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
        except Exception as e:
            # If there's an issue with syntax or the resume doesn't exist
            error_message = str(e)
            if "invalid input syntax" in error_message or "not found" in error_message:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
            raise
        
        # Create Resume object from stored data using the helper function
        try:
            resume_object = create_resume_object(stored_resume["resume_data"])
            logger.debug(f"Successfully created Resume object from stored data")
        except Exception as e:
            logger.error(f"Error creating Resume object: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing resume data: {str(e)}"
            )
        
        # Set up the FacadeManager
        log_path = os.path.join(LOG_DIR, f"resume_generation_{current_user.id}_{resume_id}.log")
        style_manager = StyleManager()
        resume_generator = ResumeGenerator(GROQ_API_KEY)
        
        facade_manager = FacadeManager(
            groq_api_key=GROQ_API_KEY,
            style_manager=style_manager,
            resume_generator=resume_generator,
            resume_object=resume_object,
            log_path=log_path,
            storage_service=storage_service
        )
        
        # Set the selected style
        try:
            facade_manager.set_style(style_name)
        except ValueError as e:
            # If style is invalid, return available styles to help the user
            style_manager = StyleManager()
            style_manager.set_styles_directory(Path(STYLES_DIR))
            styles_dict = style_manager.get_styles()
            formatted_styles = style_manager.format_choices(styles_dict)
            
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"Style error: {str(e)}",
                    "available_styles": formatted_styles
                }
            )
            
        # Generate the PDF
        try:
            pdf_data = facade_manager.generate_pdf(
                job_description_url=job_description_url,
                job_description_text=job_description_text
            )
            
            if not pdf_data:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Failed to generate PDF"}
                )
            
            # Cache the PDF for future requests
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            pdf_cache[cache_key] = (pdf_base64, current_time)
            
            # Save a cached copy to disk too for long-term caching
            cache_dir = os.path.join(OUTPUT_DIR, "cache")
            cache_file = os.path.join(cache_dir, f"{cache_key}.pdf")
            with open(cache_file, "wb") as f:
                f.write(pdf_data)
            
            # Instead of storing in Supabase, return the PDF data directly if there's a permission issue
            try:
                # Store PDF in Supabase
                pdf_url = await storage_service.store_pdf(
                    resume_id=resume_id,
                    user_id=current_user.id,
                    pdf_data=pdf_data,
                    style_name=style_name
                )
                
                return JSONResponse(content={
                    "status": "success",
                    "data": {
                        "resume": {
                            "id": resume_id,
                            "user_id": current_user.id,
                            "data": stored_resume["resume_data"],
                            "created_at": stored_resume["created_at"],
                            "updated_at": stored_resume["updated_at"]
                        },
                        "pdf": {
                            "url": pdf_url,
                            "style": style_name,
                            "filename": f"resume_{resume_id}_{style_name}.pdf",
                            "generated_at": datetime.now(UTC).isoformat()
                        }
                    },
                    "metadata": {
                        "processing_time": time.time() - start_time,
                        "file_size": len(pdf_data),
                        "version": "1.0.0"
                    }
                })
                
            except Exception as storage_error:
                # Handle RLS policy violation error
                error_str = str(storage_error)
                if "row-level security policy" in error_str.lower() or "statusCode': 403" in error_str:
                    logger.warning(f"RLS policy prevented PDF storage: {error_str}")
                    # Return success but with a warning about storage
                    return JSONResponse(content={
                        "status": "partial_success",
                        "warning": "PDF was generated but could not be stored due to permission issues",
                        "data": {
                            "resume": {
                                "id": resume_id,
                                "user_id": current_user.id,
                                "data": stored_resume["resume_data"],
                                "created_at": stored_resume["created_at"],
                                "updated_at": stored_resume["updated_at"]
                            },
                            "pdf": {
                                "content_base64": pdf_base64,
                                "style": style_name,
                                "filename": f"resume_{resume_id}_{style_name}.pdf",
                                "generated_at": datetime.now(UTC).isoformat()
                            }
                        },
                        "metadata": {
                            "processing_time": time.time() - start_time,
                            "file_size": len(pdf_data),
                            "version": "1.0.0"
                        }
                    })
                else:
                    # Re-raise for other storage errors
                    raise
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            if "dict' object has no attribute 'read'" in str(e):
                # Special handling for YAML parsing error
                return JSONResponse(
                    status_code=500,
                    content={"detail": "YAML parsing error. Check your style file format."}
                )
            raise HTTPException(
                status_code=500,
                detail=f"Error generating PDF: {str(e)}"
            )
            
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error in resume generation process: {str(e)}")
        error_message = str(e)
        
        # Check for specific errors that should return specific status codes
        if "invalid input syntax" in error_message:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Resume not found with ID: {resume_id}"}
            )
        elif "Style" in error_message:
            return JSONResponse(
                status_code=400,
                content={"detail": error_message}
            )
        # The YAML parsing error
        elif "dict' object has no attribute 'read'" in error_message:
            return JSONResponse(
                status_code=500,
                content={"detail": "YAML parsing error in style file. Please check the style format."}
            )
        
        # Default case: return 500 error
        raise HTTPException(
            status_code=500,
            detail=f"Error in resume generation process: {str(e)}"
        )

@router.get("/list")
async def list_resumes(
    request: Request,
    current_user = Depends(get_current_user)
):
    """
    List all resumes belonging to the current user.
    """
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
            logger.info("Token extracted and set for authentication")
            
        # Get all resumes for the user
        result = storage_service.supabase.table("resumes").select("*").eq("user_id", current_user.id).execute()
        resumes = result.data
        
        # Get all PDFs for each resume
        for resume in resumes:
            pdfs = await storage_service.get_resume_pdfs(resume["id"], current_user.id)
            resume["generated_pdfs"] = pdfs
        
        return JSONResponse(content={
            "status": "success",
            "data": {
                "resumes": resumes
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to list resumes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list resumes: {str(e)}"
        )

@router.delete("/{resume_id}")
async def delete_resume(
    request: Request,
    resume_id: str,
    current_user = Depends(get_current_user)
):
    """
    Delete a resume and all its associated PDFs.
    """
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
            logger.info("Token extracted and set for authentication")
            
        try:
            # First check if the resume exists
            resume = await storage_service.get_resume(resume_id, current_user.id)
            if not resume:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
                
            # Proceed with deletion if it exists
            await storage_service.delete_resume(resume_id, current_user.id)
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Resume {resume_id} and all associated PDFs deleted successfully"
            })
        except Exception as e:
            error_message = str(e)
            if "invalid input syntax" in error_message:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
            raise
            
    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={"detail": str(e)}
        )
    except Exception as e:
        logger.error(f"Failed to delete resume: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete resume: {str(e)}"
        )

@router.get("/styles")
async def get_available_styles():
    """
    Get all available resume styles with their author information.
    Returns a list of styles in the format: "style_name (style author -> author_link)"
    """
    try:
        style_manager = StyleManager()
        style_manager.set_styles_directory(Path(STYLES_DIR))
        
        # Get styles dictionary
        styles_dict = style_manager.get_styles()
        
        # Format choices for frontend display
        formatted_styles = style_manager.format_choices(styles_dict)
        
        return JSONResponse(content={
            "status": "success",
            "data": {
                "styles": formatted_styles,
                "raw_styles": styles_dict  # Include raw data for more flexibility
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get styles: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get styles: {str(e)}"
        )

@router.get("/test-auth")
async def test_auth(
    request: Request,
    current_user = Depends(get_current_user)
):
    """
    Test authentication and user ID extraction
    """
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
            logger.info("Token extracted and set for authentication")
            
        return JSONResponse(content={
            "status": "success",
            "data": {
                "user_id": current_user.id,
                "token_available": bool(token)
            },
            "message": "Authentication successful"
        })
    except Exception as e:
        logger.error(f"Auth test failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Auth test failed: {str(e)}"
        )

@router.post("/prepare-generation")
async def prepare_resume_generation(
    request: Request,
    resume_id: str = Form(...),
    current_user = Depends(get_current_user)
):
    """
    Prepare for resume generation by checking the resume and providing available styles.
    This is a helper endpoint to use before calling the generate endpoint.
    """
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
            logger.info("Token extracted and set for authentication")
            
        # First check if the resume exists and is valid
        try:
            stored_resume = await storage_service.get_resume(resume_id, current_user.id)
            if not stored_resume:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
                
            # Try to create a Resume object to validate the data
            resume_object = create_resume_object(stored_resume["resume_data"])
            logger.debug(f"Successfully created Resume object from stored data")
            
        except Exception as e:
            # If there's an issue with the resume
            error_message = str(e)
            if "invalid input syntax" in error_message or "not found" in error_message:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
            logger.error(f"Error processing resume data: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error processing resume data: {str(e)}"}
            )
        
        # Get available styles
        style_manager = StyleManager()
        style_manager.set_styles_directory(Path(STYLES_DIR))
        styles_dict = style_manager.get_styles()
        formatted_styles = style_manager.format_choices(styles_dict)
        
        # Return resume data and available styles
        return JSONResponse(content={
            "status": "success",
            "data": {
                "resume": {
                    "id": resume_id,
                    "user_id": current_user.id,
                    "created_at": stored_resume["created_at"],
                    "updated_at": stored_resume["updated_at"]
                },
                "available_styles": formatted_styles,
                "style_details": styles_dict
            },
            "message": "Ready for resume generation. Please select a style."
        })
        
    except Exception as e:
        logger.error(f"Error preparing for resume generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error preparing for resume generation: {str(e)}"
        )

@router.get("/prepare-generation/{resume_id}")
async def prepare_resume_generation_get(
    request: Request,
    resume_id: str,
    current_user = Depends(get_current_user)
):
    """
    GET version of prepare-generation endpoint. Prepares for resume generation by checking
    the resume and providing available styles.
    This is a helper endpoint to use before calling the generate endpoint.
    """
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
            logger.info("Token extracted and set for authentication")
            
        # First check if the resume exists and is valid
        try:
            stored_resume = await storage_service.get_resume(resume_id, current_user.id)
            if not stored_resume:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
                
            # Try to create a Resume object to validate the data
            resume_object = create_resume_object(stored_resume["resume_data"])
            logger.debug(f"Successfully created Resume object from stored data")
            
        except Exception as e:
            # If there's an issue with the resume
            error_message = str(e)
            if "invalid input syntax" in error_message or "not found" in error_message:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Resume not found with ID: {resume_id}"}
                )
            logger.error(f"Error processing resume data: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error processing resume data: {str(e)}"}
            )
        
        # Get available styles
        style_manager = StyleManager()
        style_manager.set_styles_directory(Path(STYLES_DIR))
        styles_dict = style_manager.get_styles()
        formatted_styles = style_manager.format_choices(styles_dict)
        
        # Return resume data and available styles
        return JSONResponse(content={
            "status": "success",
            "data": {
                "resume": {
                    "id": resume_id,
                    "user_id": current_user.id,
                    "created_at": stored_resume["created_at"],
                    "updated_at": stored_resume["updated_at"]
                },
                "available_styles": formatted_styles,
                "style_details": styles_dict
            },
            "message": "Ready for resume generation. Please select a style."
        })
        
    except Exception as e:
        logger.error(f"Error preparing for resume generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error preparing for resume generation: {str(e)}"
        )

@router.post("/clear-cache")
async def clear_cache(request: Request, current_user = Depends(get_current_user)):
    """
    Manually clear the PDF cache (admin only)
    """
    try:
        # Extract and set JWT token
        token = await get_token_from_request(request)
        if token:
            storage_service.set_auth_token(token)
        
        # Check if user is an admin (optional - implement your admin check)
        # For now, let any authenticated user clear their cache
        
        # Clear memory cache
        pdf_cache.clear()
        
        # Clear disk cache
        cache_dir = os.path.join(OUTPUT_DIR, "cache")
        if os.path.exists(cache_dir):
            # Remove and recreate the directory
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Cache cleared successfully"
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )