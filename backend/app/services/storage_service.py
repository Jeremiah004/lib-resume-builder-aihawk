from supabase import create_client, Client
from app.core.config import SUPABASE_URL, SUPABASE_KEY
import logging
from datetime import datetime
import uuid
import json
import tempfile
import os
import base64

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.bucket_name = "pdf-resumes"
        self._auth_token = None
        
        # Try to ensure the bucket exists
        try:
            # Check if bucket exists by listing buckets
            buckets = self.supabase.storage.list_buckets()
            bucket_exists = any(bucket['name'] == self.bucket_name for bucket in buckets)
            
            if not bucket_exists:
                logger.info(f"Bucket {self.bucket_name} does not exist. Attempting to create it.")
                self.supabase.storage.create_bucket(self.bucket_name, {'public': True})
                logger.info(f"Successfully created bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Could not verify or create bucket: {str(e)}")
            # This is non-fatal, we'll continue even if this fails

    def set_auth_token(self, token):
        """Store the authentication token for logging purposes only"""
        self._auth_token = token
        logger.info("Authentication token stored (but not used directly)")

    async def store_resume(self, user_id: str, resume_data: dict) -> str:
        """
        Store a resume object in Supabase and return its ID
        """
        try:
            resume_id = str(uuid.uuid4())
            data = {
                "id": resume_id,
                "user_id": user_id,
                "resume_data": resume_data,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("resumes").insert(data).execute()
            return resume_id
            
        except Exception as e:
            logger.error(f"Error storing resume: {str(e)}")
            raise

    async def get_resume(self, resume_id: str, user_id: str) -> dict:
        """
        Retrieve a resume object from Supabase
        """
        try:
            result = self.supabase.table("resumes").select("*").eq("id", resume_id).eq("user_id", user_id).execute()
            if not result.data:
                raise ValueError("Resume not found")
            return result.data[0]
            
        except Exception as e:
            logger.error(f"Error retrieving resume: {str(e)}")
            raise

    async def store_pdf(self, resume_id: str, user_id: str, pdf_data: bytes, style_name: str) -> str:
        """
        Store a generated PDF in Supabase storage and record in database
        """
        try:
            # Generate unique filename - make sure the directory structure uses user_id first
            filename = f"{user_id}/{resume_id}/{style_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
            
            # Log the attempt
            logger.info(f"Attempting to upload file to bucket: {self.bucket_name}, path: {filename}")
            logger.info(f"User ID from request: {user_id}")
            logger.info(f"Using authenticated token: {bool(self._auth_token)}")
            
            # Try several upload methods to accommodate different versions of the Supabase client
            try:
                # Log more details about the environment
                logger.info(f"Supabase client type: {type(self.supabase)}")
                logger.info(f"Storage client type: {type(self.supabase.storage)}")
                
                # First approach - using the storage API directly
                result = self.supabase.storage.from_(self.bucket_name).upload(
                    filename,
                    pdf_data,
                    {"content-type": "application/pdf"}
                )
                logger.info(f"Upload successful with method 1: {result}")
            except Exception as e1:
                logger.warning(f"First upload method failed: {str(e1)}")
                try:
                    # Save the PDF to a temporary file first
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(pdf_data)
                        tmp_path = tmp.name
                    
                    # Try uploading the file from disk
                    with open(tmp_path, 'rb') as f:
                        result = self.supabase.storage.from_(self.bucket_name).upload(
                            filename,
                            f,
                            {"content-type": "application/pdf"}
                        )
                    
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    
                    logger.info(f"Upload successful with method 2 (from file): {result}")
                except Exception as e2:
                    logger.error(f"Second upload method also failed: {str(e2)}")
                    # Clean up temp file if it exists
                    tmp_path = tmp_path if 'tmp_path' in locals() else None
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    
                    # Return the base64 encoded PDF as a fallback
                    logger.warning("Falling back to returning base64 encoded PDF")
                    pdf_url = f"data:application/pdf;base64,{base64.b64encode(pdf_data).decode('utf-8')}"
                    return pdf_url
            
            # Get public URL
            pdf_url = self.supabase.storage.from_(self.bucket_name).get_public_url(filename)
            logger.info(f"Generated public URL: {pdf_url}")
            
            # Store metadata in database
            pdf_id = str(uuid.uuid4())
            data = {
                "id": pdf_id,
                "resume_id": resume_id,
                "style_name": style_name,
                "file_path": filename,
                "url": pdf_url,
                "created_at": datetime.utcnow().isoformat()
            }
            
            try:
                db_result = self.supabase.table("generated_pdfs").insert(data).execute()
                logger.info(f"Database insert response: {db_result}")
            except Exception as db_error:
                logger.error(f"Database insert failed, but file upload was successful: {str(db_error)}")
                # Continue anyway since we at least have the file
            
            return pdf_url
            
        except Exception as e:
            logger.error(f"Error storing PDF: {str(e)}")
            # Log more details about the error
            if hasattr(e, 'response') and hasattr(e.response, 'content'):
                logger.error(f"Error response content: {e.response.content}")
            
            # Return base64 encoded PDF as fallback
            logger.warning("Exception occurred. Falling back to returning base64 encoded PDF")
            return f"data:application/pdf;base64,{base64.b64encode(pdf_data).decode('utf-8')}"

    async def get_resume_pdfs(self, resume_id: str, user_id: str) -> list:
        """
        Get all PDFs generated for a resume
        """
        try:
            result = self.supabase.table("generated_pdfs").select("*").eq("resume_id", resume_id).execute()
            return result.data
            
        except Exception as e:
            logger.error(f"Error retrieving PDFs: {str(e)}")
            raise

    async def delete_resume(self, resume_id: str, user_id: str) -> None:
        """
        Delete a resume and all its associated PDFs
        """
        try:
            # Get all PDFs for this resume
            pdfs = await self.get_resume_pdfs(resume_id, user_id)
            
            # Delete PDFs from storage
            for pdf in pdfs:
                self.supabase.storage.from_(self.bucket_name).remove([pdf["file_path"]])
            
            # Delete PDF records from database
            self.supabase.table("generated_pdfs").delete().eq("resume_id", resume_id).execute()
            
            # Delete resume record
            self.supabase.table("resumes").delete().eq("id", resume_id).eq("user_id", user_id).execute()
            
        except Exception as e:
            logger.error(f"Error deleting resume: {str(e)}")
            raise 