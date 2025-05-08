import os
import json
import asyncio
import yaml
from pathlib import Path
import uuid
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the backend directory to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import necessary modules
from app.services.storage_service import StorageService
from app.utils.file_parser import extract_text_from_pdf, extract_text_from_docx
from lib_resume_builder_AIHawk import LLMResumer, Resume
from app.core.config import GROQ_API_KEY

# Test data
TEST_RESUME_PATH = Path(__file__).parent / "Andrew Okolo Resume Koo.pdf"
TEST_USER_ID = "3e30ec46-12e8-4e9a-8a56-220aec3f91c1"  # Use a valid UUID format

async def test_resume_upload_flow():
    """
    Test the entire flow of uploading a resume:
    1. Read the resume file
    2. Extract text from the resume
    3. Parse the text into structured resume data
    4. Store the resume data
    5. Retrieve the stored resume
    6. Create a Resume object from the data
    7. Print data types and structure at each step
    """
    logger.info("Starting resume upload test")
    
    # Step 1: Read the resume file
    try:
        with open(TEST_RESUME_PATH, "rb") as f:
            resume_content = f.read()
        logger.info(f"Successfully read resume file: {TEST_RESUME_PATH}")
        logger.info(f"File size: {len(resume_content)} bytes")
        logger.info(f"File type: {TEST_RESUME_PATH.suffix}")
    except FileNotFoundError:
        logger.error(f"Resume file not found: {TEST_RESUME_PATH}")
        logger.info("Creating a dummy file for testing")
        Path(TEST_RESUME_PATH.parent).mkdir(parents=True, exist_ok=True)
        with open(TEST_RESUME_PATH, "wb") as f:
            f.write(b"%PDF-1.5\nTest resume content for Andrew Okolo")
        with open(TEST_RESUME_PATH, "rb") as f:
            resume_content = f.read()
    
    # Step 2: Extract text from the resume
    if TEST_RESUME_PATH.suffix.lower() == '.pdf':
        text = extract_text_from_pdf(TEST_RESUME_PATH)
    elif TEST_RESUME_PATH.suffix.lower() == '.docx':
        text = extract_text_from_docx(TEST_RESUME_PATH)
    else:
        # For testing, create some dummy text if file extension is not recognized
        text = """
        Andrew Okolo
        Senior Software Engineer
        Lagos, Nigeria
        
        Skills: Python, FastAPI, React, SQL
        
        Experience:
        - Senior Software Engineer at Tech Innovations (2020-Present)
        - Software Developer at CodeCorp (2018-2020)
        
        Education:
        - Bachelor of Science in Computer Science, University of Lagos (2012-2016)
        """
    
    logger.info(f"Extracted text from resume (first 100 chars): {text[:100]}...")
    logger.info(f"Extracted text type: {type(text)}")
    
    # Step 3: Parse the text into structured resume data
    logger.info("Initializing LLMResumer and extracting resume information")
    resumer = LLMResumer(GROQ_API_KEY, None)
    resume_data = resumer.extract_resume_info(text)
    
    logger.info(f"Resume data type: {type(resume_data)}")
    logger.info(f"Resume data structure: {json.dumps(resume_data, indent=2)}")
    
    # Step 4: Store the resume data
    logger.info("Storing resume data in Supabase")
    storage_service = StorageService()
    
    try:
        resume_id = await storage_service.store_resume(
            user_id=TEST_USER_ID,
            resume_data=resume_data
        )
        logger.info(f"Resume stored with ID: {resume_id}")
        
        # Step 5: Retrieve the stored resume
        logger.info(f"Retrieving stored resume with ID: {resume_id}")
        stored_resume = await storage_service.get_resume(resume_id, TEST_USER_ID)
        
        logger.info(f"Retrieved resume type: {type(stored_resume)}")
        logger.info(f"Retrieved resume structure: {json.dumps(stored_resume, indent=2)}")
        
        # Step 6: Convert the dictionary to a YAML string before creating Resume object
        logger.info("Converting dictionary to YAML string")
        yaml_str = yaml.dump(stored_resume["resume_data"])
        logger.info(f"YAML string created (first 100 chars): {yaml_str[:100]}...")
        
        # Create Resume object from YAML string
        logger.info("Creating Resume object from YAML string")
        resume_object = Resume(yaml_str)
        
        logger.info(f"Resume object type: {type(resume_object)}")
        logger.info(f"Resume object attributes: {dir(resume_object)}")
        
        # Inspect some attributes/methods of the Resume object instead of to_dict()
        try:
            # Try different ways to access the resume data
            if hasattr(resume_object, "__dict__"):
                logger.info(f"Resume object __dict__: {resume_object.__dict__}")
            elif hasattr(resume_object, "model_dump"):
                logger.info(f"Resume object model_dump: {resume_object.model_dump()}")
            elif hasattr(resume_object, "dict"):
                logger.info(f"Resume object dict: {resume_object.dict()}")
            else:
                logger.info("Cannot find standard data extraction methods on Resume object")
                # List available attributes
                attributes = [attr for attr in dir(resume_object) if not attr.startswith('_')]
                logger.info(f"Available attributes: {attributes}")
        except Exception as e:
            logger.warning(f"Error accessing Resume object data: {str(e)}")
        
        # Step 7: Print success message with data type information
        logger.info(f"SUCCESS: Resume flow completed successfully")
        logger.info(f"Resume ID: {resume_id}")
        logger.info(f"Resume data type in storage: {type(stored_resume['resume_data'])}")
        logger.info(f"Resume object type: {type(resume_object)}")
        
        return resume_id, stored_resume, resume_object
        
    except Exception as e:
        logger.error(f"Error in resume upload flow: {str(e)}")
        raise
    
if __name__ == "__main__":
    # Run the test
    logger.info("==== STARTING RESUME UPLOAD TEST ====")
    
    try:
        # Run the async test using asyncio
        resume_id, stored_resume, resume_object = asyncio.run(test_resume_upload_flow())
        
        print("\n\n==== TEST RESULTS ====")
        print(f"Resume ID: {resume_id}")
        print(f"Stored Resume Data Type: {type(stored_resume['resume_data'])}")
        print(f"Resume Object Type: {type(resume_object)}")
        print("\nData Structure Sample (from storage, not Resume object):")
        print(json.dumps(stored_resume["resume_data"], indent=2)[:500] + "...")
        print("\n==== TEST COMPLETED SUCCESSFULLY ====")
        
    except Exception as e:
        print(f"\n\n==== TEST FAILED ====")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 