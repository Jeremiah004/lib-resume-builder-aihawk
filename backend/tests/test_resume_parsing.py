import os
import logging
from pathlib import Path
from app.utils.file_parser import extract_text_from_pdf, extract_text_from_docx
from lib_resume_builder_AIHawk import LLMResumer
import yaml

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_resume_parsing(file_path: str):
    """
    Test the resume parsing functionality.
    
    Args:
        file_path (str): Path to the resume file
    """
    try:
        # Determine file type
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            text = extract_text_from_docx(file_path)
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return
        
        if not text:
            logger.error("Failed to extract text from file")
            return
            
        # Initialize LLMResumer (you'll need to provide the actual API key)
        resumer = LLMResumer(os.getenv("GROQ_API_KEY"), None)
        
        # Parse the resume text
        resume_data = resumer.extract_resume_info(text)
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save the parsed data as YAML
        output_file = output_dir / f"{Path(file_path).stem}_parsed.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(resume_data, f, default_flow_style=False)
            
        logger.info(f"Resume parsed successfully. Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in resume parsing test: {str(e)}")

if __name__ == "__main__":
    # Test with a sample resume file
    test_file = r"C:\Users\JERRY\Music\twitter-sentiment-analysis\linkedIn_auto_jobs_applier_with_AI-main\Testing\resume_render_from_job_description\lib_resume_builder_AIHawk\backend\tests\__pycache__\Andrew Okolo Resume Koo.pdf"  # or .docx
    test_resume_parsing(test_file) 