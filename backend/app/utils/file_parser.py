import logging
from typing import Optional
import PyPDF2
from docx import Document
import os

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """
    Extract text from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Extracted text or None if extraction fails
    """
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_docx(file_path: str) -> Optional[str]:
    """
    Extract text from a DOCX file.
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        Optional[str]: Extracted text or None if extraction fails
    """
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        return None

# Note: For production, we might want to:
# 1. Add more robust error handling
# 2. Implement retry mechanisms for large files
# 3. Add file size limits
# 4. Add file type validation
# 5. Consider using more advanced PDF parsing libraries for better text extraction
# 6. Add support for more file formats
# 7. Implement caching for frequently accessed files
# 8. Add progress tracking for large files
# 9. Consider using async operations for better performance
# 10. Add support for encrypted PDFs 