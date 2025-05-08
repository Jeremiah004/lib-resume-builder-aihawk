import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import uuid
import os
import json
import random
from unittest.mock import MagicMock, patch
import asyncio

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Now import app modules after adding to path
from app.main import app
from app.core.auth import get_current_user
from app.services.storage_service import StorageService
from lib_resume_builder_AIHawk import StyleManager

# Utility function to create a coroutine that returns a value
def async_return(value):
    async def mock_coroutine(*args, **kwargs):
        return value
    return mock_coroutine

# Test client
client = TestClient(app)

# Test data
TEST_RESUME_PATH = Path(__file__).parent / "Andrew Okolo Resume Koo.pdf"
TEST_USER_ID = "3e30ec46-12e8-4e9a-8a56-220aec3f91c1"

# Get actual styles from StyleManager
def get_random_style():
    try:
        style_manager = StyleManager()
        style_manager.set_styles_directory(Path(os.environ.get('STYLES_DIR', 'styles')))
        styles = style_manager.get_styles()
        if styles:
            return random.choice(list(styles.keys()))
        return "Cloyola Grey"  # fallback if no styles found
    except Exception as e:
        print(f"Error getting styles: {e}")
        return "Cloyola Grey"  # fallback if there's an error

# Set TEST_STYLE_NAME to a random style from the style manager
TEST_STYLE_NAME = get_random_style()
print(f"Using random style for testing: {TEST_STYLE_NAME}")

# Mock resume data
MOCK_RESUME_ID = str(uuid.uuid4())
MOCK_RESUME_DATA = {
    "id": MOCK_RESUME_ID,
    "user_id": TEST_USER_ID,
    "resume_data": {
        "personal_information": {
            "name": "Andrew Okolo",
            "surname": "Okolo",  # Required field
            "email": "andrew.okolo@example.com",
            "phone": "123-456-7890",
            "phone_prefix": "+234",  # Required field
            "location": "Lagos, Nigeria",
            "country": "Nigeria",  # Required field
            "city": "Lagos",  # Required field
            "address": "123 Main Street",  # Required field
            "date_of_birth": "1990-01-01",  # Required field
            "website": "https://github.com/andrewokolo",
            "linkedin": "https://linkedin.com/in/andrewokolo"
        },
        "summary": "Experienced software engineer with expertise in Python and FastAPI",
        "skills": ["Python", "FastAPI", "Testing", "SQL", "Docker"],
        "experience": [
            {
                "company": "Tech Innovations",
                "position": "Senior Software Engineer",
                "start_date": "2020-01",
                "end_date": "Present",
                "description": "Developed and maintained backend services"
            }
        ],
        "education": [
            {
                "institution": "University of Lagos",
                "degree": "Bachelor of Science in Computer Science",
                "start_date": "2012",
                "end_date": "2016"
            }
        ]
    },
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-01T00:00:00Z"
}

# Mock storage service methods
async def mock_store_resume(user_id, resume_data):
    print(f"\n\n----- STORED RESUME DATA (Type: {type(resume_data)}) -----")
    print(json.dumps(resume_data, indent=2))
    print("---------------------------------------------\n\n")
    return MOCK_RESUME_ID

async def mock_get_resume(resume_id, user_id):
    print(f"\n\n----- RETRIEVED RESUME DATA -----")
    print(json.dumps(MOCK_RESUME_DATA, indent=2))
    print("---------------------------------------------\n\n")
    return MOCK_RESUME_DATA

async def mock_get_resume_pdfs(resume_id, user_id):
    print(f"\n\n----- RETRIEVED PDFs for Resume: {resume_id} -----")
    print("No PDFs generated yet")
    print("---------------------------------------------\n\n")
    return []

async def mock_store_pdf(resume_id, user_id, pdf_data, style_name):
    print(f"\n\n----- STORED PDF for Resume: {resume_id} -----")
    print(f"Style: {style_name}")
    print(f"PDF Size: {len(pdf_data)} bytes")
    print("---------------------------------------------\n\n")
    return "https://example.com/test.pdf"

async def mock_delete_resume(resume_id, user_id):
    print(f"\n\n----- DELETED Resume: {resume_id} -----")
    print("---------------------------------------------\n\n")
    return True

# Mock current user
async def mock_get_current_user():
    class MockUser:
        def __init__(self):
            self.id = TEST_USER_ID
    return MockUser()

# Apply mocks
app.dependency_overrides[get_current_user] = mock_get_current_user

# Create test fixtures
@pytest.fixture
def mock_storage_service():
    """Mock StorageService for testing"""
    with patch('app.routes.resume.storage_service') as mock_service:
        # Mock for getting a resume
        mock_service.get_resume.side_effect = async_return(MOCK_RESUME_DATA)
        
        # Mock for storing a resume
        mock_service.store_resume.side_effect = async_return(MOCK_RESUME_ID)
        
        # Mock for storing a PDF
        mock_service.store_pdf.side_effect = async_return("https://example.com/pdf/test.pdf")
        
        # Mock for deleting a resume
        mock_service.delete_resume.side_effect = async_return(True)
        
        # Mock for getting resume PDFs
        mock_service.get_resume_pdfs.side_effect = async_return([])
        
        # Mock the supabase table query chain for list_resumes
        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.execute.return_value.data = [MOCK_RESUME_DATA]
        mock_service.supabase.table.return_value = mock_table
        
        yield mock_service

@pytest.fixture
def test_resume_file():
    """Fixture to provide the test resume file"""
    try:
        with open(TEST_RESUME_PATH, "rb") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Test resume file not found at {TEST_RESUME_PATH}")
        # Return a small placeholder PDF content
        return b"%PDF-1.5\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 612 792]/Resources<<>>/Parent 2 0 R>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"

# Mock LLMResumer
@pytest.fixture(autouse=True)
def mock_llm_resumer():
    with patch('app.routes.resume.LLMResumer') as mock_llm:
        mock_instance = mock_llm.return_value
        
        # Define the behavior for extract_resume_info
        def fake_extract(text):
            print(f"\n\n----- EXTRACTED TEXT FROM RESUME (length: {len(text)}) -----")
            print(f"Sample text: {text[:200]}...") 
            print("Extracting structured resume data...")
            print(f"Return type: {type(MOCK_RESUME_DATA['resume_data'])}")
            print("---------------------------------------------\n\n")
            return MOCK_RESUME_DATA["resume_data"]
        
        mock_instance.extract_resume_info = MagicMock(side_effect=fake_extract)
        yield mock_llm

# Mock extract text functions
@pytest.fixture(autouse=True)
def mock_extractors():
    with patch('app.routes.resume.extract_text_from_pdf') as mock_pdf, \
         patch('app.routes.resume.extract_text_from_docx') as mock_docx:
        
        def fake_extract_pdf(path):
            print(f"\n\n----- EXTRACTING TEXT FROM PDF: {path} -----")
            sample_text = "Andrew Okolo\nSenior Software Engineer\nLagos, Nigeria\nandrew.okolo@example.com\n"
            print(f"Sample extracted text: {sample_text}")
            print("---------------------------------------------\n\n")
            return sample_text
            
        def fake_extract_docx(path):
            print(f"\n\n----- EXTRACTING TEXT FROM DOCX: {path} -----")
            sample_text = "Andrew Okolo\nSenior Software Engineer\nLagos, Nigeria\nandrew.okolo@example.com\n"
            print(f"Sample extracted text: {sample_text}")
            print("---------------------------------------------\n\n")
            return sample_text
        
        mock_pdf.side_effect = fake_extract_pdf
        mock_docx.side_effect = fake_extract_docx
        yield

# Mock StyleManager and FacadeManager
@pytest.fixture(autouse=True)
def mock_facade():
    # First patch StyleManager and create its mock instance
    with patch('app.routes.resume.StyleManager') as mock_style_cls:
        style_instance = mock_style_cls.return_value
        
        def fake_get_styles():
            print(f"\n\n----- GETTING AVAILABLE STYLES -----")
            styles = {
                TEST_STYLE_NAME: (f"{TEST_STYLE_NAME.lower().replace(' ', '_')}.css", "Author Name"),
                "Modern Clean": ("modern_clean.css", "Modern Author"),
                "Professional Blue": ("professional_blue.css", "Blue Author")
            }
            print(f"Available styles: {', '.join(styles.keys())}")
            print("---------------------------------------------\n\n")
            return styles
            
        style_instance.get_styles.side_effect = fake_get_styles
        
        def fake_format_choices(styles_dict):
            return [f"{style_name} (style author -> {author_link})" 
                    for style_name, (file_name, author_link) in styles_dict.items()]
            
        style_instance.format_choices.side_effect = fake_format_choices
        
        # Now patch the FacadeManager
        with patch('app.routes.resume.FacadeManager') as mock_facade_cls:
            facade_instance = mock_facade_cls.return_value
            facade_instance.set_style.return_value = None
            
            def fake_generate_pdf(job_description_url=None, job_description_text=None):
                print(f"\n\n----- GENERATING PDF WITH FACADE -----")
                print(f"Job description text: {job_description_text[:100] if job_description_text else 'None'}")
                print(f"Job description URL: {job_description_url}")
                print(f"Using style: {TEST_STYLE_NAME}")
                print("---------------------------------------------\n\n")
                return b"%PDF-1.5\nFake PDF content for testing"
                
            facade_instance.generate_pdf.side_effect = fake_generate_pdf
            
            yield mock_facade_cls

def test_upload_resume(test_resume_file, mock_storage_service):
    """Test resume upload and processing"""
    print(f"\n\n===== STARTING TEST: upload_resume =====\n\n")
    
    # Prepare file upload
    files = {
        "file": ("Andrew Okolo Resume Koo.pdf", test_resume_file, "application/pdf")
    }
    
    # Mock the LLMResumer.extract_resume_info method to return valid structured data
    proper_resume_data = {
        "personal_information": {
            "name": "Andrew Okolo",
            "surname": "Okolo",  # Required field
            "email": "andrew.okolo@example.com",
            "phone": "123-456-7890",
            "phone_prefix": "+234",  # Required field
            "location": "Lagos, Nigeria",
            "country": "Nigeria",  # Required field
            "city": "Lagos",  # Required field
            "address": "123 Main Street",  # Required field
            "date_of_birth": "1990-01-01",  # Required field
            "website": "https://github.com/andrewokolo",
            "linkedin": "https://linkedin.com/in/andrewokolo"
        },
        "summary": "Experienced software engineer with expertise in Python and FastAPI",
        "skills": ["Python", "FastAPI", "Testing", "SQL", "Docker"],
        "experience": [
            {
                "company": "Tech Innovations",
                "position": "Senior Software Engineer",
                "start_date": "2020-01",
                "end_date": "Present",
                "description": "Developed and maintained backend services"
            }
        ],
        "education": [
            {
                "institution": "University of Lagos",
                "degree": "Bachelor of Science in Computer Science",
                "start_date": "2012",
                "end_date": "2016"
            }
        ]
    }
    
    with patch('app.routes.resume.LLMResumer.extract_resume_info', return_value=proper_resume_data):
        # Make request
        response = client.post("/resume/upload", files=files)
        
        # Print response for debugging
        print(f"\n\n----- API RESPONSE -----")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response Body: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Raw Response Body: {response.content}")
            print(f"Error parsing response: {str(e)}")
        print("---------------------------------------------\n\n")
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "resume" in data["data"]
        assert data["data"]["resume"]["user_id"] == TEST_USER_ID

def test_generate_resume(test_resume_file, mock_storage_service):
    """Test resume generation with style"""
    print(f"\n\n===== STARTING TEST: generate_resume =====\n\n")
    
    # Create mock data
    resume_data = {
        "id": MOCK_RESUME_ID,
        "user_id": TEST_USER_ID,
        "resume_data": {
            "personal_information": {
                "name": "Andrew Okolo",
                "surname": "Okolo",
                "email": "andrew.okolo@example.com",
                "phone": "123-456-7890",
                "phone_prefix": "+234",
                "location": "Lagos, Nigeria",
                "country": "Nigeria",
                "city": "Lagos",
                "address": "123 Main Street",
                "date_of_birth": "1990-01-01",
                "website": "https://github.com/andrewokolo",
                "linkedin": "https://linkedin.com/in/andrewokolo"
            },
            "summary": "Experienced software engineer with expertise in Python and FastAPI",
            "skills": ["Python", "FastAPI", "Testing", "SQL", "Docker"],
            "experience": [
                {
                    "company": "Tech Innovations",
                    "position": "Senior Software Engineer",
                    "start_date": "2020-01",
                    "end_date": "Present",
                    "description": "Developed and maintained backend services"
                }
            ],
            "education": [
                {
                    "institution": "University of Lagos",
                    "degree": "Bachelor of Science in Computer Science",
                    "start_date": "2012",
                    "end_date": "2016"
                }
            ]
        },
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z"
    }
    
    # Create a partial mock setup that allows the Resume class to function
    with patch('app.routes.resume.FacadeManager') as mock_facade_cls, \
         patch('app.routes.resume.storage_service') as mock_store, \
         patch('app.routes.resume.create_resume_object') as mock_create_resume:
        
        # Configure mocks
        facade_instance = mock_facade_cls.return_value
        facade_instance.set_style.return_value = None
        facade_instance.generate_pdf.return_value = b"%PDF-1.5\nFake PDF content for testing"
        
        # Configure storage_service to return awaitable coroutines
        mock_store.get_resume = MagicMock(side_effect=async_return(resume_data))
        mock_store.store_pdf = MagicMock(side_effect=async_return("https://example.com/pdf/test.pdf"))
        
        # Mock the resume object creation
        mock_resume = MagicMock()
        mock_create_resume.return_value = mock_resume
        
        # Prepare generation request
        data = {
            "resume_id": MOCK_RESUME_ID,
            "style_name": TEST_STYLE_NAME,
            "job_description_text": "Looking for a software engineer with Python experience who is proficient in FastAPI and has experience with database design and implementation."
        }
        
        print(f"\n\n----- GENERATE REQUEST DATA -----")
        print(f"Resume ID: {MOCK_RESUME_ID}")
        print(f"Style Name: {TEST_STYLE_NAME}")
        print(f"Job Description: {data['job_description_text'][:50]}...")
        print("---------------------------------------------\n\n")
        
        # Make request
        response = client.post("/resume/generate", data=data)
        
        # Print response for debugging
        print(f"\n\n----- API RESPONSE -----")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response Body: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Raw Response Body: {response.content}")
            print(f"Error parsing response: {str(e)}")
        print("---------------------------------------------\n\n")
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "pdf" in data["data"]
        assert "url" in data["data"]["pdf"]
        assert data["data"]["pdf"]["style"] == TEST_STYLE_NAME

def test_list_resumes(mock_storage_service):
    """Test listing user's resumes"""
    print(f"\n\n===== STARTING TEST: list_resumes =====\n\n")
    
    # Create a list of test resumes
    test_resumes = [{
        "id": MOCK_RESUME_ID,
        "user_id": TEST_USER_ID,
        "resume_data": {
            "personal_information": {
                "name": "Andrew Okolo",
                "surname": "Okolo",
                "email": "andrew.okolo@example.com",
                "phone": "123-456-7890",
                "phone_prefix": "+234",
                "location": "Lagos, Nigeria",
                "country": "Nigeria",
                "city": "Lagos",
                "address": "123 Main Street",
                "date_of_birth": "1990-01-01",
                "website": "https://github.com/andrewokolo",
                "linkedin": "https://linkedin.com/in/andrewokolo"
            },
            "summary": "Experienced software engineer",
            "skills": ["Python", "FastAPI", "Testing"]
        },
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
    }]
    
    with patch('app.routes.resume.storage_service.supabase.table') as mock_table_method:
        # Setup the chain of method calls with proper return values
        mock_execute = MagicMock()
        mock_execute.execute.return_value.data = test_resumes
        
        mock_eq = MagicMock()
        mock_eq.eq.return_value = mock_execute
        
        mock_select = MagicMock()
        mock_select.select.return_value = mock_eq
        
        mock_table_method.return_value = mock_select
        
        # Also mock the get_resume_pdfs method
        with patch('app.routes.resume.storage_service.get_resume_pdfs', side_effect=async_return([])):
            # Make request
            response = client.get("/resume/list")
            
            # Print response for debugging
            print(f"\n\n----- API RESPONSE -----")
            print(f"Status Code: {response.status_code}")
            try:
                print(f"Response Body: {json.dumps(response.json(), indent=2)}")
            except Exception as e:
                print(f"Raw Response Body: {response.content}")
                print(f"Error parsing response: {str(e)}")
            print("---------------------------------------------\n\n")
            
            # Assert response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "resumes" in data["data"]
            assert len(data["data"]["resumes"]) > 0
            assert data["data"]["resumes"][0]["user_id"] == TEST_USER_ID

def test_delete_resume(mock_storage_service):
    """Test resume deletion"""
    print(f"\n\n===== STARTING TEST: delete_resume =====\n\n")
    
    # Create mock resume data for the get_resume call
    resume_data = {
        "id": MOCK_RESUME_ID,
        "user_id": TEST_USER_ID,
        "resume_data": {
            "personal_information": {
                "name": "Andrew Okolo",
                "surname": "Okolo",
                "email": "andrew.okolo@example.com",
                "phone": "123-456-7890",
                "phone_prefix": "+234",
                "location": "Lagos, Nigeria",
                "country": "Nigeria",
                "city": "Lagos",
                "address": "123 Main Street",
                "date_of_birth": "1990-01-01",
                "website": "https://github.com/andrewokolo",
                "linkedin": "https://linkedin.com/in/andrewokolo"
            },
            "summary": "Experienced software engineer",
            "skills": ["Python", "FastAPI", "Testing"]
        },
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z"
    }
    
    # Mock the specific service methods needed for this test
    with patch('app.routes.resume.storage_service.get_resume', side_effect=async_return(resume_data)), \
         patch('app.routes.resume.storage_service.delete_resume', side_effect=async_return(True)):
        
        # Make request
        response = client.delete(f"/resume/{MOCK_RESUME_ID}")
        
        # Print response for debugging
        print(f"\n\n----- API RESPONSE -----")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response Body: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Raw Response Body: {response.content}")
            print(f"Error parsing response: {str(e)}")
        print("---------------------------------------------\n\n")
        
        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert f"Resume {MOCK_RESUME_ID}" in data["message"]

def test_invalid_file_upload():
    """Test upload with invalid file type"""
    print(f"\n\n===== STARTING TEST: invalid_file_upload =====\n\n")
    
    # Prepare invalid file
    files = {
        "file": ("test.txt", b"invalid content", "text/plain")
    }
    
    # Make request
    response = client.post("/resume/upload", files=files)
    
    # Print response for debugging
    print(f"\n\n----- API RESPONSE -----")
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")
    print("---------------------------------------------\n\n")
    
    # Assert response
    assert response.status_code == 400
    assert "Only PDF and DOCX files are supported" in response.json()["detail"]

def test_invalid_style_generation(mock_storage_service):
    """Test generation with invalid style"""
    print(f"\n\n===== STARTING TEST: invalid_style_generation =====\n\n")
    
    # Create mock data
    resume_data = {
        "id": MOCK_RESUME_ID,
        "user_id": TEST_USER_ID,
        "resume_data": {
            "personal_information": {
                "name": "Andrew Okolo",
                "surname": "Okolo",
                "email": "andrew.okolo@example.com",
                "phone": "123-456-7890",
                "phone_prefix": "+234",
                "location": "Lagos, Nigeria",
                "country": "Nigeria",
                "city": "Lagos",
                "address": "123 Main Street",
                "date_of_birth": "1990-01-01",
                "website": "https://github.com/andrewokolo",
                "linkedin": "https://linkedin.com/in/andrewokolo"
            },
            "skills": ["Python", "FastAPI", "Testing"]
        },
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z"
    }
    
    # Create a partial mock setup that allows the Resume class to function
    with patch('app.routes.resume.FacadeManager') as mock_facade_cls, \
         patch('app.routes.resume.storage_service') as mock_store, \
         patch('app.routes.resume.create_resume_object') as mock_create_resume:
        
        # Configure FacadeManager mock to raise error for invalid style
        facade_instance = mock_facade_cls.return_value
        facade_instance.set_style.side_effect = ValueError("Style 'invalid_style' not found")
        
        # Configure storage_service to return awaitable coroutines
        mock_store.get_resume = MagicMock(side_effect=async_return(resume_data))
        
        # Mock the resume object creation
        mock_resume = MagicMock()
        mock_create_resume.return_value = mock_resume
        
        # Prepare generation request with invalid style
        data = {
            "resume_id": MOCK_RESUME_ID,
            "style_name": "invalid_style",
            "job_description_text": "Test job description"
        }
        
        print(f"\n\n----- GENERATE REQUEST DATA WITH INVALID STYLE -----")
        print(f"Resume ID: {MOCK_RESUME_ID}")
        print(f"Style Name: invalid_style (intentionally invalid)")
        print(f"Job Description: {data['job_description_text']}")
        print("---------------------------------------------\n\n")
        
        # Make request
        response = client.post("/resume/generate", data=data)
        
        # Print response for debugging
        print(f"\n\n----- API RESPONSE -----")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response Body: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Raw Response Body: {response.content}")
            print(f"Error parsing response: {str(e)}")
        print("---------------------------------------------\n\n")
        
        # Assert response
        assert response.status_code == 400
        assert "Style" in response.json()["detail"]

def test_nonexistent_resume_generation():
    """Test generation with nonexistent resume"""
    print(f"\n\n===== STARTING TEST: nonexistent_resume_generation =====\n\n")
    
    # Mock at the route level to properly simulate the error return
    with patch('app.routes.resume.storage_service.get_resume') as mock_get_resume:
        # Configure mock to return None for the resume (not found case)
        mock_get_resume.return_value = None
        
        # Prepare generation request with invalid resume_id
        nonexistent_id = "nonexistent_id"
        data = {
            "resume_id": nonexistent_id,
            "style_name": TEST_STYLE_NAME,
            "job_description_text": "Test job description"
        }
        
        print(f"\n\n----- GENERATE REQUEST DATA WITH NONEXISTENT RESUME -----")
        print(f"Resume ID: {nonexistent_id} (intentionally nonexistent)")
        print(f"Style Name: {TEST_STYLE_NAME}")
        print(f"Job Description: {data['job_description_text']}")
        print("---------------------------------------------\n\n")
        
        # Make request
        response = client.post("/resume/generate", data=data)
        
        # Print response for debugging
        print(f"\n\n----- API RESPONSE -----")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response Body: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Raw Response Body: {response.content}")
            print(f"Error parsing response: {str(e)}")
        print("---------------------------------------------\n\n")
        
        # Assert response
        assert response.status_code == 404
        assert "Resume not found" in response.json()["detail"]

def test_nonexistent_resume_deletion():
    """Test deletion of nonexistent resume"""
    print(f"\n\n===== STARTING TEST: nonexistent_resume_deletion =====\n\n")
    
    # Mock at the route level to handle the error correctly
    with patch('app.routes.resume.storage_service.get_resume') as mock_get_resume:
        # Configure mock to return None for the resume (not found case)
        mock_get_resume.return_value = None
        
        nonexistent_id = "nonexistent_id"
        print(f"\n\n----- DELETE REQUEST FOR NONEXISTENT RESUME -----")
        print(f"Resume ID: {nonexistent_id} (intentionally nonexistent)")
        print("---------------------------------------------\n\n")
        
        # Make request
        response = client.delete(f"/resume/{nonexistent_id}")
        
        # Print response for debugging
        print(f"\n\n----- API RESPONSE -----")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response Body: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Raw Response Body: {response.content}")
            print(f"Error parsing response: {str(e)}")
        print("---------------------------------------------\n\n")
        
        # Assert response
        assert response.status_code == 404
        assert "Resume not found" in response.json()["detail"]

def test_get_available_styles():
    """Test getting available resume styles"""
    print(f"\n\n===== STARTING TEST: get_available_styles =====\n\n")
    
    # Make request
    response = client.get("/resume/styles")
    
    # Print response for debugging
    print(f"\n\n----- API RESPONSE -----")
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")
    print("---------------------------------------------\n\n")
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "styles" in data["data"]
    assert "raw_styles" in data["data"]
    assert isinstance(data["data"]["styles"], list)
    if data["data"]["styles"]:
        assert "(" in data["data"]["styles"][0]
        assert "->" in data["data"]["styles"][0]
    assert isinstance(data["data"]["raw_styles"], dict)
    if data["data"]["raw_styles"]:
        style_name, (file_name, author_link) = next(iter(data["data"]["raw_styles"].items()))
        assert isinstance(style_name, str)
        assert isinstance(file_name, str)
        assert isinstance(author_link, str) 