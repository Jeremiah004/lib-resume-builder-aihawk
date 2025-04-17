import json
import os
import tempfile
import textwrap
import time
from datetime import datetime
from typing import Dict, List
from langchain_community.document_loaders import TextLoader
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from lib_resume_builder_AIHawk.config import global_config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re  # For regex parsing, especially in `parse_wait_time_from_error_message`
from requests.exceptions import HTTPError as HTTPStatusError  # Handling HTTP status errors
import random
import functools
from pathlib import Path
from langchain_groq import ChatGroq
import yaml
from lib_resume_builder_AIHawk.resume import Resume
load_dotenv()

log_folder = 'log'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Configura il file di log
log_file = os.path.join(log_folder, 'app.log')

# Configura il logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

# Add to the top of your file
import queue
import threading

# Create a request queue and rate limiter
request_queue = queue.Queue()
MAX_REQUESTS_PER_MINUTE = 10  # Adjust based on your API quota
request_semaphore = threading.Semaphore(MAX_REQUESTS_PER_MINUTE)

# Create a worker that processes requests from the queue
def request_worker():
    while True:
        task = request_queue.get()
        if task is None:  # Shutdown signal
            break
        # Process the task...
        request_queue.task_done()

# Start worker threads
workers = []
for _ in range(3):  # 3 concurrent workers
    t = threading.Thread(target=request_worker, daemon=True)
    t.start()
    workers.append(t)

class RateLimiter:
    def __init__(self, max_calls, period=60):
        self.calls = []
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
    
    def __call__(self, f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            with self.lock:
                now = time.time()
                # Remove calls older than period
                self.calls = [t for t in self.calls if now - t < self.period]
                
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    logging.info(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                    time.sleep(max(0, sleep_time))
                    
                self.calls.append(time.time())
            
            return f(*args, **kwargs)
        return wrapped
    
    
logger = logging.getLogger(__name__)

class LLMLogger:
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm

    @staticmethod
    def log_request(prompts, parsed_reply: Dict[str, Dict]):
        try:
            if global_config.LOG_OUTPUT_FILE_PATH is None:
                global_config.LOG_OUTPUT_FILE_PATH = Path("logs")
            log_path = Path(global_config.LOG_OUTPUT_FILE_PATH) if global_config.LOG_OUTPUT_FILE_PATH else Path("logs")
            log_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist                                                                                          )
            calls_log = log_path / "google_gemini_calls.json"
            if isinstance(prompts, StringPromptValue):
                prompts = prompts.text
            elif isinstance(prompts, Dict):
                # Convert prompts to a dictionary if they are not in the expected format
                prompts = {
                    f"prompt_{i+1}": prompt.content
                    for i, prompt in enumerate(prompts.messages)
                }
            else:
                prompts = {
                    f"prompt_{i+1}  ": prompt.content
                    for i, prompt in enumerate(prompts.messages)
                }

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Extract token usage details from the response
            # Safely extract token usage with defaults
            token_usage = parsed_reply.get("usage_metadata", {})
            output_tokens = token_usage.get("output_tokens", 0)
            input_tokens = token_usage.get("input_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            
            # Extract model details from the response
            response_metadata = parsed_reply.get("response_metadata", {})
            model_name = response_metadata.get("model_name", "Unknown_Model")
            prompt_price_per_token = 0.00000015
            completion_price_per_token = 0.0000006
            try:
                # Calculate the total cost of the API call
                total_cost = (float(input_tokens) * prompt_price_per_token) + (
                    float(output_tokens) * completion_price_per_token
                )
            except (TypeError, ValueError):
                total_cost = 0.0

            # Create a log entry with all relevant information
            log_entry = {
                "model": model_name,
                "time": current_time,
                "prompts": prompts,
                "replies": parsed_reply.get("content", ""),  # Response content
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": total_cost,
            }
            try:
                # Write the log entry to the log file in JSON format
                with open(calls_log, "a", encoding="utf-8") as f:
                    json_string = json.dumps(log_entry, ensure_ascii=False, indent=4)
                    f.write(json_string + "\n")
            except Exception as e:
                logging.error(f"Failed to write to log file: {e}")
        except Exception as e:
            logging.error(f"Error in log_request: {e}", exc_info=True)
            
class LoggerChatModel:

    def __init__(self, llm: ChatGroq):
        if not llm:
            raise ValueError("LLM instance cannot be None")
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def parse_llmresult(self, reply):
        """Parse the LLM result into a structured format.
        
        Args:
            reply: The raw reply from the LLM
            
        Returns:
            dict: A structured dictionary containing the parsed reply
        """
        try:
            if not reply:
                raise ValueError("Reply cannot be None")
            
            # Extract content
            content = reply.content if hasattr(reply, 'content') else str(reply)
            
            # Extract metadata
            metadata = getattr(reply, 'metadata', {})
            usage_metadata = metadata.get('usage_metadata', {})
            response_metadata = metadata.get('response_metadata', {})
            
            return {
                "content": content,
                "usage_metadata": usage_metadata,
                "response_metadata": response_metadata
            }
        except Exception as e:
            self.logger.error(f"Error parsing LLM result: {str(e)}")
            return {
                "content": str(reply),
                "usage_metadata": {},
                "response_metadata": {}
            }

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            raise ValueError("Messages cannot be empty")
            
        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                reply = self.llm.invoke(messages)
                
                if not reply:
                    self.logger.error("Received None response from LLM")
                    raise ValueError("Received None response from LLM")
                
                parsed_reply = self.parse_llmresult(reply)
                LLMLogger.log_request(prompts=messages, parsed_reply=parsed_reply)
                return reply
                
            except HTTPStatusError as err:
                if err.response.status_code == 429:
                    delay = min(60, base_delay * (2 ** attempt)) * (0.5 + random.random())
                    self.logger.warning(f"HTTP 429 Rate Limited: Retrying in {delay:.1f} seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Unexpected HTTP error: {err}", exc_info=True)
                    raise
            except Exception as e:
                self.logger.error(f"Error in LLM call: {str(e)}", exc_info=True)
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))

class LLMResumer:
    def __init__(self, groq_api_key, strings, resume_object=None):
        self.llm_cheap = LoggerChatModel(
            ChatGroq(
                model_name="llama-3.1-8b-instant", groq_api_key= groq_api_key, temperature=0.4
            )
        )
        self.strings = strings
        self.job_description = None

    def extract_resume_info(self, pdf_text: str) -> dict:
        """
        Extract structured information from PDF text and convert it to YAML format.
        
        Args:
            pdf_text (str): Raw text extracted from PDF
            
        Returns:
            dict: Structured resume information in YAML format
        """
        try:
            # Create a prompt for the LLM to extract structured information
            extraction_prompt = """
            Extract the following information from the resume text and format it as a YAML structure.
            Return ONLY the YAML content without any markdown formatting or code block markers.
            
            IMPORTANT: Do NOT use phrases like "Not provided", "Not specified", "N/A", or leave any field empty.
            Always provide actual content for each field using information from the resume or reasonable defaults.
            
            Follow this exact structure and ensure all fields are completely filled with valid values:
            
            personal_information:
            name: "John"
            surname: "Doe"
            date_of_birth: "1990-01-01"  # Use format YYYY-MM-DD or DD/MM/YYYY consistently
            country: "USA"
            city: "New York"
            address: "123 Main St"
            phone_prefix: "+1"
            phone: "555-1234"
            email: "john@example.com"
            github: "https://github.com/johndoe"  # Must be full URL with https://
            linkedin: "https://linkedin.com/in/johndoe"  # Must be full URL with https://
            
            education_details:
            - education_level: "Bachelor's"
                institution: "University of Example"
                field_of_study: "Computer Science"
                final_evaluation_grade: "3.8"
                start_date: "2010"
                year_of_completion: "2014"  # Must be a valid integer year or string representation
                exam:
                "Data Structures": "A"
                "Algorithms": "A-"
            
            experience_details:
            - position: "Software Engineer"
                company: "Tech Corp"
                employment_period: "2014-2018"
                location: "San Francisco"
                industry: "Technology"  # Industry field is required
                description: "Developed web applications and led team of 5 developers"
                skills_acquired:
                - "Python"
                - "JavaScript"
            
            projects:
            - name: "Project A"
                description: "Description of project A"
                link: "https://github.com/johndoe/project-a"  # Must be full URL with https://
            
            achievements:
            - name: "Best Employee 2017"
                description: "Awarded for outstanding performance"
            
            certifications:
            - name: "AWS Certified Developer"
                description: "Completed in 2018"
            
            languages:
            - language: "English"
                proficiency: "Native"
            
            interests:
            - "Programming"
            - "Reading"
            
            Resume Text:
            {pdf_text}
            
            Return only the YAML structure starting from personal information to interests in that order, no additional text or explanations.
            If any information is not found, Do Not Use Not Provided or Empty Value, use defaults:
            - For URLs, use proper https:// format
            - For dates of birth, use YYYY-MM-DD format
            - For years, use valid integer values like 2020
            - Always include industry field for experience
            remember, do not produce Not Provided
            """
            
            # Create a prompt template
            prompt = ChatPromptTemplate.from_template(extraction_prompt)
            
            # Create a chain with the LLM
            chain = prompt | self.llm_cheap | StrOutputParser()
            
            # Invoke the chain with the PDF text
            yaml_str = chain.invoke({"pdf_text": pdf_text})
            
            # Clean the response
            yaml_str = yaml_str.replace('```yml', '').replace('```yaml', '').replace('```', '').strip()
            
            try:
                # Parse the YAML
                resume_data = yaml.safe_load(yaml_str)
            except yaml.YAMLError as e:
                logging.error(f"YAML parsing error: {e}")
                logging.error(f"YAML string that caused error: {yaml_str}")
                raise
            
            # Validate and clean the data
            cleaned_data = self._clean_resume_data(resume_data)
            
            return cleaned_data
            
        except Exception as e:
            logging.error(f"Error in extract_resume_info: {str(e)}", exc_info=True)
            raise

    def _clean_resume_data(self, data: dict) -> dict:
        """
        Clean and validate the extracted resume data.
        
        Args:
            data (dict): Raw extracted data
            
        Returns:
            dict: Cleaned and validated data
        """
        cleaned_data = {}
        
        # Clean personal information
        if 'personal_information' in data:
            pi = data['personal_information']
            
            # Ensure required fields exist
            if 'date_of_birth' not in pi or not pi['date_of_birth'] or pi['date_of_birth'].lower() in ['not provided', 'not specified']:
                pi['date_of_birth'] = "1990-01-01"  # Default date
                
            # Handle URL fields
            github = pi.get('github', '')
            linkedin = pi.get('linkedin', '')
            
            # Validate and correct GitHub URL
            if not github or not isinstance(github, str) or github.lower() in ['not available', 'not specified', 'not provided']:
                github = "https://github.com/placeholder"
            elif not github.startswith(('http://', 'https://')):
                github = f"https://github.com/{github.strip('/').split('/')[-1]}"
            
            # Validate and correct LinkedIn URL
            if not linkedin or not isinstance(linkedin, str) or linkedin.lower() in ['not available', 'not specified', 'not provided']:
                linkedin = "https://linkedin.com/in/placeholder"
            elif not linkedin.startswith(('http://', 'https://')):
                linkedin = f"https://linkedin.com/in/{linkedin.strip('/').split('/')[-1]}"
            
            # Update with cleaned data
            cleaned_data['personal_information'] = {
                'name': str(pi.get('name', 'John')),
                'surname': str(pi.get('surname', 'Doe')),
                'date_of_birth': str(pi.get('date_of_birth', '1990-01-01')),
                'country': str(pi.get('country', '')),
                'city': str(pi.get('city', '')),
                'address': str(pi.get('address', '')),
                'phone_prefix': str(pi.get('phone_prefix', '')),
                'phone': str(pi.get('phone', '')),
                'email': str(pi.get('email', '')),
                'github': github,
                'linkedin': linkedin
            }
        
        # Clean education details
        cleaned_data['education_details'] = []
        if 'education_details' in data and isinstance(data['education_details'], list):
            for edu in data['education_details']:
                # Ensure year_of_completion is an integer or valid integer string
                year_of_completion = edu.get('year_of_completion', '2020')
                if not year_of_completion or year_of_completion in ['Not provided', 'not specified']:
                    year_of_completion = '2020'  # Default year
                
                # Try to ensure the year is a valid string that could be an integer
                try:
                    int(year_of_completion)  # Just to test if it's valid
                except (ValueError, TypeError):
                    year_of_completion = '2020'  # Default if invalid
                
                # Ensure exam dictionary exists
                exam_dict = edu.get('exam', {})
                if not isinstance(exam_dict, dict):
                    exam_dict = {}
                    
                cleaned_edu = {
                    'education_level': str(edu.get('education_level', '')),
                    'institution': str(edu.get('institution', '')),
                    'field_of_study': str(edu.get('field_of_study', '')),
                    'final_evaluation_grade': str(edu.get('final_evaluation_grade', '')),
                    'start_date': str(edu.get('start_date', '')),
                    'year_of_completion': year_of_completion,
                    'exam': exam_dict
                }
                cleaned_data['education_details'].append(cleaned_edu)
        
        # Clean experience details
        cleaned_data['experience_details'] = []
        if 'experience_details' in data and isinstance(data['experience_details'], list):
            for exp in data['experience_details']:
                # Ensure industry field exists
                industry = exp.get('industry', '')
                if not industry:
                    # Try to infer industry from company or position
                    company = exp.get('company', '').lower()
                    position = exp.get('position', '').lower()
                    
                    if any(tech in company or tech in position for tech in ['tech', 'software', 'it', 'computer']):
                        industry = 'Technology'
                    elif any(fin in company or fin in position for fin in ['finance', 'bank', 'invest']):
                        industry = 'Finance'
                    elif any(edu in company or edu in position for edu in ['education', 'university', 'school']):
                        industry = 'Education'
                    else:
                        industry = 'Business Services'  # Default industry
                
                # Handle skills acquired
                skills = exp.get('skills_acquired', [])
                if not isinstance(skills, list):
                    if isinstance(skills, str):
                        skills = [skill.strip() for skill in skills.split(',') if skill.strip()]
                    else:
                        skills = []
                
                cleaned_exp = {
                    'position': str(exp.get('position', '')),
                    'company': str(exp.get('company', '')),
                    'employment_period': str(exp.get('employment_period', '')),
                    'location': str(exp.get('location', '')),
                    'industry': str(industry),
                    'description': str(exp.get('description', '')),
                    'skills_acquired': skills
                }
                cleaned_data['experience_details'].append(cleaned_exp)
        
        # Clean projects
        cleaned_data['projects'] = []
        if 'projects' in data and isinstance(data['projects'], list):
            for proj in data['projects']:
                # Clean and validate link
                link = proj.get('link', '')
                if not link or not isinstance(link, str) or link.lower() in ['not available', 'not specified', 'not provided']:
                    link = f"https://github.com/placeholder/{proj.get('name', 'project').lower().replace(' ', '-')}"
                elif not link.startswith(('http://', 'https://')):
                    link = f"https://github.com/{link.strip('/').split('/')[-1]}"
                
                cleaned_proj = {
                    'name': str(proj.get('name', '')),
                    'description': str(proj.get('description', '')),
                    'link': link
                }
                cleaned_data['projects'].append(cleaned_proj)
        
        # Copy remaining sections with minimal cleaning
        for section in ['achievements', 'certifications', 'languages', 'interests']:
            if section in data:
                cleaned_data[section] = data[section]
        
        return cleaned_data
    

    @staticmethod
    def _preprocess_template_string(template: str) -> str:
        # Preprocess a template string to remove unnecessary indentation.
        return textwrap.dedent(template)
    

    def set_resume(self, resume):
        self.resume = resume

    def generate_header(self) -> str:
        header_prompt_template = self._preprocess_template_string(
            self.strings.prompt_header
        )
        prompt = ChatPromptTemplate.from_template(header_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "personal_information": self.resume.personal_information
        })
        return output

    def generate_education_section(self) -> str:
        education_prompt_template = self._preprocess_template_string(
            self.strings.prompt_education
        )
        prompt = ChatPromptTemplate.from_template(education_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "education_details": self.resume.education_details,
        })
        return output

    def generate_work_experience_section(self) -> str:
        work_experience_prompt_template = self._preprocess_template_string(
            self.strings.prompt_working_experience
        )
        prompt = ChatPromptTemplate.from_template(work_experience_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "experience_details": self.resume.experience_details
        })
        return output

    def generate_side_projects_section(self) -> str:
        side_projects_prompt_template = self._preprocess_template_string(
            self.strings.prompt_side_projects
        )
        prompt = ChatPromptTemplate.from_template(side_projects_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "projects": self.resume.projects
        })
        return output

    def generate_achievements_section(self) -> str:
        logging.debug("Starting achievements section generation")

        achievements_prompt_template = self._preprocess_template_string(
            self.strings.prompt_achievements
        )
        logging.debug(f"Achievements template: {achievements_prompt_template}")

        prompt = ChatPromptTemplate.from_template(achievements_prompt_template)
        logging.debug(f"Prompt: {prompt}")

        chain = prompt | self.llm_cheap | StrOutputParser()
        logging.debug(f"Chain created: {chain}")

        input_data = {
            "achievements": self.resume.achievements,
            "certifications": self.resume.certifications,
        }
        logging.debug(f"Input data for the chain: {input_data}")

        output = chain.invoke(input_data)
        logging.debug(f"Chain invocation result: {output}")

        logging.debug("Achievements section generation completed")
        return output

    def generate_certifications_section(self) -> str:
        logging.debug("Starting Certifications section generation")

        certifications_prompt_template = self._preprocess_template_string(
            self.strings.prompt_certifications
        )
        logging.debug(f"Certifications template: {certifications_prompt_template}")

        prompt = ChatPromptTemplate.from_template(certifications_prompt_template)
        logging.debug(f"Prompt: {prompt}")

        chain = prompt | self.llm_cheap | StrOutputParser()
        logging.debug(f"Chain created: {chain}")

        input_data = {
            "certifications": self.resume.certifications,
            "job_description": self.job_description
        }
        logging.debug(f"Input data for the chain: {input_data}")

        output = chain.invoke(input_data)
        logging.debug(f"Chain invocation result: {output}")

        logging.debug("Certifications section generation completed")
        return output
    
    

    def generate_additional_skills_section(self) -> str:
        additional_skills_prompt_template = self._preprocess_template_string(
            self.strings.prompt_additional_skills
        )
        
        skills = set()

        if self.resume.experience_details:
            for exp in self.resume.experience_details:
                if exp.skills_acquired:
                    skills.update(exp.skills_acquired)

        if self.resume.education_details:
            for edu in self.resume.education_details:
                if edu.exam:
                    for exam in edu.exam:
                        skills.update(exam.keys())
        prompt = ChatPromptTemplate.from_template(additional_skills_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "languages": self.resume.languages,
            "interests": self.resume.interests,
            "skills": skills,
        })
        
        return output

    def generate_html_resume(self) -> str:
        # Define a list of functions to execute in parallel
        def header_fn():
            if self.resume.personal_information:
                return self.generate_header()
            return ""

        def education_fn():
            if self.resume.education_details:
                return self.generate_education_section()
            return ""

        def work_experience_fn():
            if self.resume.experience_details:
                return self.generate_work_experience_section()
            return ""

        def side_projects_fn():
            if self.resume.projects:
                return self.generate_side_projects_section()
            return ""

        def achievements_fn():
            if self.resume.achievements:
                return self.generate_achievements_section()
            return ""
        
        def certifications_fn():
            if self.resume.certifications:
                return self.generate_certifications_section()
            return ""

        def additional_skills_fn():
            if (self.resume.experience_details or self.resume.education_details or
                self.resume.languages or self.resume.interests):
                return self.generate_additional_skills_section()
            return ""

        # Create a dictionary to map the function names to their respective callables
        functions = {
            "header": header_fn,
            "education": education_fn,
            "work_experience": work_experience_fn,
            "side_projects": side_projects_fn,
            "achievements": achievements_fn,
            "certifications": certifications_fn,
            "additional_skills": additional_skills_fn,
        }

        # Use ThreadPoolExecutor to run the functions in parallel
        with ThreadPoolExecutor() as executor:
            future_to_section = {executor.submit(fn): section for section, fn in functions.items()}
            results = {}
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    result = future.result()
                    if result:
                        results[section] = result
                except Exception as exc:
                    print(f'{section} ha generato un\'eccezione: {exc}')
        full_resume = "<body>\n"
        full_resume += f"  {results.get('header', '')}\n"
        full_resume += "  <main>\n"
        full_resume += f"    {results.get('education', '')}\n"
        full_resume += f"    {results.get('work_experience', '')}\n"
        full_resume += f"    {results.get('side_projects', '')}\n"
        full_resume += f"    {results.get('achievements', '')}\n"
        full_resume += f"    {results.get('certifications', '')}\n"
        full_resume += f"    {results.get('additional_skills', '')}\n"
        full_resume += "  </main>\n"
        full_resume += "</body>"
        return full_resume