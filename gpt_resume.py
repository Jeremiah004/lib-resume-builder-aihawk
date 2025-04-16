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
        self.resume_object = resume_object or Resume()

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
            
            Follow this exact structure:
            
            personal_info:
              name: "John"
              surname: "Doe"
              date_of_birth: "1990-01-01"
              country: "USA"
              city: "New York"
              address: "123 Main St"
              phone_prefix: "+1"
              phone: "555-1234"
              email: "john@example.com"
              github: "github.com/johndoe"
              linkedin: "linkedin.com/in/johndoe"
            
            education_details:
              - education_level: "Bachelor's"
                institution: "University of Example"
                field_of_study: "Computer Science"
                final_evaluation_grade: "3.8"
                start_date: "2010"
                year_of_completion: "2014"
                exam:
                  "Data Structures": "A"
                  "Algorithms": "A-"
            
            experience_details:
              - role: "Software Engineer"
                company: "Tech Corp"
                duration: "2014-2018"
                location: "San Francisco"
                description: "Developed web applications and led team of 5 developers"
                skills_acquired:
                  - "Python"
                  - "JavaScript"
            
            projects:
              - name: "Project A"
                description: "Description of project A"
                link: "github.com/johndoe/project-a"
            
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
            
            Return only the YAML structure, no additional text or explanations.
            """
            
            # Create a prompt template
            prompt = ChatPromptTemplate.from_template(extraction_prompt)
            
            # Create a chain with the LLM
            chain = prompt | self.llm_cheap | StrOutputParser()
            
            # Invoke the chain with the PDF text
            yaml_str = chain.invoke({"pdf_text": pdf_text})
            
            # Clean the response
            yaml_str = yaml_str.replace('```yml', '').replace('```yaml', '').replace('```', '').strip()
            
            # Remove any non-YAML content after the last valid YAML entry
            last_valid_line = None
            for line in yaml_str.split('\n'):
                if line.strip() and not line.startswith('Resume Text'):
                    last_valid_line = line
            if last_valid_line:
                yaml_str = yaml_str[:yaml_str.rindex(last_valid_line) + len(last_valid_line)]
            
            # Fix common YAML formatting issues
            yaml_str = yaml_str.replace('- role:', '\n  - role:')  # Fix experience details formatting
            yaml_str = yaml_str.replace('- description:', '\n    - description:')  # Fix key responsibilities formatting
            
            try:
                # Parse the YAML
                resume_data = yaml.safe_load(yaml_str)
            except yaml.YAMLError as e:
                logging.error(f"YAML parsing error: {e}")
                logging.error(f"YAML string that caused error: {yaml_str}")
                raise
            
            # Validate and clean the data
            cleaned_data = self._clean_resume_data(resume_data)
            
            # Set the resume object using model_validate
            self.resume_object = Resume.model_validate(cleaned_data)
            
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
        if 'personal_info' in data:
            # Handle empty URLs with default values
            github = data['personal_info'].get('github', '')
            linkedin = data['personal_info'].get('linkedin', '')
            
            # Validate and correct GitHub URL
            if not github or github.lower() in ['not available', 'not specified', 'https://not specified']:
                github = "https://github.com/placeholder"
            elif not github.startswith(('http://', 'https://')):
                github = f"https://github.com/{github}"
            
            # Validate and correct LinkedIn URL
            if not linkedin or linkedin.lower() in ['not available', 'not specified', 'https://not specified']:
                linkedin = "https://linkedin.com/in/placeholder"
            elif not linkedin.startswith(('http://', 'https://')):
                linkedin = f"https://linkedin.com/in/{linkedin}"
            
            # Ensure all strings are properly formatted
            cleaned_data['personal_info'] = {
                'name': str(data['personal_info'].get('name', '')),
                'surname': str(data['personal_info'].get('surname', '')),
                'date_of_birth': str(data['personal_info'].get('date_of_birth', '')),
                'country': str(data['personal_info'].get('country', '')),
                'city': str(data['personal_info'].get('city', '')),
                'address': str(data['personal_info'].get('address', '')),
                'phone_prefix': str(data['personal_info'].get('phone_prefix', '')),
                'phone': str(data['personal_info'].get('phone', '')),
                'email': str(data['personal_info'].get('email', '')),
                'github': github,
                'linkedin': linkedin
            }
        
        # Clean education details
        if 'education_details' in data:
            cleaned_data['education_details'] = []
            for edu in data['education_details']:
                # Handle year_of_completion
                year = edu.get('year_of_completion', '')
                try:
                    year = int(year) if year else 0
                except (ValueError, TypeError):
                    year = 0
                
                # Handle exam field - ensure it's a dictionary
                exam = edu.get('exam', {})
                if not isinstance(exam, dict) or not exam:
                    exam = {}
                
                cleaned_edu = {
                    'education_level': str(edu.get('education_level', '')),
                    'institution': str(edu.get('institution', '')),
                    'field_of_study': str(edu.get('field_of_study', '')),
                    'final_evaluation_grade': str(edu.get('final_evaluation_grade', '')),
                    'start_date': str(edu.get('start_date', '')),
                    'year_of_completion': year,
                    'exam': exam
                }
                cleaned_data['education_details'].append(cleaned_edu)
        
        # Clean experience details
        if 'experience_details' in data:
            cleaned_data['experience_details'] = []
            for exp in data['experience_details']:
                cleaned_exp = {
                    'role': str(exp.get('role', '')),
                    'company': str(exp.get('company', '')),
                    'duration': str(exp.get('duration', '')),
                    'location': str(exp.get('location', '')),
                    'description': str(exp.get('description', '')),
                    'skills_acquired': exp.get('skills_acquired', [])
                }
                cleaned_data['experience_details'].append(cleaned_exp)
        
        # Clean projects
        if 'projects' in data:
            cleaned_data['projects'] = []
            for proj in data['projects']:
                # Handle empty project links
                link = proj.get('link', '')
                if not link or link.lower() in ['not specified', 'not available']:
                    link = "https://github.com/placeholder/project"
                elif not link.startswith(('http://', 'https://')):
                    link = f"https://github.com/{link}"
                
                cleaned_proj = {
                    'name': str(proj.get('name', '')),
                    'description': str(proj.get('description', '')),
                    'link': link
                }
                cleaned_data['projects'].append(cleaned_proj)
        
        # Clean achievements
        if 'achievements' in data:
            cleaned_data['achievements'] = []
            for ach in data['achievements']:
                cleaned_ach = {
                    'name': str(ach.get('name', '')),
                    'description': str(ach.get('description', ''))
                }
                cleaned_data['achievements'].append(cleaned_ach)
        
        # Clean certifications
        if 'certifications' in data:
            cleaned_data['certifications'] = []
            for cert in data['certifications']:
                cleaned_cert = {
                    'name': str(cert.get('name', '')),
                    'description': str(cert.get('description', ''))
                }
                cleaned_data['certifications'].append(cleaned_cert)
        
        # Clean languages
        if 'languages' in data:
            cleaned_data['languages'] = []
            for lang in data['languages']:
                cleaned_lang = {
                    'language': str(lang.get('language', '')),
                    'proficiency': str(lang.get('proficiency', ''))
                }
                cleaned_data['languages'].append(cleaned_lang)
        
        # Clean interests
        if 'interests' in data:
            cleaned_data['interests'] = [str(interest) for interest in data['interests']]
        
        return cleaned_data

    @staticmethod
    def _preprocess_template_string(template: str) -> str:
        # Preprocess a template string to remove unnecessary indentation.
        return textwrap.dedent(template)

    def set_resume(self, resume_object, job_description=None):
        """Set the resume object and optionally a job description.
        
        Args:
            resume_object: The resume object to use
            job_description (str, optional): A job description to tailor the resume to
        """
        try:
            if not resume_object:
                raise ValueError("Resume object cannot be None")
            self.resume_object = resume_object
            self.job_description = job_description
        except Exception as e:
            logging.error(f"Error in set_resume: {str(e)}")
            raise

    def generate_header(self) -> str:
        try:
            print(self.resume_object.personal_info)
            print(self.resume_object.personal_info.name if self.resume_object.personal_info else "No name")

            if not self.resume_object.personal_info or not self.resume_object:
                logging.error("Personal information or job description not provided")
                return ""
            logging.debug(f"Starting header generation with personal_info: {self.resume_object.personal_info}")
            logging.debug(f"Using template: {self.strings.prompt_header}")
        
            header_prompt_template = self._preprocess_template_string(
                self.strings.prompt_header
            )
            
            logging.debug(f"Preprocessed template: {header_prompt_template}")
            prompt = ChatPromptTemplate.from_template(header_prompt_template)
            chain = prompt | self.llm_cheap | StrOutputParser()
            output = chain.invoke({
                "personal_info": self.resume_object.personal_info
            })
            logging.debug(f"Generated header: {output}")
            return output
        except Exception as e:
            logging.error(f"Error in generate_header: {str(e)}", exc_info=True)
            return ""

    def generate_education_section(self) -> str:
        education_prompt_template = self._preprocess_template_string(
            self.strings.prompt_education
        )
        prompt = ChatPromptTemplate.from_template(education_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "education_details": self.resume_object.education_details,
        })
        return output

    def generate_work_experience_section(self) -> str:
        work_experience_prompt_template = self._preprocess_template_string(
            self.strings.prompt_working_experience
        )
        prompt = ChatPromptTemplate.from_template(work_experience_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "experience_details": self.resume_object.experience_details
        })
        return output

    def generate_side_projects_section(self) -> str:
        side_projects_prompt_template = self._preprocess_template_string(
            self.strings.prompt_side_projects
        )
        prompt = ChatPromptTemplate.from_template(side_projects_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "projects": self.resume_object.projects
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
            "achievements": self.resume_object.achievements,
            "certifications": self.resume_object.certifications,
            "job_description": self.job_description
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
            "certifications": self.resume_object.certifications,
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

        if self.resume_object.experience_details:
            for exp in self.resume_object.experience_details:
                if exp.skills_acquired:
                    skills.update(exp.skills_acquired)

        if self.resume_object.education_details:
            for edu in self.resume_object.education_details:
                if edu.exam:
                    for exam in edu.exam:
                        skills.update(exam.keys())
        prompt = ChatPromptTemplate.from_template(additional_skills_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "languages": self.resume_object.languages,
            "interests": self.resume_object.interests,
            "skills": skills,
        })
        
        return output

    def generate_html_resume(self) -> str:
        # Define a list of functions to execute in parallel
        def header_fn():
            if self.resume_object.personal_info:
                return self.generate_header()
            return ""

        def education_fn():
            if self.resume_object.education_details:
                return self.generate_education_section()
            return ""

        def work_experience_fn():
            if self.resume_object.experience_details:
                return self.generate_work_experience_section()
            return ""

        def side_projects_fn():
            if self.resume_object.projects:
                return self.generate_side_projects_section()
            return ""

        def achievements_fn():
            if self.resume_object.achievements:
                return self.generate_achievements_section()
            return ""
        
        def certifications_fn():
            if self.resume_object.certifications:
                return self.generate_certifications_section()
            return ""

        def additional_skills_fn():
            if (self.resume_object.experience_details or self.resume_object.education_details or
                self.resume_object.languages or self.resume_object.interests):
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