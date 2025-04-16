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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from lib_resume_builder_AIHawk.config import global_config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re  # For regex parsing, especially in `parse_wait_time_from_error_message`
from requests.exceptions import HTTPError as HTTPStatusError  # Handling HTTP status errors
import openai
from langchain_groq import ChatGroq
from huggingface_hub import HfApi, login
from pathlib import Path
import random
from bs4 import BeautifulSoup
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from tiktoken import encoding_for_model

# Add requirements for pip
REQUIREMENTS = [
    "langchain-huggingface>=0.0.5",
    "optree>=0.13.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
]

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

logger = logging.getLogger(__name__)

def get_huggingface_token():
    """Get Hugging Face token from environment or config"""
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.warning("HUGGINGFACE_TOKEN not found in environment variables")
        return None
    return token

def initialize_embeddings():
    """Initialize Hugging Face embeddings with proper authentication and caching"""
    try:
        token = get_huggingface_token()
        if token:
            login(token=token)
            logger.info("Successfully authenticated with Hugging Face")
        else:
            logger.warning("No Hugging Face token found, trying without authentication")
        
        # Set up caching directory
        cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
                'cache_dir': cache_dir,
            },
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=cache_dir
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        raise

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
        self.llm = llm
        self.logger = logging.getLogger(__name__)  # Ensure logger is set
        self.logger.setLevel(logging.INFO)  


    def __call__(self, messages: List[Dict[str, str]]) -> str:
        max_retries = 5  # Reduce from 15 to prevent excessive waiting
        base_delay = 5  # Start with 5 seconds instead of 10

        for attempt in range(max_retries):
            try:
                reply = self.llm.invoke(messages)
                
                if reply is None:
                    self.logger.error("Received None response from LLm")
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
                    break  # Exit loop for non-retryable HTTP errors
            except Exception as e:
                delay = min(60, base_delay * (2 ** attempt)) * (0.5 + random.random())
                self.logger.error(f"Unexpected error: {str(e)}, retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)

        self.logger.critical("Failed to get a response from the model after multiple attempts.")
        return AIMessage(content="[Content generation failed due to API limits. Please try again later.]") # Return an AIMessage


    def parse_llmresult(self, llmresult: AIMessage) -> Dict[str, Dict]:
        try:
            # Handle None or missing attributes safely
            content = ""
            response_metadata = {}
            id_ = ""
            usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            # Extract content safely
            if hasattr(llmresult, "content") and llmresult.content is not None:
                content = str(llmresult.content)
            
            # Extract response_metadata safely
            if hasattr(llmresult, "response_metadata") and llmresult.response_metadata is not None:
                response_metadata = llmresult.response_metadata
            
            # Extract id safely
            if hasattr(llmresult, "id") and llmresult.id is not None:
                id_ = str(llmresult.id)
            
            # Extract usage_metadata safely
            if hasattr(llmresult, "usage_metadata") and llmresult.usage_metadata is not None:
                if isinstance(llmresult.usage_metadata, dict):
                    usage_metadata = llmresult.usage_metadata
                
            parsed_result = {
                "content": content,
                "response_metadata": {
                    "model_name": str(response_metadata.get("model_name", "")) if response_metadata else "",
                    "system_fingerprint": str(response_metadata.get("system_fingerprint", "")) if response_metadata else "",
                    "finish_reason": str(response_metadata.get("finish_reason", "")) if response_metadata else "",
                    "logprobs": response_metadata.get("logprobs", None) if response_metadata else None,
                },
                "id": id_,
                "usage_metadata": {
                    "input_tokens": int(usage_metadata.get("input_tokens", 0)) if usage_metadata else 0,
                    "output_tokens": int(usage_metadata.get("output_tokens", 0)) if usage_metadata else 0,
                    "total_tokens": int(usage_metadata.get("total_tokens", 0)) if usage_metadata else 0,
                },
            }
            return parsed_result
        except Exception as e:
            logging.error(f"Error parsing LLM result: {e}", exc_info=True)
            # Return a safe fallback result
            return {
                "content": str(llmresult) if llmresult is not None else "",
                "response_metadata": {"model_name": "unknown"},
                "id": "",
                "usage_metadata": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }


    def parse_wait_time_from_error_message(self, error_message: str) -> int:
        # Extract wait time from error message
        match = re.search(r"Please try again in (\d+)([smhd])", error_message)
        if match:
            value, unit = match.groups()
            value = int(value)
            if unit == "s":
                return value
            elif unit == "m":
                return value * 60
            elif unit == "h":
                return value * 3600
            elif unit == "d":
                return value * 86400
        # Default wait time if not found
        return 30


class LLMResumeJobDescription:
    def __init__(self, groq_api_key, strings):
        self.llm_cheap = LoggerChatModel(ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.4))
        try:
            self.llm_embeddings = initialize_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            # Fallback to a simpler approach without embeddings
            self.llm_embeddings = None
        self.strings = strings
        # Initialize token counter using GPT-3.5-turbo's tokenizer (compatible with most models)
        self.token_counter = TokenCounter()

    @staticmethod
    def _preprocess_template_string(template: str) -> str:
        # Preprocess a template string to remove unnecessary indentation.
        return textwrap.dedent(template)

    def set_resume(self, resume):
        self.resume = resume

    def set_job_description_from_url(self, url_job_description):
        from lib_resume_builder_AIHawk.utils import create_driver_selenium
        driver = create_driver_selenium()
        driver.get(url_job_description)
        time.sleep(3)
        body_element = driver.find_element("tag name", "body")
        response = body_element.get_attribute("outerHTML")
        driver.quit()
        
        # Clean HTML content before processing
        soup = BeautifulSoup(response, 'html.parser')
        # Get only the text content
        cleaned_text = soup.get_text(separator=' ', strip=True)
        
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(cleaned_text)
            temp_file_path = temp_file.name
            
        try:
            loader = TextLoader(temp_file_path, encoding="utf-8", autodetect_encoding=True)
            document = loader.load()
        finally:
            os.remove(temp_file_path)
            
        # Use very small chunks to stay well under token limits
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
        all_splits = text_splitter.split_documents(document)
        
        # Process in very small batches with strict token counting
        processed_sections = []
        current_batch = []
        current_tokens = 0
        max_tokens_per_request = 2000  # Conservative limit
        
        for chunk in all_splits:
            chunk_text = chunk.page_content
            chunk_tokens = self.token_counter.estimate_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > max_tokens_per_request:
                # Process current batch
                if current_batch:
                    try:
                        result = self._process_batch(current_batch)
                        if result:
                            processed_sections.append(result)
                        # Reset batch
                        current_batch = []
                        current_tokens = 0
                        # Add delay between batches
                        time.sleep(10)  # 10 second delay between batches
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
            
            current_batch.append(chunk_text)
            current_tokens += chunk_tokens
        
        # Process any remaining chunks
        if current_batch:
            try:
                result = self._process_batch(current_batch)
                if result:
                    processed_sections.append(result)
            except Exception as e:
                logger.error(f"Error processing final batch: {str(e)}")
        
        # Combine and summarize results
        combined_text = " ".join(processed_sections)
        
        try:
            # Final summarization with very strict token limit
            final_prompt = """
            Create a brief summary of this job description.
            Focus only on the most essential requirements.
            Keep your response under 300 words.
            
            Job Details:
            {text}
            
            Summary:
            """
            
            # Check if final text is within token limits
            prompt_tokens = self.token_counter.estimate_tokens(final_prompt)
            text_tokens = self.token_counter.estimate_tokens(combined_text)
            
            if prompt_tokens + text_tokens > max_tokens_per_request:
                # Truncate the combined text to fit within limits
                while prompt_tokens + text_tokens > max_tokens_per_request:
                    combined_text = combined_text[:int(len(combined_text) * 0.8)]  # Reduce by 20%
                    text_tokens = self.token_counter.estimate_tokens(combined_text)
            
            prompt = PromptTemplate(template=final_prompt, input_variables=["text"])
            chain = prompt | self.llm_cheap | StrOutputParser()
            
            # Add delay before final summarization
            time.sleep(5)
            
            final_summary = chain.invoke({"text": combined_text})
            self.job_description = final_summary
            
        except Exception as e:
            logger.error(f"Error in final summarization: {str(e)}")
            # Use a safe subset of the processed sections
            self.job_description = combined_text[:1000]  # Use first 1000 characters as fallback
        
        return self.job_description

    def _process_batch(self, batch_texts):
        """Process a batch of texts with strict token limiting."""
        try:
            combined_text = " ".join(batch_texts)
            prompt = """
            Summarize the key points from this section.
            Focus on requirements and responsibilities.
            Keep your response under 100 words.
            
            Section:
            {text}
            
            Summary:
            """
            
            # Verify total tokens
            total_tokens = self.token_counter.estimate_tokens(prompt + combined_text)
            if total_tokens > 2000:  # Very conservative limit
                # Truncate text to fit
                while total_tokens > 2000:
                    combined_text = combined_text[:int(len(combined_text) * 0.8)]
                    total_tokens = self.token_counter.estimate_tokens(prompt + combined_text)
            
            prompt_template = PromptTemplate(template=prompt, input_variables=["text"])
            chain = prompt_template | self.llm_cheap | StrOutputParser()
            
            return chain.invoke({"text": combined_text})
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return None

    def set_job_description_from_text(self, job_description_text):
        prompt = ChatPromptTemplate.from_template(self.strings.summarize_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({"text": job_description_text})
        self.job_description = output
    
    def generate_header(self) -> str:
        header_prompt_template = self._preprocess_template_string(
            self.strings.prompt_header
        )
        prompt = ChatPromptTemplate.from_template(header_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "personal_information": self.resume.personal_information,
            "job_description": self.job_description
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
            "job_description": self.job_description
        })
        return output

    def generate_work_experience_section(self) -> str:
        work_experience_prompt_template = self._preprocess_template_string(
            self.strings.prompt_working_experience
        )
        prompt = ChatPromptTemplate.from_template(work_experience_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "experience_details": self.resume.experience_details,
            "job_description": self.job_description
        })
        return output

    def generate_side_projects_section(self) -> str:
        side_projects_prompt_template = self._preprocess_template_string(
            self.strings.prompt_side_projects
        )
        prompt = ChatPromptTemplate.from_template(side_projects_prompt_template)
        chain = prompt | self.llm_cheap | StrOutputParser()
        output = chain.invoke({
            "projects": self.resume.projects,
            "job_description": self.job_description
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
            "job_description": self.job_description
        })
        return output


    def generate_html_resume(self) -> str:
        def header_fn():
            if self.resume.personal_information and self.job_description:
                return self.generate_header()
            return ""

        def education_fn():
            if self.resume.education_details and self.job_description:
                return self.generate_education_section()
            return ""

        def work_experience_fn():
            if self.resume.experience_details and self.job_description:
                return self.generate_work_experience_section()
            return ""

        def side_projects_fn():
            if self.resume.projects and self.job_description:
                return self.generate_side_projects_section()
            return ""

        def achievements_fn():
            if self.resume.achievements and self.job_description:
                return self.generate_achievements_section()
            return ""
        
        def certifications_fn():
            if self.resume.certifications and self.job_description:
                return self.generate_certifications_section()
            return ""

        def additional_skills_fn():
            if (self.resume.experience_details or self.resume.education_details or
                self.resume.languages or self.resume.interests) and self.job_description:
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
                    logging.debug(f'{section} generated 1 exc: {exc}')
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

    def _process_without_embeddings(self, chunks):
        """Process job description without using embeddings."""
        logger.info(f"Processing job description using direct summarization (chunks: {len(chunks)})")
        
        # Combine chunks into sections of approximately 1000 tokens
        sections = []
        current_section = ""
        current_tokens = 0
        
        for chunk in chunks:
            chunk_text = chunk.page_content
            chunk_tokens = self.token_counter.estimate_tokens(chunk_text)
            
            # If adding this chunk would exceed our section size, start a new section
            if current_tokens + chunk_tokens > 1000:
                if current_section:
                    sections.append(current_section)
                current_section = chunk_text
                current_tokens = chunk_tokens
            else:
                current_section += "\n" + chunk_text
                current_tokens += chunk_tokens
        
        # Add the last section if not empty
        if current_section:
            sections.append(current_section)
        
        logger.info(f"Created {len(sections)} sections for summarization")
        
        # Process each section sequentially with proper rate limiting
        processed_sections = []
        
        for i, section in enumerate(sections):
            try:
                logger.info(f"Processing section {i+1}/{len(sections)}")
                
                # Create a focused prompt for this section
                prompt = PromptTemplate(
                    template="""
                    Summarize the key points from this section of the job description.
                    Focus on requirements, responsibilities, and qualifications.
                    Keep your response under 200 words.
                    
                    Section:
                    {text}
                    
                    Summary:
                    """,
                    input_variables=["text"]
                )
                
                # Estimate tokens for this request
                prompt_tokens = self.token_counter.estimate_tokens(section)
                if prompt_tokens > self.rate_limiter.max_tokens_per_request:
                    logger.warning(f"Section {i+1} too large ({prompt_tokens} tokens), splitting further")
                    # Split into smaller subsections
                    subsections = textwrap.wrap(section, width=1000)  # Approximate token limit
                    section_results = []
                    for sub in subsections:
                        try:
                            chain = prompt | self.llm_cheap | StrOutputParser()
                            result = chain.invoke({"text": sub})
                            section_results.append(result)
                        except Exception as e:
                            if "413" in str(e) or "too large" in str(e):
                                logger.error(f"Token limit exceeded even for subsection: {str(e)}")
                                continue
                            raise
                    processed_sections.extend(section_results)
                else:
                    # Process normal-sized section
                    chain = prompt | self.llm_cheap | StrOutputParser()
                    result = chain.invoke({"text": section})
                    processed_sections.append(result)
                
                # Add delay between sections to respect rate limits
                if i < len(sections) - 1:
                    time.sleep(5)  # Basic rate limiting between sections
                
            except Exception as e:
                logger.error(f"Error processing section {i+1}: {str(e)}")
                if "413" in str(e) or "too large" in str(e):
                    logger.error("Token limit exceeded, skipping section")
                    continue
                raise
        
        # Combine processed sections and create final summary
        combined_text = "\n\n".join(processed_sections)
        
        try:
            # Final summarization with strict token limit
            final_prompt = PromptTemplate(
                template="""
                Create a focused summary of this job description.
                Include only the most important requirements and responsibilities.
                Keep your response under 500 words.
                
                Job Details:
                {text}
                
                Summary:
                """,
                input_variables=["text"]
            )
            
            # Check final token count
            final_tokens = self.token_counter.estimate_tokens(combined_text)
            if final_tokens > self.rate_limiter.max_tokens_per_request:
                logger.warning("Final text too large, using truncated version")
                # Take first part of processed sections that fits within token limit
                truncated_text = ""
                for section in processed_sections:
                    if self.token_counter.estimate_tokens(truncated_text + section) < self.rate_limiter.max_tokens_per_request:
                        truncated_text += section + "\n\n"
                    else:
                        break
                combined_text = truncated_text
            
            chain = final_prompt | self.llm_cheap | StrOutputParser()
            final_summary = chain.invoke({"text": combined_text})
            self.job_description = final_summary
            
        except Exception as e:
            logger.error(f"Error in final summarization: {str(e)}")
            # Fallback: Use the combined sections directly
            self.job_description = combined_text[:4000]  # Truncate to safe size
            
        return self.job_description

    def _process_with_embeddings(self, chunks):
        """Process job description using embeddings for better semantic understanding."""
        logger.info(f"Processing job description using embeddings (chunks: {len(chunks)})")
        
        try:
            # Create vector store from chunks
            vectorstore = FAISS.from_documents(documents=chunks, embedding=self.llm_embeddings)
            
            # Define a focused prompt for semantic analysis
            prompt = PromptTemplate(
                template="""
                Analyze this section of the job description and extract the key requirements.
                Focus on technical skills, experience requirements, and core responsibilities.
                Keep your response under 200 words.
                
                Section: {context}
                Key Requirements:
                """,
                input_variables=["context"]
            )
            
            def format_docs(docs):
                # Ensure we don't exceed token limits when formatting documents
                formatted = "\n\n".join(doc.page_content for doc in docs)
                if self.token_counter.estimate_tokens(formatted) > self.rate_limiter.max_tokens_per_request:
                    # Take only as many documents as we can fit
                    safe_docs = []
                    current_text = ""
                    for doc in docs:
                        if self.token_counter.estimate_tokens(current_text + doc.page_content) < self.rate_limiter.max_tokens_per_request:
                            current_text += doc.page_content + "\n\n"
                            safe_docs.append(doc)
                        else:
                            break
                    formatted = "\n\n".join(doc.page_content for doc in safe_docs)
                return formatted
            
            # Set up the processing chain
            context_formatter = vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs
            chain_job_description = prompt | self.llm_cheap | StrOutputParser()
            
            try:
                # Process with semantic search
                context = context_formatter.invoke("key requirements and responsibilities")
                
                # Check token count before processing
                if self.token_counter.estimate_tokens(context) > self.rate_limiter.max_tokens_per_request:
                    logger.warning("Context too large, falling back to direct processing")
                    return self._process_without_embeddings(chunks)
                
                result = chain_job_description.invoke({"context": context})
                
                # Final summarization with strict token control
                final_prompt = PromptTemplate(
                    template="""
                    Create a clear and concise summary of this job description.
                    Focus on the most important requirements and responsibilities.
                    Keep your response under 500 words.
                    
                    Job Details:
                    {text}
                    
                    Summary:
                    """,
                    input_variables=["text"]
                )
                
                chain_summarize = final_prompt | self.llm_cheap | StrOutputParser()
                final_summary = chain_summarize.invoke({"text": result})
                self.job_description = final_summary
                
            except Exception as e:
                if "413" in str(e) or "too large" in str(e):
                    logger.warning("Token limit exceeded in embedding processing, falling back to direct processing")
                    return self._process_without_embeddings(chunks)
                raise
                
        except Exception as e:
            logger.error(f"Error in embedding processing: {str(e)}")
            return self._process_without_embeddings(chunks)
        
        return self.job_description

class TokenCounter:
    def __init__(self):
        self.encoding = encoding_for_model("gpt-3.5-turbo")
    
    def estimate_tokens(self, text):
        """Estimate the number of tokens in the given text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error estimating tokens: {str(e)}")
            # Fallback: use rough character-based estimation
            return len(text) // 4  # Rough estimate of 4 characters per token
