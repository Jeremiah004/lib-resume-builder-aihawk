import json
import os
import unittest
from unittest.mock import patch, MagicMock
import sys
import logging
import yaml
from pathlib import Path

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import the classes we need for testing
from lib_resume_builder_AIHawk import Resume, ResumeGenerator, StyleManager, FacadeManager
from lib_resume_builder_AIHawk.gpt_resume import LLMResumer, LoggerChatModel


class TestLLMResumer(unittest.TestCase):
    """Test class focused on the LLMResumer functionality"""

    def setUp(self):
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger()
        
        # Create a file handler for the debug output
        debug_file = logging.FileHandler('llm_resumer_debug.log')
        debug_file.setLevel(logging.DEBUG)
        self.logger.addHandler(debug_file)
        
        # Load test data
        with open('./yaml_example/secrets.yaml', 'r') as f:
            self.secrets = yaml.safe_load(f)
        
        with open('./yaml_example/plain_text_resume.yaml', 'r') as f:
            self.plain_text_data = yaml.safe_load(f)
            self.logger.debug(f"Resume fields: {list(self.plain_text_data.keys())}")
        
        # Convert to YAML string for Resume object
        self.plain_text_resume = yaml.dump(self.plain_text_data, default_flow_style=False)
        
        # Create a Resume object
        self.resume_object = Resume(self.plain_text_resume)
        
        # Extract API key
        self.llm_api_key = self.secrets['llm_api_key']
        
        # Create a mock for the strings class
        self.mock_strings = MagicMock()
        
        # Set sample prompts for testing
        self.mock_strings.prompt_header = "Generate a header section for this resume: {personal_information}"
        self.mock_strings.prompt_education = "Generate an education section for this resume: {education_details}"
        self.mock_strings.prompt_working_experience = "Generate a work experience section for this resume: {experience_details}"
        self.mock_strings.prompt_side_projects = "Generate a side projects section for this resume: {projects}"
        self.mock_strings.prompt_achievements = "Generate an achievements section for this resume: {achievements} based on job description: {job_description}"
        self.mock_strings.prompt_certifications = "Generate a certifications section for this resume: {certifications} based on job description: {job_description}"
        self.mock_strings.prompt_additional_skills = "Generate an additional skills section for this resume: Languages: {languages}, Interests: {interests}, Skills: {skills}"
        
        # Set up output path
        self.output_path = Path("data_folder/output")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare log file
        calls_log = self.output_path / "gemini_ai_calls.json"
        if not calls_log.exists():
            with open(calls_log, "w", encoding="utf-8") as f:
                json.dump([], f)
        
        with patch('lib_resume_builder_AIHawk.gpt_resume.LoggerChatModel') as mock_logger_chat_model:
            mock_logger_chat_model.return_value = self.create_mock_llm_instance()
            if mock_logger_chat_model.return_value is None:
                self.logger.error("Mock LoggerChatModel did not return an instance")
                raise ValueError("Mocked LoggerChatModel is returning None")
            self.llm_resumer = LLMResumer(self.llm_api_key, self.mock_strings)

            # Set the resume object
            self.llm_resumer.set_resume(self.resume_object, "Sample job description for testing")
    
    def create_mock_llm_instance(self):
        """Create a mock LLM instance that returns predictable results"""
        mock_llm = MagicMock()
        mock_llm.side_effect = lambda messages: f"Generated content for {messages.to_string()[:30]}..."
        return mock_llm

    def test_generate_header(self):
        """Test generation of the header section"""
        try:
            self.logger.debug("Testing generate_header method")
            result = self.llm_resumer.generate_header()
            self.logger.debug(f"Header result: {result}")
            self.assertTrue(result, "Header generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in header generation: {str(e)}")
            self.fail(f"Header generation raised exception: {str(e)}")

    def test_generate_education_section(self):
        """Test generation of the education section"""
        try:
            self.logger.debug("Testing generate_education_section method")
            result = self.llm_resumer.generate_education_section()
            self.logger.debug(f"Education result: {result}")
            self.assertTrue(result, "Education section generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in education section generation: {str(e)}")
            self.fail(f"Education section generation raised exception: {str(e)}")

    def test_generate_work_experience_section(self):
        """Test generation of the work experience section"""
        try:
            self.logger.debug("Testing generate_work_experience_section method")
            result = self.llm_resumer.generate_work_experience_section()
            self.logger.debug(f"Work experience result: {result}")
            self.assertTrue(result, "Work experience section generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in work experience section generation: {str(e)}")
            self.fail(f"Work experience section generation raised exception: {str(e)}")

    def test_generate_side_projects_section(self):
        """Test generation of the side projects section"""
        try:
            self.logger.debug("Testing generate_side_projects_section method")
            result = self.llm_resumer.generate_side_projects_section()
            self.logger.debug(f"Side projects result: {result}")
            self.assertTrue(result, "Side projects section generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in side projects section generation: {str(e)}")
            self.fail(f"Side projects section generation raised exception: {str(e)}")

    def test_generate_achievements_section(self):
        """Test generation of the achievements section"""
        try:
            self.logger.debug("Testing generate_achievements_section method")
            result = self.llm_resumer.generate_achievements_section()
            self.logger.debug(f"Achievements result: {result}")
            self.assertTrue(result, "Achievements section generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in achievements section generation: {str(e)}")
            self.fail(f"Achievements section generation raised exception: {str(e)}")

    def test_generate_certifications_section(self):
        """Test generation of the certifications section"""
        try:
            self.logger.debug("Testing generate_certifications_section method")
            result = self.llm_resumer.generate_certifications_section()
            self.logger.debug(f"Certifications result: {result}")
            self.assertTrue(result, "Certifications section generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in certifications section generation: {str(e)}")
            self.fail(f"Certifications section generation raised exception: {str(e)}")

    def test_generate_additional_skills_section(self):
        """Test generation of the additional skills section"""
        try:
            self.logger.debug("Testing generate_additional_skills_section method")
            result = self.llm_resumer.generate_additional_skills_section()
            self.logger.debug(f"Additional skills result: {result}")
            self.assertTrue(result, "Additional skills section generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in additional skills section generation: {str(e)}")
            self.fail(f"Additional skills section generation raised exception: {str(e)}")

    def test_generate_html_resume(self):
        """Test generation of the complete HTML resume"""
        try:
            self.logger.debug("Testing generate_html_resume method")
            
            # Patch each individual section generation method to isolate issues
            with patch.object(self.llm_resumer, 'generate_header', return_value="<header>Header Content</header>") as mock_header, \
                 patch.object(self.llm_resumer, 'generate_education_section', return_value="<section>Education Content</section>") as mock_education, \
                 patch.object(self.llm_resumer, 'generate_work_experience_section', return_value="<section>Work Experience Content</section>") as mock_work, \
                 patch.object(self.llm_resumer, 'generate_side_projects_section', return_value="<section>Side Projects Content</section>") as mock_projects, \
                 patch.object(self.llm_resumer, 'generate_achievements_section', return_value="<section>Achievements Content</section>") as mock_achievements, \
                 patch.object(self.llm_resumer, 'generate_certifications_section', return_value="<section>Certifications Content</section>") as mock_certifications, \
                 patch.object(self.llm_resumer, 'generate_additional_skills_section', return_value="<section>Additional Skills Content</section>") as mock_skills:
                
                result = self.llm_resumer.generate_html_resume()
                self.logger.debug(f"HTML resume result length: {len(result)}")
                self.logger.debug(f"HTML resume result (first 200 chars): {result[:200]}")
                
                # Check which methods were called
                self.logger.debug(f"Header method called: {mock_header.called}")
                self.logger.debug(f"Education method called: {mock_education.called}")
                self.logger.debug(f"Work experience method called: {mock_work.called}")
                self.logger.debug(f"Side projects method called: {mock_projects.called}")
                self.logger.debug(f"Achievements method called: {mock_achievements.called}")
                self.logger.debug(f"Certifications method called: {mock_certifications.called}")
                self.logger.debug(f"Additional skills method called: {mock_skills.called}")
                
                self.assertTrue(result, "HTML resume generation failed to produce output")
                self.assertIn("<body>", result, "HTML does not contain body tag")
                self.assertIn("<main>", result, "HTML does not contain main tag")
        except Exception as e:
            self.logger.error(f"Error in HTML resume generation: {str(e)}")
            self.fail(f"HTML resume generation raised exception: {str(e)}")

    def test_real_llm_integration(self):
        """Test with actual LLM integration for a single simple section"""
        # Only run this test if explicitly enabled since it uses the real API
        if os.environ.get('ENABLE_REAL_LLM_TEST') != 'true':
            self.logger.debug("Skipping real LLM integration test (not enabled)")
            self.skipTest("Real LLM integration test not enabled. Set ENABLE_REAL_LLM_TEST=true to enable.")
        
        try:
            # Create a real LLMResumer instance
            real_llm_resumer = LLMResumer(self.llm_api_key, self.mock_strings)
            real_llm_resumer.set_resume(self.resume_object, "Sample job description for testing")
            
            # Test a single method with real LLM
            self.logger.debug("Testing real LLM with header generation")
            result = real_llm_resumer.generate_side_projects_section()
            self.logger.debug(f"Real LLM header result: {result}")
            self.assertTrue(result, "Real LLM header generation failed to produce output")
        except Exception as e:
            self.logger.error(f"Error in real LLM test: {str(e)}")
            self.fail(f"Real LLM test raised exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()