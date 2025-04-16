from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import yaml
from pydantic import BaseModel, EmailStr, HttpUrl, Field, validator, field_validator



class PersonalInfo(BaseModel):
    """Personal information model."""
    name: Optional[str] = None
    surname: Optional[str] = None
    date_of_birth: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    phone_prefix: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    github: Optional[str] = None
    linkedin: Optional[str] = None

    @field_validator('github', 'linkedin')
    @classmethod
    def validate_urls(cls, v: Optional[str]) -> str:
        """Validate URL fields."""
        if not v or v.lower() in ['not available', 'not specified', 'https://not specified']:
            return "https://placeholder.com"
        if not v.startswith(('http://', 'https://')):
            return f"https://{v}"
        return v


class EducationDetail(BaseModel):
    """Education detail model."""
    education_level: Optional[str] = None
    institution: Optional[str] = None
    field_of_study: Optional[str] = None
    final_evaluation_grade: Optional[str] = None
    start_date: Optional[str] = None
    year_of_completion: Optional[int] = None
    exam: Optional[Dict[str, str]] = None

    @field_validator('exam')
    @classmethod
    def validate_exam(cls, v: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Validate exam field."""
        return v if v is not None else {}


class ExperienceDetail(BaseModel):
    """Experience detail model."""
    role: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    skills_acquired: Optional[List[str]] = None

    @field_validator('skills_acquired')
    @classmethod
    def validate_skills_acquired(cls, v: Optional[List[str]]) -> List[str]:
        """Validate skills_acquired field."""
        return v if v is not None else []

    @field_validator('role', 'company', 'duration', 'location', 'description')
    @classmethod
    def validate_required_fields(cls, v: Optional[str]) -> str:
        """Validate required fields."""
        return v if v is not None else "Not specified"


class Project(BaseModel):
    """Project model."""
    name: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None

    @field_validator('link')
    @classmethod
    def validate_link(cls, v: Optional[str]) -> str:
        """Validate link field."""
        if not v or v.lower() in ['not available', 'not specified']:
            return "https://github.com/placeholder/project"
        if not v.startswith(('http://', 'https://')):
            return f"https://github.com/{v}"
        return v


class Achievement(BaseModel):
    """Achievement model."""
    name: Optional[str] = None
    description: Optional[str] = None


class Certification(BaseModel):
    """Certification model."""
    name: Optional[str] = None
    description: Optional[str] = None


class Language(BaseModel):
    """Language model."""
    language: Optional[str] = None
    proficiency: Optional[str] = None


class Availability(BaseModel):
    notice_period: Optional[str]


class SalaryExpectations(BaseModel):
    salary_range_usd: Optional[str]


class SelfIdentification(BaseModel):
    gender: Optional[str]
    pronouns: Optional[str]
    veteran: Optional[str]
    disability: Optional[str]
    ethnicity: Optional[str]


class LegalAuthorization(BaseModel):
    eu_work_authorization: Optional[str]
    us_work_authorization: Optional[str]
    requires_us_visa: Optional[str]
    requires_us_sponsorship: Optional[str]
    requires_eu_visa: Optional[str]
    legally_allowed_to_work_in_eu: Optional[str]
    legally_allowed_to_work_in_us: Optional[str]
    requires_eu_sponsorship: Optional[str]


class Resume(BaseModel):
    """Resume model."""
    personal_info: Optional[PersonalInfo] = None
    education_details: List[EducationDetail] = Field(default_factory=list)
    experience_details: List[ExperienceDetail] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    achievements: List[Achievement] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)

    @field_validator('personal_info')
    @classmethod
    def validate_personal_info(cls, v: Optional[PersonalInfo]) -> PersonalInfo:
        """Validate personal_info field."""
        return v if v is not None else PersonalInfo()

    @field_validator('education_details', 'experience_details', 'projects', 'achievements', 'certifications', 'languages', 'interests')
    @classmethod
    def validate_lists(cls, v: Optional[List[Any]]) -> List[Any]:
        """Validate list fields."""
        return v if v is not None else []

    @staticmethod
    def normalize_exam_format(exam):
        """Normalize exam format to ensure it's a dictionary."""
        if exam is None:
            return {}
        if isinstance(exam, list):
            # Convert list of exam entries to a dictionary
            exam_dict = {}
            for entry in exam:
                if isinstance(entry, dict):
                    exam_dict.update(entry)
            return exam_dict
        if isinstance(exam, dict):
            return exam
        return {}

    def __init__(self, yaml_str: str = None, **data):
        """Initialize Resume model."""
        try:
            # Parse the YAML string if provided
            if yaml_str:
                data = yaml.safe_load(yaml_str)

            # Clean up exam data in education details
            if 'education_details' in data:
                for edu in data['education_details']:
                    if 'exam' in edu:
                        edu['exam'] = self.normalize_exam_format(edu['exam'])

            # Clean up skills_acquired in experience details
            if 'experience_details' in data:
                for exp in data['experience_details']:
                    if 'skills_acquired' in exp:
                        exp['skills_acquired'] = [skill if skill is not None else "" for skill in exp.get('skills_acquired', [])]

            # Create an instance of Resume from the parsed data
            super().__init__(**data)

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error while parsing YAML: {e}")


    def _process_personal_information(self, data: Dict[str, Any]) -> PersonalInfo:
        try:
            return PersonalInfo(**data)
        except TypeError as e:
            raise TypeError(f"Invalid data for PersonalInfo: {e}") from e
        except AttributeError as e:
            raise AttributeError(f"AttributeError in PersonalInfo: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error in PersonalInfo processing: {e}") from e

    def _process_education_details(self, data: List[Dict[str, Any]]) -> List[EducationDetail]:
        education_list = []
        for edu in data:
            try:
                exams = [Exam(name=k, grade=v) for k, v in edu.get('exam', {}).items()]
                education = EducationDetail(
                    education_level=edu.get('education_level'),
                    institution=edu.get('institution'),
                    field_of_study=edu.get('field_of_study'),
                    final_evaluation_grade=edu.get('final_evaluation_grade'),
                    start_date=edu.get('start_date'),
                    year_of_completion=edu.get('year_of_completion'),
                    exam=exams
                )
                education_list.append(education)
            except KeyError as e:
                raise KeyError(f"Missing field in education details: {e}") from e
            except TypeError as e:
                raise TypeError(f"Invalid data for Education: {e}") from e
            except AttributeError as e:
                raise AttributeError(f"AttributeError in Education: {e}") from e
            except Exception as e:
                raise Exception(f"Unexpected error in Education processing: {e}") from e
        return education_list

    def _process_experience_details(self, data: List[Dict[str, Any]]) -> List[ExperienceDetail]:
        experience_list = []
        for exp in data:
            try:
                key_responsibilities = [
                    Responsibility(description=list(resp.values())[0])
                    for resp in exp.get('key_responsibilities', [])
                ]
                skills_acquired = [str(skill) for skill in exp.get('skills_acquired', [])]
                experience = ExperienceDetail(
                    position=exp['position'],
                    company=exp['company'],
                    employment_period=exp['employment_period'],
                    location=exp['location'],
                    industry=exp['industry'],
                    key_responsibilities=key_responsibilities,
                    skills_acquired=skills_acquired
                )
                experience_list.append(experience)
            except KeyError as e:
                raise KeyError(f"Missing field in experience details: {e}") from e
            except TypeError as e:
                raise TypeError(f"Invalid data for Experience: {e}") from e
            except AttributeError as e:
                raise AttributeError(f"AttributeError in Experience: {e}") from e
            except Exception as e:
                raise Exception(f"Unexpected error in Experience processing: {e}") from e
        return experience_list


@dataclass
class Exam:
    name: str
    grade: str

@dataclass
class Responsibility:
    description: str
