"""
Resume Builder AI Hawk Library
A powerful library for generating professional resumes using AI.
"""

__version__ = '1.0.0'

from .resume import Resume
from .resume_generator import ResumeGenerator
from .style_manager import StyleManager
from .manager_facade import FacadeManager
from .gpt_resume import LLMResumer


__all__ = [
    'Resume',
    'ResumeGenerator',
    'StyleManager',
    'FacadeManager',
    'LLMResumer',
    'LoggerChatModel',
    'LLMLogger'
]