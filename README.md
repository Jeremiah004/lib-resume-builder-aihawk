# Resume Builder AIHawk Library

A Python library for generating and managing resumes using AI-powered tools.

## Features

- Resume generation from PDF or text input
- AI-powered resume parsing and extraction
- Customizable resume styles and templates
- PDF generation with support for various formats
- Integration with Groq API for AI processing

## Installation

```bash
pip install lib-resume-builder-aihawk
```

## Usage

```python
from lib_resume_builder_AIHawk import FacadeManager, StyleManager, ResumeGenerator

# Initialize components
style_manager = StyleManager()
resume_generator = ResumeGenerator()

# Create FacadeManager instance
manager = FacadeManager(
    groq_api_key="your_groq_api_key",
    style_manager=style_manager,
    resume_generator=resume_generator,
    resume_object=resume_object,
    log_path="path/to/log/directory"
)

# Generate resume
pdf_data = manager.pdf_base64()
```

## Requirements

- Python 3.8+
- PyPDF2
- WeasyPrint
- PyYAML
- python-dotenv

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 