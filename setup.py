from setuptools import setup, find_packages

setup(
    name="lib-resume-builder-aihawk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyPDF2>=3.0.0",
        "WeasyPrint>=60.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",
        "selenium>=4.0.0",
        "inquirer>=3.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for generating and managing resumes using AI-powered tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lib-resume-builder-aihawk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 