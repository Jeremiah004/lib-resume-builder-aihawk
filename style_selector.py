from pathlib import Path
from typing import Dict, List, Tuple
import os

class StyleSelector:
    def __init__(self, styles_directory: Path):
        self.styles_directory = styles_directory
        self._styles_cache = None

    def get_available_styles(self) -> Dict[str, Tuple[str, str]]:
        """Get all available styles with their file names and author links."""
        if self._styles_cache is not None:
            return self._styles_cache

        styles_to_files = {}
        try:
            files = os.listdir(self.styles_directory)
            for f in files:
                file_path = self.styles_directory / Path(f)
                if file_path.is_file() and f.endswith('.css'):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        first_line = file.readline().strip()
                        if first_line.startswith("/*") and first_line.endswith("*/"):
                            content = first_line[2:-2].strip()
                            if '$' in content:
                                style_name, author_link = content.split('$', 1)
                                style_name = style_name.strip()
                                author_link = author_link.strip()
                                styles_to_files[style_name] = (f, author_link)
        except FileNotFoundError:
            print(f"Directory {self.styles_directory} not found.")
        except PermissionError:
            print(f"Permission denied to access {self.styles_directory}.")
        
        self._styles_cache = styles_to_files
        return styles_to_files

    def get_style_list(self) -> List[str]:
        """Get a list of available style names."""
        styles = self.get_available_styles()
        return list(styles.keys())

    def get_style_path(self, style_name: str) -> Path:
        """Get the path to a specific style file."""
        styles = self.get_available_styles()
        if style_name not in styles:
            raise ValueError(f"Style '{style_name}' not found")
        file_name, _ = styles[style_name]
        return self.styles_directory / file_name

    def get_style_info(self, style_name: str) -> Tuple[str, str]:
        """Get the file name and author link for a specific style."""
        styles = self.get_available_styles()
        if style_name not in styles:
            raise ValueError(f"Style '{style_name}' not found")
        return styles[style_name] 