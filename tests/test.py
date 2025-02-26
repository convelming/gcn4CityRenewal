from pathlib import Path

# 生成项目基础目录
def create_project_structure(base_path="data-modeling-project"):
    base = Path(base_path)

    # Define directories
    directories = [
        "data/raw", "data/processed", "data/external", "data/interim",
        "notebooks/exploratory", "notebooks/modeling", "notebooks/results",
        "src/data", "src/models", "src/visualization", "src/utils",
        "tests",
        "scripts",
        "reports/figures",
        "config"
    ]

    # Define files with initial content
    files = {
        "README.md": "# Data Modeling Project\n\nProject description and setup instructions.",
        "LICENSE": "MIT License",
        "requirements.txt": "# List project dependencies here",
        "environment.yml": "# Conda environment file",
        "pyproject.toml": "# Project configuration",
        ".gitignore": "*.pyc\n__pycache__/\ndata/raw/*\n.env\n",
        "Makefile": "# Define automation tasks",
        "setup.py": "from setuptools import setup, find_packages\n\nsetup(name='data_modeling', packages=find_packages())",
        "config/params.yaml": "learning_rate: 0.01\nbatch_size: 32",
        "config/database.json": "{\"host\": \"localhost\", \"port\": 5432}",
        "config/logging.yaml": "version: 1\ndisable_existing_loggers: False"
    }

    # Create directories
    for directory in directories:
        (base / directory).mkdir(parents=True, exist_ok=True)

    # Create files
    for file, content in files.items():
        file_path = base / file
        if not file_path.exists():
            file_path.write_text(content, encoding='utf-8')

    print(f"Project structure created at: {base_path}")


# Run the function
create_project_structure()
