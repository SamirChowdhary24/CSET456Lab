from pathlib import Path


LAB2_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LAB2_DIR.parent

REPOSITORIES_DIR = Path(r"C:\DevOpsLab1\target_repositories")

REPOSITORIES = {
    "flask": "https://github.com/pallets/flask",
    "requests": "https://github.com/psf/requests",
    "pytest": "https://github.com/pytest-dev/pytest",
    "fastapi": "https://github.com/fastapi/fastapi",
    "scikit-learn": "https://github.com/scikit-learn/scikit-learn",
}

DATA_DIR = LAB2_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = LAB2_DIR / "output"

SOURCE_RAW_FILE = RAW_DATA_DIR / "source_code_raw.csv"
COMMIT_RAW_FILE = RAW_DATA_DIR / "commit_history_raw.csv"

SOURCE_DATASET_FILE = PROCESSED_DATA_DIR / "source_code_dataset.csv"
COMMIT_DATASET_FILE = PROCESSED_DATA_DIR / "commit_history_dataset.csv"
COMBINED_DATASET_FILE = PROCESSED_DATA_DIR / "combined_dataset.csv"

SOURCE_STATS_FILE = OUTPUT_DIR / "source_code_statistics.json"
COMMIT_STATS_FILE = OUTPUT_DIR / "commit_history_statistics.json"
COMBINED_STATS_FILE = OUTPUT_DIR / "combined_dataset_statistics.json"

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "dist",
    "build",
    "site-packages",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".pyw",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".swift",
    ".kt",
    ".kts",
}