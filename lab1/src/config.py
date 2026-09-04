from pathlib import Path


REPOSITORY_URL = "https://github.com/fastapi/fastapi.git"
REPOSITORY_NAME = "fastapi"

LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parents[0]
DEVOPS_ROOT = PROJECT_ROOT.parent

TARGET_REPOSITORIES_DIR = DEVOPS_ROOT / "target_repositories"
TARGET_REPOSITORY_PATH = TARGET_REPOSITORIES_DIR / REPOSITORY_NAME

DATA_DIR = LAB_ROOT / "data"
OUTPUT_DIR = LAB_ROOT / "output"

FILE_DATASET_PATH = DATA_DIR / "file_dataset.csv"
REPOSITORY_STATS_PATH = OUTPUT_DIR / "repository_stats.json"