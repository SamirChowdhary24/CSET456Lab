from collections import Counter
from pathlib import Path

import pandas as pd

from config import FILE_DATASET_PATH, TARGET_REPOSITORY_PATH


IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}


LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".md": "Markdown",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".txt": "Text",
    ".html": "HTML",
    ".css": "CSS",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".sh": "Shell",
    ".bat": "Batch",
    ".ps1": "PowerShell",
    ".rst": "reStructuredText",
}


# These are the languages we consider actual source code
# for the purpose of this lab.
SOURCE_CODE_LANGUAGES = {
    "Python",
    "JavaScript",
    "TypeScript",
    "HTML",
    "CSS",
    "Shell",
    "Batch",
    "PowerShell",
}


def should_skip(path: Path) -> bool:
    """Return True if the path is inside a directory we want to ignore."""
    return any(part in IGNORE_DIRS for part in path.parts)


def get_language(path: Path) -> str:
    """Return the language associated with a file extension."""
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "Other")


def count_lines(path: Path) -> int:
    """Count the number of lines in a text file."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            return sum(1 for _ in file)
    except (OSError, UnicodeDecodeError):
        return 0


def count_python_methods(path: Path) -> int:
    """Count Python functions and methods using the AST module."""
    if path.suffix.lower() != ".py":
        return 0

    try:
        import ast

        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)

        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(tree)
        )

    except (OSError, SyntaxError, UnicodeDecodeError):
        return 0


def analyze_files():
    """Analyze repository files and generate the file-level dataset."""

    records = []

    for path in TARGET_REPOSITORY_PATH.rglob("*"):

        if not path.is_file():
            continue

        if should_skip(path):
            continue

        language = get_language(path)

        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0

        records.append(
            {
                "file_path": str(path.relative_to(TARGET_REPOSITORY_PATH)),
                "language": language,
                "extension": path.suffix.lower(),
                "loc": count_lines(path),
                "method_count": count_python_methods(path),
                "size_bytes": size_bytes,
            }
        )

    file_dataset = pd.DataFrame(records)

    if file_dataset.empty:
        raise RuntimeError("No files were found in the repository.")

    # ---------------------------------------------------------
    # Source-code files
    # ---------------------------------------------------------
    source_files = file_dataset[
        file_dataset["language"].isin(SOURCE_CODE_LANGUAGES)
    ]

    # ---------------------------------------------------------
    # Language distribution
    # ---------------------------------------------------------
    language_distribution = Counter(file_dataset["language"])

    # ---------------------------------------------------------
    # File-type distribution
    # ---------------------------------------------------------
    extension_distribution = Counter(
        file_dataset["extension"].replace("", "[no extension]")
    )

    # ---------------------------------------------------------
    # Largest source files
    # ---------------------------------------------------------
    largest_source_files = (
        source_files.nlargest(10, "loc")[
            ["file_path", "language", "loc", "size_bytes"]
        ]
        .to_dict(orient="records")
    )

    # ---------------------------------------------------------
    # Directory count
    # ---------------------------------------------------------
    directories = set()

    for path in TARGET_REPOSITORY_PATH.rglob("*"):
        if path.is_dir() and not should_skip(path):
            directories.add(path)

    # ---------------------------------------------------------
    # Inventory
    # ---------------------------------------------------------
    inventory = {
        "repository_name": TARGET_REPOSITORY_PATH.name,

        "number_of_files": int(len(file_dataset)),

        "number_of_source_code_files": int(len(source_files)),

        "number_of_files_by_programming_language": {
            language: int(count)
            for language, count in language_distribution.items()
        },

        "programming_languages": sorted(
            source_files["language"].unique().tolist()
        ),

        "number_of_directories": len(directories),

        "total_loc": int(file_dataset["loc"].sum()),

        "average_loc_per_file": round(
            float(file_dataset["loc"].mean()),
            2,
        ),

        "average_loc_per_source_file": round(
            float(source_files["loc"].mean()),
            2
        )
        if not source_files.empty
        else 0,

        "total_python_methods": int(
            file_dataset["method_count"].sum()
        ),

        "largest_source_files": largest_source_files,

        "file_type_distribution": {
            extension: int(count)
            for extension, count in extension_distribution.items()
        },
    }

    # ---------------------------------------------------------
    # Save CSV dataset
    # ---------------------------------------------------------
    FILE_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_dataset.to_csv(
        FILE_DATASET_PATH,
        index=False,
    )

    return file_dataset, inventory