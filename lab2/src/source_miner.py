from pathlib import Path
import pandas as pd

from config import REPOSITORIES_DIR


# File extensions considered relevant for source-code mining
SOURCE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
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
    ".scala",
    ".sh",
    ".sql",
}


# Directories that should not be mined
IGNORED_DIRECTORIES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}


def count_lines(text: str) -> int:
    """Return the number of lines in a text file."""
    if not text:
        return 0

    return len(text.splitlines())


def mine_repository(repo_name: str, repo_path: Path) -> list[dict]:
    """
    Mine source-code files from one repository.
    """

    records = []

    for file_path in repo_path.rglob("*"):

        if not file_path.is_file():
            continue

        # Ignore files inside irrelevant directories
        if any(part in IGNORED_DIRECTORIES for part in file_path.parts):
            continue

        # Only process source-code extensions
        if file_path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue

        try:
            source_code = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            relative_path = file_path.relative_to(repo_path)

            records.append(
                {
                    "repository_name": repo_name,
                    "file_path": str(relative_path),
                    "file_extension": file_path.suffix.lower(),
                    "file_size_bytes": file_path.stat().st_size,
                    "line_count": count_lines(source_code),
                    "source_code": source_code,
                }
            )

        except (OSError, UnicodeError) as error:
            print(f"[SKIP] Could not read {file_path}: {error}")

    return records


def mine_all_repositories() -> pd.DataFrame:
    """
    Mine source-code files from all repositories.
    """

    all_records = []

    for repo_path in sorted(REPOSITORIES_DIR.iterdir()):

        if not repo_path.is_dir():
            continue

        repo_name = repo_path.name

        print(f"[MINE] Source code from: {repo_name}")

        repository_records = mine_repository(
            repo_name,
            repo_path
        )

        all_records.extend(repository_records)

        print(
            f"[DONE] {repo_name}: "
            f"{len(repository_records)} source files"
        )

    return pd.DataFrame(all_records)


def save_source_dataset() -> Path:
    """
    Mine all repositories and save the source-code dataset.
    """

    output_directory = Path("lab2/data/raw")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "source_code_dataset.csv"

    dataframe = mine_all_repositories()

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"[SAVED] Dataset written to: {output_path}")
    print(f"[INFO] Dataset shape: {dataframe.shape}")

    return output_path


if __name__ == "__main__":
    save_source_dataset()