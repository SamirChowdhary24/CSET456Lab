import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from git import Repo
from pydriller import Repository


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

FILE_DATASET_PATH = OUTPUT_DIR / "file_dataset.csv"
STATS_PATH = OUTPUT_DIR / "repository_stats.json"

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


def should_skip(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def get_language(path: Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "Other")


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            return sum(1 for _ in file)
    except OSError:
        return 0


def build_file_dataset():
    rows = []
    directories = set()
    file_type_distribution = Counter()
    languages = Counter()

    for path in REPO_ROOT.rglob("*"):
        relative_path = path.relative_to(REPO_ROOT)

        if should_skip(relative_path):
            continue

        if path.is_dir():
            directories.add(str(relative_path))
            continue

        if path.is_file():
            extension = path.suffix.lower() if path.suffix else "no_extension"
            language = get_language(path)
            loc = count_lines(path)
            size_bytes = path.stat().st_size

            rows.append(
                {
                    "file_path": str(relative_path).replace("\\", "/"),
                    "language": language,
                    "extension": extension,
                    "loc": loc,
                    "size_bytes": size_bytes,
                }
            )

            file_type_distribution[extension] += 1
            languages[language] += 1

    df = pd.DataFrame(rows)
    df.to_csv(FILE_DATASET_PATH, index=False)

    source_files = df[df["language"] != "Other"]

    inventory = {
        "repository_name": REPO_ROOT.name,
        "number_of_files": int(len(df)),
        "number_of_source_code_files": int(len(source_files)),
        "programming_languages": sorted(source_files["language"].unique().tolist()),
        "number_of_directories": len(directories),
        "total_loc": int(df["loc"].sum()),
        "loc_per_source_file": round(float(source_files["loc"].mean()), 2)
        if len(source_files) > 0
        else 0,
        "largest_source_files": source_files.sort_values("loc", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "file_type_distribution": dict(file_type_distribution),
    }

    return inventory


def mine_git_history():
    repo = Repo(REPO_ROOT)

    commits = list(repo.iter_commits("HEAD"))
    contributors = Counter()
    changed_files = Counter()
    commits_per_month = Counter()
    files_changed_per_month = defaultdict(int)

    total_additions = 0
    total_deletions = 0
    total_files_changed = 0

    for commit in Repository(str(REPO_ROOT)).traverse_commits():
        author = f"{commit.author.name} <{commit.author.email}>"
        month = commit.author_date.strftime("%Y-%m")

        contributors[author] += 1
        commits_per_month[month] += 1

        files_in_commit = len(commit.modified_files)
        total_files_changed += files_in_commit
        files_changed_per_month[month] += files_in_commit

        for modified_file in commit.modified_files:
            file_path = modified_file.new_path or modified_file.old_path

            if file_path:
                changed_files[file_path] += 1

            total_additions += modified_file.added_lines
            total_deletions += modified_file.deleted_lines

    number_of_commits = len(commits)
    number_of_months = len(commits_per_month)

    history_stats = {
        "number_of_commits": number_of_commits,
        "number_of_contributors": len(contributors),
        "most_active_contributor": contributors.most_common(1)[0]
        if contributors
        else None,
        "files_changed_most_frequently": changed_files.most_common(10),
        "commits_per_month": dict(sorted(commits_per_month.items())),
        "average_files_changed_per_month": round(
            total_files_changed / number_of_months, 2
        )
        if number_of_months > 0
        else 0,
        "average_additions_per_commit": round(total_additions / number_of_commits, 2)
        if number_of_commits > 0
        else 0,
        "average_deletions_per_commit": round(total_deletions / number_of_commits, 2)
        if number_of_commits > 0
        else 0,
    }

    return history_stats


def main():
    print("Building file-level dataset...")
    inventory = build_file_dataset()

    print("Mining Git history. This may take several minutes...")
    history_stats = mine_git_history()

    all_stats = {
        "repository_inventory": inventory,
        "git_history": history_stats,
    }

    with STATS_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_stats, file, indent=4)

    print("Done.")
    print(f"CSV created: {FILE_DATASET_PATH}")
    print(f"JSON created: {STATS_PATH}")


if __name__ == "__main__":
    main()