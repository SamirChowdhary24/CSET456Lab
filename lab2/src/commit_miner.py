from pathlib import Path
import pandas as pd
from git import Repo

from config import REPOSITORIES_DIR


def mine_repository_commits(repo_name: str, repo_path: Path) -> list[dict]:
    """
    Extract commit-history information from one repository.
    """

    records = []
    repo = Repo(repo_path)

    print(f"[MINE] Commit history from: {repo_name}")

    for commit in repo.iter_commits():

        try:
            stats = commit.stats.total

            records.append(
                {
                    "repository_name": repo_name,
                    "commit_hash": commit.hexsha,
                    "author_name": commit.author.name,
                    "commit_date": commit.committed_datetime.isoformat(),
                    "commit_message": commit.message.strip(),
                    "files_changed": stats.get("files", 0),
                    "insertions": stats.get("insertions", 0),
                    "deletions": stats.get("deletions", 0),
                    "total_changes": (
                        stats.get("insertions", 0)
                        + stats.get("deletions", 0)
                    ),
                }
            )

        except Exception as error:
            print(
                f"[SKIP] Could not process commit "
                f"{commit.hexsha}: {error}"
            )

    print(f"[DONE] {repo_name}: {len(records)} commits")

    return records


def mine_all_commit_histories() -> pd.DataFrame:
    """
    Mine commit histories from all repositories.
    """

    all_records = []

    for repo_path in sorted(REPOSITORIES_DIR.iterdir()):

        if not repo_path.is_dir():
            continue

        repo_name = repo_path.name

        repository_records = mine_repository_commits(
            repo_name,
            repo_path
        )

        all_records.extend(repository_records)

    return pd.DataFrame(all_records)


def save_commit_dataset() -> Path:
    """
    Mine all commit histories and save the raw dataset.
    """

    output_directory = Path("lab2/data/raw")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "commit_history_dataset.csv"

    dataframe = mine_all_commit_histories()

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"[SAVED] Dataset written to: {output_path}")
    print(f"[INFO] Dataset shape: {dataframe.shape}")

    return output_path


if __name__ == "__main__":
    save_commit_dataset()