from pathlib import Path

from git import Repo

from config import REPOSITORIES, REPOSITORIES_DIR


def ensure_repository(repo_name: str, repo_url: str) -> Path:
    """
    Ensure that a repository exists locally.

    If the repository already exists, reuse it.
    Otherwise, clone it from GitHub.
    """
    repo_path = REPOSITORIES_DIR / repo_name

    if repo_path.exists() and (repo_path / ".git").exists():
        print(f"[REUSE] Repository already exists: {repo_name}")
        return repo_path

    print(f"[CLONE] Cloning repository: {repo_name}")
    Repo.clone_from(repo_url, repo_path)

    return repo_path


def prepare_repositories() -> dict[str, Path]:
    """
    Ensure all configured repositories are available locally.
    """
    REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)

    local_repositories = {}

    for repo_name, repo_url in REPOSITORIES.items():
        local_repositories[repo_name] = ensure_repository(
            repo_name,
            repo_url,
        )

    return local_repositories