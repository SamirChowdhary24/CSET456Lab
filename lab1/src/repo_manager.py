# creates the folder C:\DevOpsLab1\target_repositories
# clones FastAPI there if it is not already cloned
# avoids cloning again if the folder already exists

from git import Repo

from config import REPOSITORY_URL, TARGET_REPOSITORIES_DIR, TARGET_REPOSITORY_PATH


def clone_repository_if_needed() -> None:
    TARGET_REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_REPOSITORY_PATH.exists():
        print(f"Repository already exists at: {TARGET_REPOSITORY_PATH}")
        return

    print(f"Cloning repository from: {REPOSITORY_URL}")
    Repo.clone_from(REPOSITORY_URL, TARGET_REPOSITORY_PATH)
    print(f"Repository cloned to: {TARGET_REPOSITORY_PATH}")