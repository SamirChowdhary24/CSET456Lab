from collections import Counter
from git import Repo

from config import TARGET_REPOSITORY_PATH


def mine_git_history() -> dict:
    repo = Repo(TARGET_REPOSITORY_PATH)

    total_commits = 0

    contributors = Counter()
    changed_files = Counter()
    commits_per_month = Counter()
    files_changed_per_month = Counter()
    additions_per_month = Counter()
    deletions_per_month = Counter()

    total_files_changed = 0
    total_additions = 0
    total_deletions = 0

    for commit in repo.iter_commits("--all"):
        total_commits += 1

        # Contributor information
        author_name = commit.author.name or "Unknown"
        contributors[author_name] += 1

        # Month of commit
        commit_month = commit.committed_datetime.strftime("%Y-%m")
        commits_per_month[commit_month] += 1

        try:
            commit_stats = commit.stats
            files_changed = commit_stats.files

            files_count = len(files_changed)

            total_files_changed += files_count
            files_changed_per_month[commit_month] += files_count

            # Track frequently changed files
            for file_path in files_changed:
                changed_files[file_path] += 1

            # Track additions and deletions
            additions = commit_stats.total["insertions"]
            deletions = commit_stats.total["deletions"]

            total_additions += additions
            total_deletions += deletions

            additions_per_month[commit_month] += additions
            deletions_per_month[commit_month] += deletions

        except Exception:
            continue

    # Number of months represented in the Git history
    number_of_months = len(commits_per_month)

    # Average files changed per commit
    average_files_changed_per_commit = (
        round(total_files_changed / total_commits, 2)
        if total_commits
        else 0
    )

    # Average files changed per month
    average_files_changed_per_month = (
        round(total_files_changed / number_of_months, 2)
        if number_of_months
        else 0
    )

    # Average additions and deletions per commit
    average_additions_per_commit = (
        round(total_additions / total_commits, 2)
        if total_commits
        else 0
    )

    average_deletions_per_commit = (
        round(total_deletions / total_commits, 2)
        if total_commits
        else 0
    )

    # Average additions and deletions per month
    average_additions_per_month = (
        round(total_additions / number_of_months, 2)
        if number_of_months
        else 0
    )

    average_deletions_per_month = (
        round(total_deletions / number_of_months, 2)
        if number_of_months
        else 0
    )

    return {
        "total_commits": total_commits,

        "unique_contributors": len(contributors),

        "most_active_contributors": [
            {
                "contributor": name,
                "commit_count": count,
            }
            for name, count in contributors.most_common(10)
        ],

        "most_frequently_changed_files": [
            {
                "file_path": path,
                "change_count": count,
            }
            for path, count in changed_files.most_common(10)
        ],

        "commits_per_month": dict(
            sorted(commits_per_month.items())
        ),

        "files_changed_per_month": dict(
            sorted(files_changed_per_month.items())
        ),

        "average_files_changed_per_commit":
            average_files_changed_per_commit,

        "average_files_changed_per_month":
            average_files_changed_per_month,

        "total_additions": total_additions,

        "total_deletions": total_deletions,

        "average_additions_per_commit":
            average_additions_per_commit,

        "average_deletions_per_commit":
            average_deletions_per_commit,

        "average_additions_per_month":
            average_additions_per_month,

        "average_deletions_per_month":
            average_deletions_per_month,
    }