from config import OUTPUT_DIR, REPOSITORY_STATS_PATH
from file_analyzer import analyze_files
from git_history_miner import mine_git_history
from repo_manager import clone_repository_if_needed
from report_writer import (
    write_git_history_summary,
    write_repository_stats,
)


def main() -> None:
    print("=== FastAPI Repository Analysis ===")

    print("\n[1/4] Checking repository...")
    clone_repository_if_needed()

    print("\n[2/4] Analyzing files...")
    dataframe, inventory = analyze_files()
    print(f"Files analyzed: {len(dataframe)}")

    print("\n[3/4] Mining Git history...")
    git_stats = mine_git_history()
    print(f"Total commits: {git_stats['total_commits']}")
    print(f"Unique contributors: {git_stats['unique_contributors']}")

    print("\n[4/4] Writing reports...")

    write_repository_stats(
        inventory,
        git_stats,
        REPOSITORY_STATS_PATH,
    )

    write_git_history_summary(
        git_stats,
        OUTPUT_DIR / "git_history_summary.csv",
    )

    print("\n=== Analysis Complete ===")
    print(f"File dataset: {inventory['number_of_files']} files")
    print(f"Repository stats: {REPOSITORY_STATS_PATH}")
    print(
        "Git history summary: "
        f"{OUTPUT_DIR / 'git_history_summary.csv'}"
    )


if __name__ == "__main__":
    main()