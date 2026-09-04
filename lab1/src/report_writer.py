import csv
import json
from pathlib import Path


def write_repository_stats(
    inventory: dict,
    git_stats: dict,
    output_path: Path,
) -> None:

    report = {
        "repository_inventory": inventory,
        "git_history": git_stats,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


def write_git_history_summary(
    git_stats: dict,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        (
            "total_commits",
            git_stats["total_commits"],
        ),
        (
            "unique_contributors",
            git_stats["unique_contributors"],
        ),
        (
            "average_files_changed_per_commit",
            git_stats["average_files_changed_per_commit"],
        ),
        (
            "average_files_changed_per_month",
            git_stats["average_files_changed_per_month"],
        ),
        (
            "total_additions",
            git_stats["total_additions"],
        ),
        (
            "total_deletions",
            git_stats["total_deletions"],
        ),
        (
            "average_additions_per_commit",
            git_stats["average_additions_per_commit"],
        ),
        (
            "average_deletions_per_commit",
            git_stats["average_deletions_per_commit"],
        ),
        (
            "average_additions_per_month",
            git_stats["average_additions_per_month"],
        ),
        (
            "average_deletions_per_month",
            git_stats["average_deletions_per_month"],
        ),
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["metric", "value"]
        )

        writer.writerows(rows)