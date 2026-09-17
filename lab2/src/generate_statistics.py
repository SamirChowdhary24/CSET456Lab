from pathlib import Path
import pandas as pd


PROCESSED_DIR = Path("lab2/data/processed")
OUTPUT_DIR = Path("lab2/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_statistics():
    source_df = pd.read_csv(
        PROCESSED_DIR / "clean_source_code_dataset.csv"
    )

    commit_df = pd.read_csv(
        PROCESSED_DIR / "clean_commit_history_dataset.csv"
    )

    combined_df = pd.read_csv(
        PROCESSED_DIR / "combined_repository_dataset.csv"
    )

    # Statistic 1: Source-code dataset
    source_stats = {
        "dataset": "Clean Source Code Dataset",
        "total_records": len(source_df),
        "total_repositories": source_df["repository_name"].nunique(),
        "total_lines_of_code": int(source_df["line_count"].sum()),
        "average_lines_per_file": round(source_df["line_count"].mean(), 2),
        "largest_file_size_bytes": int(source_df["file_size_bytes"].max()),
    }

    # Statistic 2: Commit-history dataset
    commit_stats = {
        "dataset": "Clean Commit History Dataset",
        "total_records": len(commit_df),
        "total_repositories": commit_df["repository_name"].nunique(),
        "total_insertions": int(commit_df["insertions"].sum()),
        "total_deletions": int(commit_df["deletions"].sum()),
        "average_files_changed_per_commit": round(
            commit_df["files_changed"].mean(), 2
        ),
        "unique_authors": int(commit_df["author_name"].nunique()),
    }

    # Statistic 3: Combined dataset
    combined_stats = {
        "dataset": "Combined Repository Dataset",
        "total_records": len(combined_df),
        "total_repositories": combined_df["repository_name"].nunique(),
        "total_source_files": int(combined_df["source_file_count"].sum()),
        "total_commits": int(combined_df["commit_count"].sum()),
        "total_lines_of_code": int(combined_df["total_lines_of_code"].sum()),
        "total_code_changes": int(combined_df["total_changes"].sum()),
    }

    all_stats = [source_stats, commit_stats, combined_stats]

    stats_df = pd.DataFrame(all_stats)
    stats_path = OUTPUT_DIR / "dataset_statistics.csv"
    stats_df.to_csv(stats_path, index=False)

    # Human-readable report
    report_path = OUTPUT_DIR / "dataset_statistics_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        for stats in all_stats:
            file.write("=" * 60 + "\n")
            file.write(f"{stats['dataset']}\n")
            file.write("=" * 60 + "\n")

            for key, value in stats.items():
                if key != "dataset":
                    file.write(f"{key}: {value}\n")

            file.write("\n")

    print("Statistics generated successfully.")
    print(f"CSV output: {stats_path}")
    print(f"Text report: {report_path}")
    print("\nStatistics:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    generate_statistics()