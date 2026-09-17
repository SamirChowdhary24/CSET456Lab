from pathlib import Path
import pandas as pd


PROCESSED_DIR = Path("lab2/data/processed")


def create_combined_dataset():
    source_path = PROCESSED_DIR / "clean_source_code_dataset.csv"
    commit_path = PROCESSED_DIR / "clean_commit_history_dataset.csv"

    source_df = pd.read_csv(source_path)
    commit_df = pd.read_csv(commit_path)

    # Summarize source-code information repository-wise
    source_summary = source_df.groupby("repository_name", as_index=False).agg(
        source_file_count=("file_path", "count"),
        total_lines_of_code=("line_count", "sum"),
        total_source_size_bytes=("file_size_bytes", "sum"),
        average_file_size_bytes=("file_size_bytes", "mean"),
        average_lines_per_file=("line_count", "mean"),
        source_extensions=(
            "file_extension",
            lambda values: ", ".join(sorted(values.dropna().unique()))
        )
    )

    # Summarize commit-history information repository-wise
    commit_summary = commit_df.groupby("repository_name", as_index=False).agg(
        commit_count=("commit_hash", "count"),
        total_insertions=("insertions", "sum"),
        total_deletions=("deletions", "sum"),
        total_changes=("total_changes", "sum"),
        average_files_changed_per_commit=("files_changed", "mean"),
        average_commit_changes=("total_changes", "mean"),
        unique_authors=("author_name", "nunique"),
        first_commit_date=("commit_date", "min"),
        last_commit_date=("commit_date", "max")
    )

    # Meaningful merge using repository_name
    combined_df = source_summary.merge(
        commit_summary,
        on="repository_name",
        how="inner"
    )

    # Round numerical averages
    combined_df["average_file_size_bytes"] = (
        combined_df["average_file_size_bytes"].round(2)
    )

    combined_df["average_lines_per_file"] = (
        combined_df["average_lines_per_file"].round(2)
    )

    combined_df["average_files_changed_per_commit"] = (
        combined_df["average_files_changed_per_commit"].round(2)
    )

    combined_df["average_commit_changes"] = (
        combined_df["average_commit_changes"].round(2)
    )

    combined_df = combined_df.sort_values("repository_name")

    output_path = PROCESSED_DIR / "combined_repository_dataset.csv"
    combined_df.to_csv(output_path, index=False)

    print("Combined dataset created successfully.")
    print(f"Output file: {output_path}")
    print(f"Shape: {combined_df.shape}")
    print("\nCombined dataset preview:")
    print(combined_df.to_string(index=False))


if __name__ == "__main__":
    create_combined_dataset()