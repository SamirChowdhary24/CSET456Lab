from pathlib import Path
import pandas as pd


RAW_DIR = Path("lab2/data/raw")
PROCESSED_DIR = Path("lab2/data/processed")


def clean_source_code_dataset() -> pd.DataFrame:
    """Clean the source-code dataset."""

    input_path = RAW_DIR / "source_code_dataset.csv"
    output_path = PROCESSED_DIR / "clean_source_code_dataset.csv"

    df = pd.read_csv(input_path)

    # Convert missing source-code values to empty strings.
    # Empty files should not be treated as invalid records.
    df["source_code"] = df["source_code"].fillna("")

    # Normalize text columns
    df["repository_name"] = df["repository_name"].astype(str).str.strip()
    df["file_path"] = df["file_path"].astype(str).str.strip()
    df["file_extension"] = df["file_extension"].astype(str).str.lower()

    # Ensure numeric columns are numeric
    df["file_size_bytes"] = pd.to_numeric(
        df["file_size_bytes"],
        errors="coerce"
    ).fillna(0)

    df["line_count"] = pd.to_numeric(
        df["line_count"],
        errors="coerce"
    ).fillna(0)

    # Remove duplicate file records
    df = df.drop_duplicates(
        subset=["repository_name", "file_path"]
    )

    # Convert numeric columns to integers
    df["file_size_bytes"] = df["file_size_bytes"].astype(int)
    df["line_count"] = df["line_count"].astype(int)

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[SAVED] {output_path}")
    print(f"[INFO] Clean source dataset shape: {df.shape}")

    return df


def clean_commit_history_dataset() -> pd.DataFrame:
    """Clean the commit-history dataset."""

    input_path = RAW_DIR / "commit_history_dataset.csv"
    output_path = PROCESSED_DIR / "clean_commit_history_dataset.csv"

    df = pd.read_csv(input_path)

    # Normalize text fields
    df["repository_name"] = df["repository_name"].astype(str).str.strip()
    df["commit_hash"] = df["commit_hash"].astype(str).str.strip()
    df["author_name"] = df["author_name"].astype(str).str.strip()
    df["commit_message"] = df["commit_message"].fillna("").astype(str).str.strip()

    # Normalize dates
    df["commit_date"] = pd.to_datetime(
        df["commit_date"],
        errors="coerce",
        utc=True
    )

    # Convert numerical fields
    numeric_columns = [
        "files_changed",
        "insertions",
        "deletions",
        "total_changes",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0).astype(int)

    # Remove duplicate commits within the same repository
    df = df.drop_duplicates(
        subset=["repository_name", "commit_hash"]
    )

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[SAVED] {output_path}")
    print(f"[INFO] Clean commit dataset shape: {df.shape}")

    return df


def create_text_corpus(
    source_df: pd.DataFrame,
    commit_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a text corpus for Lab 3 tokenization.
    """

    source_text = source_df[
        ["repository_name", "source_code"]
    ].copy()

    source_text["text_type"] = "source_code"
    source_text = source_text.rename(
        columns={"source_code": "text"}
    )

    commit_text = commit_df[
        ["repository_name", "commit_message"]
    ].copy()

    commit_text["text_type"] = "commit_message"
    commit_text = commit_text.rename(
        columns={"commit_message": "text"}
    )

    corpus = pd.concat(
        [source_text, commit_text],
        ignore_index=True
    )

    corpus["text"] = corpus["text"].fillna("").astype(str)

    # Remove completely empty text records
    corpus = corpus[corpus["text"].str.strip() != ""]

    corpus = corpus.reset_index(drop=True)
    corpus.insert(0, "record_id", range(1, len(corpus) + 1))

    output_path = PROCESSED_DIR / "text_corpus.csv"
    corpus.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[SAVED] {output_path}")
    print(f"[INFO] Text corpus shape: {corpus.shape}")

    return corpus


def run_cleaning_pipeline() -> None:
    """Run the complete cleaning pipeline."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    source_df = clean_source_code_dataset()
    commit_df = clean_commit_history_dataset()

    create_text_corpus(source_df, commit_df)

    print("[DONE] Data cleaning pipeline completed successfully.")


if __name__ == "__main__":
    run_cleaning_pipeline()