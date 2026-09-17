# DevOps Lab 2: Multi-Repository Software Dataset Mining

## 1. Objective

The objective of this lab is to mine multiple open-source software repositories and prepare structured datasets from their source code and Git commit histories.

The lab includes:

- Repository management
- Source-code mining
- Commit-history mining
- Data cleaning and preprocessing
- Text-corpus preparation
- Meaningful dataset merging
- Statistical analysis
- Documentation of dataset uses and limitations

Five open-source repositories were analyzed independently.

---

## 2. Repositories Used

The following repositories were used:

1. FastAPI
2. Flask
3. Pytest
4. Requests
5. Scikit-learn

Each repository was processed separately to maintain repository-specific information.

---

## 3. Project Structure

```text
lab2/
├── src/
│   ├── config.py
│   ├── repo_manager.py
│   ├── source_miner.py
│   ├── commit_miner.py
│   ├── data_cleaner.py
│   ├── dataset_merger.py
│   └── generate_statistics.py
│
├── data/
│   ├── raw/
│   │   ├── source_code_dataset.csv
│   │   └── commit_history_dataset.csv
│   │
│   └── processed/
│       ├── clean_source_code_dataset.csv
│       ├── clean_commit_history_dataset.csv
│       ├── text_corpus.csv
│       └── combined_repository_dataset.csv
│
└── output/
    ├── dataset_statistics.csv
    └── dataset_statistics_report.txt
```

---

## 4. Repository Management

The repository manager checks whether the required repositories already exist locally.

If a repository exists, it is reused. Otherwise, it is cloned using GitPython.

This avoids unnecessary repeated cloning and ensures that all five repositories are available for analysis.

### Main responsibilities

- Maintain repository URLs
- Create the repository storage directory
- Clone missing repositories
- Reuse already downloaded repositories
- Return local repository paths

---

## 5. Source-Code Dataset

The source-code mining process scans each repository and extracts relevant source files.

### Attributes extracted

- `repository_name`
- `file_path`
- `file_extension`
- `file_size_bytes`
- `line_count`
- `source_code`

The miner ignores irrelevant directories and files such as:

- `.git`
- Cache directories
- Build directories
- Distribution directories
- Virtual environments
- Other generated files

### Source-code dataset summary

- Total repositories: **5**
- Total source files: **2,635**
- Total lines of code: **721,510**

The raw source-code dataset is stored at:

```text
lab2/data/raw/source_code_dataset.csv
```

The cleaned source-code dataset is stored at:

```text
lab2/data/processed/clean_source_code_dataset.csv
```

---

## 6. Commit-History Dataset

The commit-history mining process uses GitPython to inspect the Git history of every repository.

For every commit, information about the commit and its changes is collected.

### Attributes extracted

- `repository_name`
- `commit_hash`
- `author_name`
- `commit_date`
- `commit_message`
- `files_changed`
- `insertions`
- `deletions`
- `total_changes`

### Commit-history dataset summary

- Total repositories: **5**
- Total commits: **71,544**
- Total insertions: **15,732,042**
- Total deletions: **12,769,636**

The raw commit-history dataset is stored at:

```text
lab2/data/raw/commit_history_dataset.csv
```

The cleaned commit-history dataset is stored at:

```text
lab2/data/processed/clean_commit_history_dataset.csv
```

---

## 7. Data Cleaning and Preprocessing

The cleaning pipeline prepares the raw datasets for reliable analysis and future machine-learning tasks.

### Cleaning operations performed

- Handling missing source-code values
- Filling empty source-code values where required
- Normalizing text fields
- Converting numerical columns to numeric data types
- Converting commit dates into consistent datetime values
- Removing duplicate source-file records
- Removing duplicate commit records
- Removing empty text records from the text corpus
- Saving cleaned datasets separately

The cleaning process ensures that the datasets are more consistent and suitable for statistical analysis.

---

## 8. Text Corpus Preparation

A separate text corpus was created for future Natural Language Processing and tokenization tasks.

The corpus combines textual information from:

1. Source-code files
2. Git commit messages

Each corpus record contains:

- `record_id`
- `repository_name`
- `text_type`
- `text`

The `text_type` field identifies whether the record contains source code or a commit message.

The text corpus is stored at:

```text
lab2/data/processed/text_corpus.csv
```

This dataset can be used in future labs for:

- Character-level tokenization
- Word-level tokenization
- Subword tokenization
- Vocabulary analysis
- Embedding generation
- Similarity analysis
- Natural Language Processing experiments

---

## 9. Combined Repository Dataset

The source-code and commit-history datasets were combined into a third dataset.

A direct join between every source file and every commit was avoided because it would create a Cartesian product. Such a join would produce a very large dataset without representing a meaningful relationship between individual files and commits.

Instead, a repository-level aggregation approach was used.

### Combination process

1. Source-code data was grouped by repository.
2. Source-code metrics were calculated for each repository.
3. Commit-history data was grouped by repository.
4. Commit-related metrics were calculated for each repository.
5. Both summaries were merged using `repository_name`.

### Source-code metrics included

- Number of source files
- Total lines of code
- Total source size
- Average file size
- Average lines per file
- Source-file extensions

### Commit-history metrics included

- Number of commits
- Total insertions
- Total deletions
- Total changes
- Average files changed per commit
- Average changes per commit
- Number of unique authors
- First commit date
- Last commit date

### Combined dataset summary

- Total records: **5**
- Total repositories: **5**
- Total source files represented: **2,635**
- Total commits represented: **71,544**
- Total lines of code represented: **721,510**
- Total code changes: **28,501,678**

The combined dataset is stored at:

```text
lab2/data/processed/combined_repository_dataset.csv
```

---

## 10. Statistical Analysis

Three statistical summaries were generated, one for each major dataset.

### Dataset 1: Clean Source-Code Dataset

Statistics include:

- Total records
- Total repositories
- Total lines of code
- Average lines per file
- Largest file size

### Dataset 2: Clean Commit-History Dataset

Statistics include:

- Total records
- Total repositories
- Total insertions
- Total deletions
- Average files changed per commit
- Number of unique authors

### Dataset 3: Combined Repository Dataset

Statistics include:

- Total records
- Total repositories
- Total source files
- Total commits
- Total lines of code
- Total code changes

The generated files are:

```text
lab2/output/dataset_statistics.csv
lab2/output/dataset_statistics_report.txt
```

---

## 11. Dataset Uses

The prepared datasets can be used for several software engineering and research applications.

### Possible uses

- Software repository analysis
- Code-size analysis
- Repository activity comparison
- Developer contribution analysis
- Commit-pattern analysis
- Software evolution studies
- Code maintenance research
- Predictive maintenance
- Software quality research
- Machine-learning dataset preparation
- Source-code tokenization
- Natural Language Processing
- Commit-message classification
- Developer activity modeling
- Repository complexity analysis

The text corpus can also support future experiments involving tokenizers and embeddings.

---

## 12. Dataset Limitations and Problems

Although the datasets are useful, they have several limitations.

### 12.1 Repository size differences

The repositories vary significantly in size. Direct comparison may therefore be misleading without normalization.

### 12.2 Author identity inconsistency

The same developer may use different names or email identities across commits. Similarly, different people may have similar author names.

### 12.3 Commit messages are inconsistent

Commit messages differ in:

- Length
- Writing style
- Level of detail
- Use of abbreviations
- Use of technical terminology

### 12.4 Lines of code do not represent code quality

A larger number of lines does not necessarily indicate better or more complex software.

### 12.5 Generated files and configuration files

Some files may not represent core application logic, even after filtering.

### 12.6 Commit activity does not equal software quality

A repository with more commits is not automatically better maintained or higher quality.

### 12.7 Loss of fine-grained relationships

The combined dataset is aggregated at repository level. Therefore, it does not preserve the exact relationship between an individual source file and the commits that modified it.

### 12.8 Historical data limitations

The dataset represents the available Git history and may not capture discussions, issue-tracker information, pull requests, or the complete reason behind every code change.

---

## 13. Execution Commands

The scripts can be executed from the project root using the following commands:

```powershell
python .\lab2\src\repo_manager.py
python .\lab2\src\source_miner.py
python .\lab2\src\commit_miner.py
python .\lab2\src\data_cleaner.py
python .\lab2\src\dataset_merger.py
python .\lab2\src\generate_statistics.py
```

---

## 14. Tools and Libraries Used

### Python

Used as the primary programming language for implementing the mining and preprocessing pipeline.

### GitPython

Used to:

- Clone repositories
- Access Git repositories
- Iterate through commits
- Extract commit metadata
- Read commit statistics

### Pandas

Used to:

- Create structured datasets
- Read and write CSV files
- Clean missing values
- Remove duplicates
- Group and aggregate data
- Merge datasets
- Generate statistics

### Git and GitHub

Used for:

- Version control
- Tracking project milestones
- Committing source code and datasets
- Pushing the completed work to GitHub

---

## 15. Version-Control Milestones

The project was developed through meaningful incremental commits:

- Initialize Lab 2 structure and configuration
- Add multi-repository management
- Implement source-code mining
- Implement commit-history mining
- Add dataset cleaning pipeline
- Create combined repository dataset
- Generate dataset statistics reports
- Add Lab 2 documentation

Each milestone represents a significant stage of the data-mining pipeline.

---

## 16. Conclusion

This lab successfully mined five open-source repositories and generated structured datasets containing source-code information and Git commit-history information.

The complete pipeline includes:

1. Repository acquisition
2. Source-code extraction
3. Commit-history extraction
4. Data cleaning
5. Text-corpus creation
6. Repository-level dataset merging
7. Statistical analysis
8. Documentation

The final datasets can be used for software analytics, repository evolution studies, machine-learning preparation, and future tokenization and embedding experiments.