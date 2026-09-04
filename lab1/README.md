# Lab 1: Software Repository Mining

## Repository Selected

**FastAPI**

GitHub Repository: https://github.com/fastapi/fastapi

FastAPI is an open-source Python web framework. It was selected as the target repository for this lab.

---

## Objective

The objective of this lab is to mine and profile an open-source software repository.

The analysis covers:

- Repository inventory
- File-level dataset generation
- Programming-language distribution
- File-type distribution
- Lines of Code (LOC) analysis
- Largest source-code files
- Git history mining
- Contributor analysis
- Frequently changed files
- Commit activity over time
- Repository-level observations

---

## Tools and Technologies Used

- Python 3
- Git
- GitPython
- Pandas
- Python `pathlib`
- Python `ast` module
- CSV
- JSON

---

## Project Structure

```text
lab1/
├── data/
│   └── file_dataset.csv
├── output/
│   ├── repository_stats.json
│   └── git_history_summary.csv
├── src/
│   ├── config.py
│   ├── file_analyzer.py
│   ├── git_history_miner.py
│   ├── main.py
│   ├── report_writer.py
│   └── repo_manager.py
└── README.md
```

### Source Files

- `config.py` - Stores repository paths and output locations.
- `repo_manager.py` - Checks whether the target repository exists and clones it when required.
- `file_analyzer.py` - Traverses the repository and generates file-level metrics.
- `git_history_miner.py` - Extracts Git commit, contributor, and file-change information.
- `report_writer.py` - Writes JSON and CSV reports.
- `main.py` - Coordinates the complete analysis pipeline.

---

## Analysis Workflow

The complete analysis pipeline follows these steps:

```text
python main.py
       │
       ▼
Check / Clone Repository
       │
       ▼
Analyze Repository Files
       │
       ▼
Mine Git History
       │
       ▼
Generate Reports
       │
       ├──────────────────────┐
       ▼                      ▼
file_dataset.csv       repository_stats.json
                              │
                              ▼
                    git_history_summary.csv
```

---

# Repository Inventory

The repository analysis produced the following results:

| Metric | Result |
|---|---:|
| Total files | 3,139 |
| Source-code files | 1,155 |
| Directories | 431 |
| Total LOC | 459,614 |
| Average LOC per file | 146.42 |
| Average LOC per source file | 99.10 |
| Python methods/functions | 4,967 |

---

## Programming Languages

The following programming languages were identified as source-code languages in the repository:

| Language | Number of Files |
|---|---:|
| Python | 1,138 |
| Shell | 5 |
| HTML | 5 |
| JavaScript | 4 |
| CSS | 3 |
| **Total** | **1,155** |

Python is the dominant programming language in the FastAPI repository.

---

## File-Type Distribution

The complete file-type/language distribution is:

| File Type / Language | Number of Files |
|---|---:|
| Markdown | 1,692 |
| Python | 1,138 |
| Other | 249 |
| YAML | 42 |
| Shell | 5 |
| HTML | 5 |
| JavaScript | 4 |
| CSS | 3 |
| TOML | 1 |
| **Total** | **3,139** |

Markdown files form a significant portion of the repository because the project contains extensive documentation and supporting material.

Documentation and configuration files are included in the file-level dataset but are not counted as source-code files for the source-code metric.

---

# Largest Source-Code Files

The analysis identifies the largest source-code files based on Lines of Code (LOC).

Some of the largest files identified were:

| File | Language | LOC |
|---|---|---:|
| `tests/test_include_router_defaults_overrides.py` | Python | 7,304 |
| `fastapi/routing.py` | Python | 6,447 |
| `fastapi/applications.py` | Python | 4,774 |
| `fastapi/param_functions.py` | Python | See JSON report |

The complete top-10 list is available in:

```text
lab1/output/repository_stats.json
```

---

# File-Level Dataset

The file-level dataset is stored at:

```text
lab1/data/file_dataset.csv
```

Each row represents one repository file and contains the following information:

```text
file_path
language
extension
loc
method_count
size_bytes
```

### Dataset Fields

| Field | Description |
|---|---|
| `file_path` | Relative path of the file inside the repository |
| `language` | Detected programming or file language |
| `extension` | File extension |
| `loc` | Number of lines in the file |
| `method_count` | Number of Python functions/methods |
| `size_bytes` | File size in bytes |

The dataset provides a machine-readable representation of the repository's file-level structure.

---

# Git History Mining

Git history was analyzed using GitPython.

The following results were obtained:

| Metric | Result |
|---|---:|
| Total commits | 7,809 |
| Unique contributors | 912 |
| Average files changed per commit | 4.90 |
| Average files changed per month | 411.05 |
| Total additions | 1,016,081 |
| Total deletions | 557,431 |
| Average additions per commit | 130.12 |
| Average deletions per commit | 71.38 |
| Average additions per month | 10,925.60 |
| Average deletions per month | 5,993.88 |

The Git-history analysis also records:

- Most active contributors
- Most frequently changed files
- Commits per month
- Files changed per month

Detailed Git-history information is stored in:

```text
lab1/output/repository_stats.json
```

A summary of the main Git metrics is stored in:

```text
lab1/output/git_history_summary.csv
```

---

# Git History Methodology

For each commit, the program extracts:

- Commit author
- Commit date
- Files changed
- Number of insertions
- Number of deletions

The collected information is aggregated to calculate:

- Total number of commits
- Unique contributors
- Most active contributors
- Frequently changed files
- Commits per month
- Files changed per month
- Average files changed per commit
- Average files changed per month
- Total additions and deletions
- Average additions and deletions per commit
- Average additions and deletions per month

---

# Repository Analysis Methodology

## File Analysis

The repository was recursively traversed using Python's `pathlib`.

For each file, the program records:

- Relative file path
- File extension
- Detected language
- Lines of code
- File size
- Number of Python methods/functions

Directories that are not relevant to repository analysis are ignored, including:

```text
.git
.venv
venv
__pycache__
.pytest_cache
.mypy_cache
.ruff_cache
node_modules
```

---

## Source-Code Classification

Source-code statistics are calculated separately from general repository files.

The following languages are considered source code for this analysis:

```text
Python
JavaScript
TypeScript
HTML
CSS
Shell
Batch
PowerShell
```

Documentation and configuration formats such as Markdown, YAML, JSON, and TOML remain part of the file-level dataset and file-type distribution but are not counted as source-code files.

This distinction prevents documentation and configuration files from artificially increasing the source-code count.

---

# Generated Outputs

The lab generates the following machine-readable outputs.

## 1. File-Level Dataset

```text
lab1/data/file_dataset.csv
```

Contains file-level repository metrics.

## 2. Repository Statistics

```text
lab1/output/repository_stats.json
```

Contains:

- Repository inventory
- Source-code statistics
- File-type distribution
- Largest source files
- Git-history statistics
- Contributor information
- Frequently changed files
- Monthly Git activity

## 3. Git History Summary

```text
lab1/output/git_history_summary.csv
```

Contains the main Git-history metrics in CSV format.

---

# How to Run

From the project root:

```powershell
..\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
python lab1\src\main.py
```

The program performs the following operations:

1. Checks whether the FastAPI repository has already been cloned.
2. Clones the repository if it does not exist.
3. Traverses and analyzes repository files.
4. Generates the file-level CSV dataset.
5. Mines the Git history.
6. Generates repository statistics in JSON format.
7. Generates a Git-history summary CSV.

---

# Verification

The complete pipeline was executed successfully.

Final execution results:

```text
Files analyzed: 3139
Total commits: 7809
Unique contributors: 912
```

The generated CSV and JSON files were inspected after execution to verify that the required repository and Git-history metrics were produced.

The generated outputs were:

```text
lab1/data/file_dataset.csv
lab1/output/repository_stats.json
lab1/output/git_history_summary.csv
```

---

# Repository-Level Observations

### 1. Python is the dominant source-code language

Python accounts for 1,138 of the 1,155 files classified as source code.

This is consistent with FastAPI being a Python web framework.

### 2. Documentation is a significant part of the repository

The repository contains 1,692 Markdown files. This shows that documentation and supporting material form a substantial part of the repository.

### 3. The repository has a large development history

The repository contains 7,809 commits and 912 unique contributors in the analyzed Git history.

### 4. The repository has experienced substantial code changes

The analyzed history contains:

- 1,016,081 additions
- 557,431 deletions

This indicates substantial development and modification activity over the repository's history.

### 5. Large source files are primarily Python files

The largest source files identified by LOC are primarily Python files, including core FastAPI modules and large test files.

### 6. Repository mining provides information beyond source-code analysis

The file-level analysis provides structural information about the repository, while Git-history mining provides information about project evolution, contributors, and change patterns.

---

# Learning Outcomes

Through this lab, the following concepts were practiced:

- Programmatic repository traversal
- File and directory analysis
- Programming-language identification
- Lines-of-code measurement
- File-level dataset generation
- Git history mining
- Contributor analysis
- Commit-frequency analysis
- Code-change analysis
- Generation of machine-readable reports
- Verification of AI-assisted code

The lab also demonstrated the difference between **source-code analysis** and **software repository mining**. Source-code analysis focuses on the contents of individual source files, while repository mining also considers repository structure, file metadata, and historical Git information.

---

# GenAI Usage Declaration

ChatGPT was used as a supporting tool during this lab for:

- Understanding the repository-mining requirements
- Planning the implementation
- Explaining Python and Git-related concepts
- Generating and reviewing parts of the implementation
- Debugging implementation issues
- Reviewing generated results
- Identifying and resolving a source-code classification issue
- Extending Git-history analysis with monthly statistics

AI-generated code was not accepted blindly.

The implementation was:

1. Reviewed before use.
2. Executed locally.
3. Checked for syntax errors.
4. Tested through the complete analysis pipeline.
5. Verified using the generated CSV and JSON outputs.

During verification, an issue was identified where documentation and configuration files could be incorrectly treated as source-code files. The source-code classification was then corrected so that source-code statistics are separated from documentation and configuration files.

The final responsibility for the implementation, verification, interpretation, and submitted results remains with the student.

---

# Conclusion

This lab demonstrated how an open-source software repository can be systematically analyzed using automated repository-mining techniques.

The completed pipeline successfully analyzed the FastAPI repository, generated a file-level dataset, calculated repository inventory statistics, and mined Git history.

The final outputs provide both structural and historical information about the repository and can serve as a foundation for further software engineering and DevOps analysis.

---