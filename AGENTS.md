# AGENT Instructions

## Code style
- Use 4 spaces for indentation in Python files.
- Prefer single quotes for strings where practical.
- Keep existing formatting; lines can exceed 80 characters if already long.
- Add docstrings to any new public functions or classes.

## Testing
- This repository has no formal test suite. Before committing, run a syntax check on all Python files:
  ```bash
  python -m py_compile $(git ls-files '*.py')
  ```
  The command should exit without errors.
- Optionally run the training scripts described in the README when data is available.

## Running the project
- Data must be downloaded separately. See `README.md` for links.
- Run image experiments with `python main_image.py --dataset <dataset_name>`.
- Run text experiments with `python main_text.py --dataset <dataset_name>`.

## Pull request guidance
- Summaries should briefly describe major changes.
- Always include the result of the syntax check or other tests in the PR description.
