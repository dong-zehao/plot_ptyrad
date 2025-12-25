# Repository Guidelines

These guidelines describe how to work with the `plot_ptyrad` CLI and codebase.

## Project Structure & Module Organization
- Source is flat at the repository root (no package folder). Key modules include `cli.py` (entry point), `plot_pt_file.py` (file discovery/processing), `data_processor.py`, `interactive_plotter.py`, `ui_components.py`, `video_generator.py`, `file_utils.py`, and `config.py`.
- Outputs are written under the user-specified parent folder as `Data_Saved/<region_name>/` (PNG/MP4/MAT artifacts).
- Documentation and usage examples live in `README.md` (Chinese).

## Build, Test, and Development Commands
- `pip install -e .` installs the editable CLI entry point `plot_ptyrad`.
- `plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.pt` processes matching `.pt` files and writes results to `Data_Saved/`.
- `plot_ptyrad --force ...` reprocesses data even if outputs already exist.
- Dev shortcut: `python cli.py --folder ... --file ...` runs the CLI without installing.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and UTF-8 encoding.
- Follow existing patterns: `snake_case` for functions/variables, `CapWords` for classes, and module names matching their file (`video_generator.py`, etc.).
- No formatter or linter is configured; keep changes minimal and readable, and add brief docstrings for new public functions.

## Testing Guidelines
- No automated tests or coverage targets are configured in this repository.
- If you add tests, prefer `pytest` and place them in `tests/` with `test_*.py` naming.
- Validate changes with a small sample dataset by running the CLI end-to-end.

## Commit & Pull Request Guidelines
- Recent commits use short prefixes like `fix:`, `Added:`, `Update`, or `Improve:` followed by a concise summary. Match that style.
- PRs should include: a clear description of the change, how to run it locally, and (when applicable) sample output paths or screenshots of generated plots.

## Configuration & Data Tips
- The CLI expects a parent folder containing region subfolders with nested `.pt` files; keep that structure intact.
- Avoid committing generated outputs under `Data_Saved/`.
