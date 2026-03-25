# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python CLI tool for visualizing PtyRAD (ptychographic reconstruction) data. Loads `.pt` (PyTorch) and `.hdf5` model files, provides interactive matplotlib-based 3D visualization with layer navigation/rotation/cropping, and exports PNG images, MP4 videos, and MAT files.

## Build & Run

```bash
# Install (editable, into conda env "ptyrad")
pip install -e .

# Run CLI
plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.pt
plot_ptyrad --folder /path/to/parent_folder --file model_iter1000.hdf5 --force

# Dev shortcut (no install needed)
python cli.py --folder ... --file ...
```

No automated tests exist. Validate changes by running the CLI end-to-end with a sample dataset. If adding tests, use `pytest` in a `tests/` directory.

## Architecture

Flat module structure at repo root (~1400 lines across 8 modules). Data flow:

```
cli.py (entry point, arg parsing)
  → file_utils.py (discover model files in region subdirs)
  → data_processor.py (load .pt/.hdf5, extract tensors, FFT/transforms)
  → interactive_plotter.py (main UI controller, event handling)
      → ui_components.py (sliders, buttons, selectors)
      → video_generator.py (frame generation, MP4 encoding via imageio-ffmpeg)
  → config.py (global constants, UI layout, matplotlib settings)
```

**Key patterns:**
- `DataProcessor.load_model_file()` abstracts .pt vs .hdf5 format differences
- `DataProcessor.render_view()` is the shared rendering pipeline (transform → FFT/real-space → labels) used by interactive display, image saving, and video generation
- Global `PROCESSING_STATE` dict tracks user interaction (next_region, end_processing) for interruption between regions
- `plot_params.json` persists per-region visualization settings across runs (includes crop center offsets)
- `batch_save_videos_and_mat()` in video_generator.py handles batch video+MAT generation for all regions
- Caching: skips regions where PNG/MP4 already exist (overridden by `--force`)

**Expected input structure:** Parent folder containing region subdirectories, each with a nested model file (e.g., `4Dregion01/.../model_iter1000.pt`).

**Output structure:** `Data_Saved/<region_name>/` containing `*.png`, `*.mp4`, `*.mat`, plus `plot_params.json` at parent level.

## Coding Conventions

- 4-space indentation, UTF-8 encoding
- `snake_case` functions/variables, `CapWords` classes
- No formatter/linter configured; keep changes minimal and readable
- Commit style: short prefixes like `fix:`, `Added:`, `Update:`, `Improve:`

## Key Dependencies

torch, h5py, numpy, matplotlib, scipy, imageio, imageio-ffmpeg, PyYAML, tqdm (Python >= 3.8)
