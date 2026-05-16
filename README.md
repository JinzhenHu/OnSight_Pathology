# OnSight Pathology MacOS – Local Development and Build Guide

For documentation, publication details, and pre-built Windows executables, please visit:

https://onsightpathology.github.io/

This document describes how to run OnSight Pathology locally, how to build the application from source, and where to find model training pipelines.

---

# System Requirements

- Operating System: macOS (supports both Intel and Apple Silicon)

---

# Running Locally (Development Mode)

## 1. Create a Virtual Environment

```bash
python -m venv onsight_env
onsight_env\Scripts\activate
```

Confirm Python version:

```bash
python --version
```

Recommended: Python 3.10.9

---

## 2. Install Dependencies

Two dependency files are provided:

| File | 
|------|
| `requirements_mac.txt` |

Install the appropriate version:

GPU:

```bash
pip install -r requirements_mac.txt
```


---

## 3. Launch the Application

```bash
python app.py
```

This launches the PyQt6 desktop application.

No special configuration is required when running directly via Python.

---

# Building the Executable (PyInstaller)

Ensure PyInstaller is installed:

```bash
pip install pyinstaller
```

Two executables must be built:

- Main application (`app.exe`)
- Worker process (`llm_worker_process.exe`)

---

## GPU Build (Default)

```bash
pyinstaller app_mac.spec --noconfirm
pyinstaller llm_worker_process.spec --noconfirm
```

---


## Post-Build Directory Structure

After building:

```
dist/
    app/
    llm_worker_process/
```

Before running `app.exe`, move only:

```
dist/llm_worker_process/llm_worker_process.exe
```

into:

```
dist/app/
```

Do **not** move the `_internal` folder from `llm_worker_process/`.

All required libraries and dependencies are already bundled within:

```
dist/app/_internal/
```

Final structure:

```
dist/
    app/
        app.exe
        llm_worker_process.exe
        _internal/
```

The main application expects the worker executable to reside in the same directory.

---

# Training Pipelines

The repository includes a:

```
training/
```

directory containing scripts used to train the models bundled with OnSight Pathology.

Each model has its own subdirectory within `training/` that contains:

- Data preparation scripts
- Dataset conversion utilities
- Training configuration files
- Training scripts
- Model-specific documentation

Detailed instructions for reproducing model training can be found in the respective `README.md` files within each model’s training directory.

These materials are provided for transparency and reproducibility.

---

# Adding New Models

To integrate additional models into OnSight Pathology:

1. Add a metadata JSON file to: `metadata/`

2. Update `settings.py` to add a dropdown entry referencing the metadata file.

3. Modify `utils.py` to define how the model is initialized and loaded when selected.

4. Create a `process_region.py` file to define how the model performs inference on captured screen frames and how outputs are structured.

Existing model implementations may be used as references.

---

# Citation

If you use OnSight Pathology in research, please cite the associated publication listed at:

https://onsightpathology.github.io/
