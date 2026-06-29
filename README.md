# 🔬 OnSight Pathology – Local Development and Build Guide

<p align="center">
  <img src="OnSight.png" alt="OnSight Pathology" width="100%">
</p>

<p align="center">
  <a href="https://onsightpathology.github.io/"><img src="https://img.shields.io/badge/Website-onsightpathology.github.io-blue?style=flat-square" alt="Website"></a>
  <img src="https://img.shields.io/badge/OS-Windows-0078D6?style=flat-square&logo=windows" alt="Windows">
  <img src="https://img.shields.io/badge/OS-macOS-000000?style=flat-square&logo=apple" alt="macOS">
  <img src="https://img.shields.io/badge/OS-Linux_(Beta)-FCC624?style=flat-square&logo=linux&logoColor=black" alt="Linux">
  <img src="https://img.shields.io/badge/Python-3.11.9-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
</p>

> 📚 **Documentation & Downloads**  
> For full documentation, publication details, and pre-built executables, please visit our official website: **[onsightpathology.github.io](https://onsightpathology.github.io/)**

This document describes how to run OnSight Pathology locally, how to build the application from source on Windows and macOS, and where to find model training pipelines.

## 📦 Downloads

Pre-built packages and additional resources are available through the following platforms:

- **Zenodo (primary, DOI-versioned):**  
  https://zenodo.org/records/20402050

- **Google Drive:**  
  https://drive.google.com/drive/folders/1EnW_5nvWHmMgmszurk0LLtyBAKExFh0p

---

## 💻 System Requirements

OnSight Pathology is designed to be cross-platform.

- **Windows:** Supported for both GPU and CPU builds.
- **macOS:** Supported for both Intel and Apple Silicon (please see the `mac` branch).
- **Linux:** Beta support available for the GPU version (please see the `linux` branch).
- **Hardware (optional):** NVIDIA GPU with CUDA 12.8 support for the GPU build. Apple Silicon GPUs are used via the MPS backend when available.

---

# Running Locally (Development Mode)

## 1. Create a Virtual Environment

```bash
python -m venv onsight_env
onsight_env\Scripts\activate          # Windows
source onsight_env/bin/activate       # macOS / Linux
```

Confirm Python version:

```bash
python --version
```

Recommended: Python 3.11.9

---

## 2. Install Dependencies

Two dependency files are provided:

| File | Description |
|------|-------------|
| `requirements.txt`     | GPU version (CUDA 12.8 PyTorch wheels) |
| `requirements_cpu.txt` | CPU-only version (also used for Apple Silicon / Intel Mac builds) |

GPU:

```bash
pip install -r requirements.txt
```

CPU:

```bash
pip install -r requirements_cpu.txt
```

Device selection at runtime is handled by `device_compat.py`, which transparently picks **CUDA → MPS → CPU** in that order. No code changes are needed when switching machines.

---

## 3. Launch the Application

```bash
python app.py
```

This launches the PyQt6 desktop application. No additional configuration is required when running directly from source.

---

# Building the Executable (PyInstaller)

Two build modes are supported and shared between Windows (`app.spec`) and macOS (`app_mac.spec`):

| Mode | Env var | What it does |
|------|---------|--------------|
| **Local + HF** (default)  | `ONSIGHT_BUILD=local` | Bundles model weights directly into the build. Larger, but works fully offline. |
| **HF-only**               | `ONSIGHT_BUILD=hf`    | No bundled weights. Models are downloaded from HuggingFace Hub on first launch. |

On Windows, you can also choose the output format:

| Format | Env var | Notes |
|--------|---------|-------|
| **onedir** (default for `local`) | `ONSIGHT_FORMAT=onedir`  | Folder of files. Wrapped by Inno Setup into an installer. |
| **onefile** (default for `hf`)   | `ONSIGHT_FORMAT=onefile` | Single self-extracting `.exe`. Easy to share. |

Ensure PyInstaller is installed:

```bash
pip install pyinstaller
```

---

## Windows Build

### Local + HF build (bundled weights, recommended)

PowerShell:

```powershell
$env:ONSIGHT_BUILD="local"
pyinstaller app.spec --noconfirm
```

Command Prompt:

```cmd
set ONSIGHT_BUILD=local
pyinstaller app.spec --noconfirm
```

Output: `dist\app_local\`

### HF-only build (online download, single exe)

```powershell
$env:ONSIGHT_BUILD="hf"
pyinstaller app.spec --noconfirm
```

Output: `dist\OnSight_HF.exe`

### CPU-only build

For machines without an NVIDIA GPU, set `BUILD_TYPE=CPU` before building. This prevents PyTorch from probing for CUDA at startup, which otherwise causes DLL discovery crashes on CPU-only systems (including older GPUs with unsupported compute capabilities, e.g. GTX 1050 Ti / sm_61).

```powershell
$env:BUILD_TYPE="CPU"
$env:ONSIGHT_BUILD="local"
pyinstaller app.spec --noconfirm
```

---

## macOS Build

The full Mac build pipeline — PyInstaller → ad-hoc codesign → DMG packaging — is automated in `build_mac.sh`:

```bash
chmod +x build_mac.sh
./build_mac.sh
```

This script performs:

1. Clean previous `build/` and `dist/`.
2. Run PyInstaller with `app_mac.spec` (defaults to `ONSIGHT_BUILD=local`).
3. Sign every inner `.dylib` / `.so` with the ad-hoc identity.
4. Ad-hoc sign the full `.app` bundle with `onsight.entitlements`.
5. Build a distributable DMG via `create-dmg`.
6. Sign the DMG itself.

Outputs:

```
dist/OnSightPathology_App.app
dist/OnSight-<version>.dmg
```

For an HF-only Mac build, set `ONSIGHT_BUILD=hf` before invoking the script (or run PyInstaller directly with `app_mac.spec`).

> **Note on Gatekeeper:** Because the app is ad-hoc signed (not Apple-notarized), downloaded DMGs carry the `com.apple.quarantine` attribute, which triggers per-dylib validation on recent macOS versions and can prevent launch. Distribute an `install_onsight.command` helper alongside the DMG that strips quarantine and copies the app into `/Applications`.

macOS-specific runtime permission handling (Screen Recording, Accessibility, Input Monitoring) is centralized in `mac_permissions.py` and surfaced through `MacPermissionDialog.py`.

---

# Creating a Windows Installer (Inno Setup)

OnSight Pathology can be packaged using Inno Setup 6.5 or newer.

Download: https://jrsoftware.org/isdl.php

Two installer scripts are provided:

| Script              | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `installer.iss`     | Wraps the onedir build (`dist\app_local\` or `dist\app_hf\`). Supports a `BuildMode` define to switch between the two. Includes 2 GB disk-spanning for large local builds. |
| `installer_hf.iss`  | Wraps the single-file `OnSight_HF.exe`. Smaller, no disk spanning.       |

### Building from the onedir build

Default (local with bundled weights):

```cmd
iscc installer.iss
```

HF-only onedir mode:

```cmd
iscc installer.iss /DBuildMode=hf
```

### Building from the single-exe HF build

```cmd
iscc installer_hf.iss
```

Compiled installers are written to `output/`.

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

Detailed instructions for reproducing model training can be found in the respective `README.md` files within each model's training directory. These materials are provided for transparency and reproducibility.

---

# Adding New Models

To integrate additional models into OnSight Pathology:

1. Add a metadata JSON file to: `metadata/` (see existing files such as `cell_vit.json`, `mib_yolo_1024.json`, `lingshu.json` for reference).
2. Update `settings.py` to add a dropdown entry referencing the metadata file.
3. Modify `utils.py` to define how the model is initialized and loaded when selected.
4. Create a `process_region_*.py` file to define how the model performs inference on captured screen frames and how outputs are structured.

Existing implementations (`process_region_cellpose.py`, `process_region_cellvit.py`, `process_region_YOLO.py`, etc.) may be used as references.

---

# Acknowledgements

The project was built on top of excellent open-source repositories including MIDOG++, CellViT, Cellpose, and the Lingshu medical VLM. We thank the authors and developers for their contributions.

---

# Citation

If you use OnSight Pathology in research, please cite the associated publication listed at:

https://onsightpathology.github.io/
